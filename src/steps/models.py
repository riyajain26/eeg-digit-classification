"""
Step 4 orchestration: model training, evaluation, and permutation testing.

Trains + evaluates whichever model cfg.model.model_name names (classical
or any EEGNet variant, for either task) via run_model_training(). This is
the single entry point - it branches on cfg.model.is_deep to pick
classical vs deep training/evaluation, because sklearn's fit()/predict()
and PyTorch's training loop are genuinely different APIs. That's the ONE
place this module still branches on model family; which MODEL to build
within either branch is fully registry-driven (see src/models/factory.py) -
model_name alone identifies it, no separate "stage" or "variant" needed.
"""

import json

import h5py
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig
from src.evaluation.metrics import evaluate_sklearn_model
from src.evaluation.permutation_test import permutation_test_sklearn, permutation_test_torch
from src.models.factory import MODEL_REGISTRY


def _load_classical_features(cfg: PipelineConfig):
    """Loads Step 3's extracted features, applies trial_concern exclusion
    (from Step 2's filtered output), and filters to digit-only trials if
    task='multiclass'. Returns (X_train, y_train, X_val, y_val, average)."""
    with h5py.File(cfg.data.features_dir / "train_features.h5", "r") as f:
        X_train, y_bin_train, y_digit_train = f["features"][:], f["label_binary"][:], f["label_digit"][:]
    with h5py.File(cfg.data.features_dir / "val_features.h5", "r") as f:
        X_val, y_bin_val, y_digit_val = f["features"][:], f["label_binary"][:], f["label_digit"][:]
    with h5py.File(cfg.data.filtered_dir / "train_filtered.h5", "r") as f:
        train_concern = f["trial_concern"][:]
    with h5py.File(cfg.data.filtered_dir / "val_filtered.h5", "r") as f:
        val_concern = f["trial_concern"][:]

    keep_train, keep_val = ~train_concern, ~val_concern

    if cfg.model.task == "multiclass":
        keep_train &= (y_digit_train != -1)
        keep_val &= (y_digit_val != -1)
        y_train, y_val, average = y_digit_train[keep_train], y_digit_val[keep_val], "macro"
    else:
        y_train, y_val, average = y_bin_train[keep_train], y_bin_val[keep_val], "binary"

    return X_train[keep_train], y_train, X_val[keep_val], y_val, average


def _load_deep_lazy_indices(cfg: PipelineConfig):
    """
    Returns (train_h5_path, train_indices, y_train, val_h5_path, val_indices, y_val)
    instead of full in-memory arrays - only labels and trial_concern flags
    (small) are loaded; the actual EEG data stays on disk and is read
    per-trial by LazyEEGDataset during training. Required at 100% scale,
    where the full filtered eeg array (~13GB) crashed Colab outright.
    """
    train_path = cfg.data.filtered_dir / "train_filtered.h5"
    val_path = cfg.data.filtered_dir / "val_filtered.h5"

    with h5py.File(train_path, "r") as f:
        y_bin_train = f["label_binary"][:]
        y_digit_train = f["label_digit"][:]
        train_concern = f["trial_concern"][:]
    with h5py.File(val_path, "r") as f:
        y_bin_val = f["label_binary"][:]
        y_digit_val = f["label_digit"][:]
        val_concern = f["trial_concern"][:]

    keep_train, keep_val = ~train_concern, ~val_concern
    if cfg.model.task == "multiclass":
        keep_train &= (y_digit_train != -1)
        keep_val &= (y_digit_val != -1)
        y_train_full, y_val_full = y_digit_train, y_digit_val
    else:
        y_train_full, y_val_full = y_bin_train, y_bin_val

    train_indices = np.where(keep_train)[0]
    val_indices = np.where(keep_val)[0]
    return train_path, train_indices, y_train_full[train_indices], val_path, val_indices, y_val_full[val_indices]


def _save_results(cfg: PipelineConfig, results: dict) -> None:
    results_dir = cfg.model.results_dir(cfg.data.variant_tag)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_dir / 'results.json'}")


def run_classical_model(cfg: PipelineConfig, run_permutation: bool = True) -> dict:
    """Trains + evaluates a classical model (LDA/SVM/RandomForest) end to
    end: load features, scale, fit, evaluate, checkpoint, permutation test."""
    X_train, y_train, X_val, y_val, average = _load_classical_features(cfg)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    builder = MODEL_REGISTRY.get(cfg.model.model_name)
    model = builder.build(cfg, n_channels=None, n_samples=None, n_classes=None)
    print(f"Models [{cfg.model.task}/{cfg.model.model_name}]: training...")
    metrics, cm, fitted_model = evaluate_sklearn_model(
        cfg.model.model_name, model, X_train_scaled, y_train, X_val_scaled, y_val, average
    )
    print(metrics)

    checkpoint_path = cfg.model.checkpoint_path(cfg.data.variant_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted_model, "scaler": scaler}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    permutation_result = None
    if run_permutation:
        print("Running permutation test...")
        n = cfg.permutation.classical_subsample_size
        if n is not None and n < len(X_train_scaled):
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(len(X_train_scaled), size=n, replace=False)
            X_perm, y_perm = X_train_scaled[idx], y_train[idx]
        else:
            X_perm, y_perm = X_train_scaled, y_train

        perm_build = builder.build_for_permutation or builder.build
        model_fn = lambda: perm_build(cfg, n_channels=None, n_samples=None, n_classes=None)
        real_acc, shuffled_accs = permutation_test_sklearn(
            model_fn, X_perm, y_perm, X_val_scaled, y_val,
            n_permutations=cfg.permutation.n_permutations_classical, seed=cfg.seed,
        )
        gap = real_acc - shuffled_accs.mean()
        permutation_result = {
            "real_accuracy": real_acc, "shuffled_mean": shuffled_accs.mean(),
            "shuffled_std": shuffled_accs.std(),
            "gap": gap, "gap_over_std": gap / shuffled_accs.std() if shuffled_accs.std() > 0 else None,
        }
    else:
        print("Skipping permutation test (run_permutation=False).")

    results = {
        "config": {"variant_tag": cfg.data.variant_tag, "task": cfg.model.task, "model": cfg.model.model_name},
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "permutation_test": permutation_result,
    }
    _save_results(cfg, results)
    return results


def run_deep_model(cfg: PipelineConfig, run_permutation: bool = True) -> dict:
    """Trains + evaluates any EEGNet variant (for either task) end to end:
    lazy-load indices, train (single-phase, or eegnet_finetuned_backbone_reuse's
    two-phase), evaluate, permutation test. Requires PyTorch - imported
    here (not at module level) so classical-only usage of this file never
    needs it installed."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.training.loop import (make_lazy_loaders, train_with_checkpointing,
                                     train_with_two_phase_finetuning, run_epoch, EEGDataset)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_h5_path, train_indices, y_train, val_h5_path, val_indices, y_val = _load_deep_lazy_indices(cfg)

    with h5py.File(train_h5_path, "r") as f:
        n_channels, n_samples = f["eeg"].shape[1], f["eeg"].shape[2]

    n_classes = cfg.model.n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Models [{cfg.model.task}/{cfg.model.model_name}]: using device {device}, "
          f"train={len(train_indices)}, val={len(val_indices)} trials")
    torch.manual_seed(cfg.seed)

    train_loader, val_loader = make_lazy_loaders(train_h5_path, train_indices, y_train,
                                                   val_h5_path, val_indices, y_val, cfg.training.batch_size)

    builder = MODEL_REGISTRY.get(cfg.model.model_name)
    model = builder.build(cfg, n_channels, n_samples, n_classes).to(device)

    checkpoint_path = cfg.model.checkpoint_path(cfg.data.variant_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.model.model_name == "eegnet_finetuned_backbone_reuse":
        p = cfg.model.eegnet
        history = train_with_two_phase_finetuning(
            model, train_loader, val_loader, checkpoint_path, device,
            freeze_epochs=p.reuse_freeze_epochs, finetune_epochs=p.reuse_finetune_epochs,
            base_learning_rate=cfg.training.learning_rate,
            finetune_lr_multiplier=p.reuse_finetune_lr_multiplier,
            early_stop_patience=cfg.training.early_stop_patience,
        )
    else:
        # eegnet_fresh, eegnet_frozen_backbone_reuse, and
        # eegnet_dual_backbone_auxiliary_input (whose frozen half has
        # requires_grad=False, so train_with_checkpointing's optimizer
        # correctly only updates the trainable parts) all use standard
        # single-phase training.
        history = train_with_checkpointing(
            model, train_loader, val_loader, checkpoint_path, device,
            cfg.training.n_epochs, cfg.training.learning_rate, cfg.training.early_stop_patience,
        )

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    average = "binary" if cfg.model.task == "binary" else "macro"
    metrics = {
        "model": cfg.model.model_name,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
        "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
        "f1": f1_score(all_labels, all_preds, average=average, zero_division=0),
    }
    print(metrics)

    permutation_result = None
    if run_permutation:
        print("Running (reduced-scale) permutation test...")
        # Reads ONLY the permutation subsample's rows from disk (h5py fancy
        # indexing), never the full train array - same lazy principle as
        # the main training loop, just a smaller in-memory-safe subsample
        # since permutation testing is already deliberately reduced-scale.
        rng = np.random.default_rng(cfg.seed)
        perm_size = min(cfg.permutation.deep_subsample_size, len(train_indices))
        perm_idx_into_indices = rng.choice(len(train_indices), size=perm_size, replace=False)
        perm_rows = np.sort(train_indices[perm_idx_into_indices])
        y_perm = y_train[np.searchsorted(train_indices, perm_rows)]

        with h5py.File(train_h5_path, "r") as f:
            eeg_perm_sub = f["eeg"][perm_rows]

        # Uses builder.build_for_permutation (a plain eegnet_fresh build,
        # see src/models/factory.py) rather than replicating the full
        # backbone-reuse procedure for every shuffled-label run - a rough
        # noise-floor check doesn't need to be architecturally identical
        # to the real model.
        perm_build = builder.build_for_permutation or builder.build

        def train_fn(eeg_sub, y_sub):
            m = perm_build(cfg, n_channels, n_samples, n_classes).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=cfg.training.learning_rate)
            crit = nn.CrossEntropyLoss()
            loader = DataLoader(EEGDataset(eeg_sub, y_sub), batch_size=cfg.training.batch_size, shuffle=True)
            for _ in range(cfg.permutation.deep_quick_epochs):
                run_epoch(m, loader, opt, crit, device, train=True)
            return m

        def eval_fn(m, loader):
            m.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    logits = m(X_batch.to(device))
                    preds.extend(logits.argmax(dim=1).cpu().numpy())
                    labels.extend(y_batch.numpy())
            return accuracy_score(labels, preds)

        real_acc, shuffled_accs = permutation_test_torch(
            train_fn, eval_fn, eeg_perm_sub, y_perm, val_loader,
            n_permutations=cfg.permutation.n_permutations_deep,
            subsample_size=perm_size, seed=cfg.seed,
        )
        gap = real_acc - shuffled_accs.mean()
        permutation_result = {
            "real_accuracy": real_acc, "shuffled_mean": shuffled_accs.mean(),
            "shuffled_std": shuffled_accs.std(),
            "gap": gap, "gap_over_std": gap / shuffled_accs.std() if shuffled_accs.std() > 0 else None,
        }
    else:
        print("Skipping permutation test (run_permutation=False).")

    results = {
        "config": {"variant_tag": cfg.data.variant_tag, "task": cfg.model.task, "model": cfg.model.model_name},
        "metrics": metrics,
        "best_val_acc": history["best_val_acc"],
        "permutation_test": permutation_result,
    }
    _save_results(cfg, results)
    return results


def run_model_training(cfg: PipelineConfig, run_permutation: bool = True) -> dict:
    """
    Full Step 4 entry point: trains + evaluates whichever model
    cfg.model.model_name names, branching to the classical or deep path.
    Requires Step 2 (preprocessing) for deep models, or Steps 2+3
    (preprocessing + features) for classical models, to have completed
    first for the same dataset_variant/subsample_fraction. Any
    backbone-reuse model additionally requires a trained eegnet_fresh
    checkpoint for cfg.model.eegnet.reuse_source_task to already exist
    (see src/models/factory.py's _load_reuse_source_backbone).
    """
    if cfg.model.is_deep:
        return run_deep_model(cfg, run_permutation=run_permutation)
    return run_classical_model(cfg, run_permutation=run_permutation)
