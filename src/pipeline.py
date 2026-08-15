"""
End-to-end orchestration, phase numbers matching roadmap.md:
Phase 2 (data acquisition) -> Phase 3 (splitting) -> Phase 4 (preprocessing
+ Path A features) -> Phase 5/6 (train + evaluate whichever model was
selected, classical or deep) + permutation test.

This file contains NO algorithm logic of its own - it only calls into
data/, preprocessing/, features/, models/, training/, and evaluation/, in
the right order, with the right arguments, based on the config produced by
build_config().

Scaling/staging/model-selection philosophy (per project decision):
- Changing dataset scale = change subsample_fraction. No code below branches
  on scale explicitly - it's entirely handled by config.py's derived paths
  and target_per_class.
- Adding Stage 2 = change stage. Only _load_stage_data() branches on it -
  every phase before modeling is stage-agnostic (processes ALL trials the
  same way regardless of which stage will eventually consume them).
- Switching models = change model_name. Only the model-training phase
  branches on cfg.model.is_deep - phases 2-4 never reference model_name at all.
"""

from pathlib import Path

import h5py
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PipelineConfig, build_config
from src.data.conversion import extract_split
from src.data.splitting import (check_session_overlap, assign_block_splits,
                                  validate_split, materialize_split)
from src.preprocessing.filters import apply_filters
from src.preprocessing.normalization import (fit_normalization_robust, apply_normalization,
                                                save_normalization_params, load_normalization_params)
from src.preprocessing.artifacts import (detect_bad_channels, exclude_channels,
                                            compute_artifact_arrays, derive_thresholds,
                                            flag_artifacts, save_artifact_params, load_artifact_params)
from src.features.extraction import extract_features_batch
from src.models.factory import build_classical_model, build_classical_model_for_permutation, build_eegnet
from src.evaluation.metrics import evaluate_sklearn_model
from src.evaluation.permutation_test import permutation_test_sklearn, permutation_test_torch


def _exists(path: Path) -> bool:
    return path.exists()


# ---------------------------------------------------------------------------
# Phase 2: Data Acquisition (HF streaming happens inside extract_split, below)
# ---------------------------------------------------------------------------

def run_phase2_data_acquisition(cfg: PipelineConfig, force: bool = False) -> None:
    cfg.data.interim_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.splits_dir.mkdir(parents=True, exist_ok=True)

    train_pool_path = cfg.data.interim_dir / "train_pool.h5"
    if force or not _exists(train_pool_path):
        print(f"Phase 2 [{cfg.data.variant_tag}]: extracting train pool "
              f"(target_per_class={cfg.data.target_per_class})...")
        counts = extract_split(
            output_path=train_pool_path,
            hf_dataset_name=cfg.data.hf_dataset_name,
            hf_split="train",
            target_per_class=cfg.data.target_per_class,
            n_channels=cfg.data.n_channels_nominal,
            n_samples=cfg.data.n_samples,
            max_stream_multiplier=cfg.data.max_stream_multiplier,
        )
        print(f"Train pool counts: {counts}")
    else:
        print(f"Phase 2: train pool already exists at {train_pool_path}, skipping.")

    test_path = cfg.data.splits_dir / "test.h5"
    if force or not _exists(test_path):
        print(f"Phase 2 [{cfg.data.variant_tag}]: extracting test set "
              f"(target_per_class={cfg.data.test_target_per_class})...")
        counts = extract_split(
            output_path=test_path,
            hf_dataset_name=cfg.data.hf_dataset_name,
            hf_split="test",
            target_per_class=cfg.data.test_target_per_class,
            n_channels=cfg.data.n_channels_nominal,
            n_samples=cfg.data.n_samples,
            max_stream_multiplier=cfg.data.max_stream_multiplier,
        )
        print(f"Test set counts: {counts}")
    else:
        print(f"Phase 2: test set already exists at {test_path}, skipping.")


# ---------------------------------------------------------------------------
# Phase 3: Leakage-Safe Splitting
# ---------------------------------------------------------------------------

def run_phase3_splitting(cfg: PipelineConfig, force: bool = False) -> None:
    train_path = cfg.data.splits_dir / "train.h5"
    val_path = cfg.data.splits_dir / "val.h5"

    if not force and _exists(train_path) and _exists(val_path):
        print("Phase 3: train/val splits already exist, skipping.")
        return

    train_pool_path = cfg.data.interim_dir / "train_pool.h5"
    test_path = cfg.data.splits_dir / "test.h5"

    overlap = check_session_overlap(train_pool_path, test_path)
    if overlap:
        raise RuntimeError(
            f"Session overlap found: {overlap}. HF's train/test boundary is NOT "
            f"leakage-safe for dataset_variant={cfg.data.dataset_variant!r} - resolve "
            "before proceeding."
        )
    print("Phase 3: session overlap check passed.")

    with h5py.File(train_pool_path, "r") as f:
        sessionnum, blocknum = f["sessionnum"][:], f["blocknum"][:]

    trial_splits = assign_block_splits(sessionnum, blocknum, cfg.split.val_fraction, cfg.split.seed)
    validation = validate_split(sessionnum, blocknum, trial_splits)
    if validation["block_overlap"]:
        raise RuntimeError(f"Block overlap detected: {validation['block_overlap']}")
    print(f"Phase 3: train={validation['n_train']}, val={validation['n_val']}, zero overlap confirmed.")

    materialize_split(train_pool_path, trial_splits, {"train": train_path, "val": val_path})
    print(f"Phase 3: wrote {train_path} and {val_path}.")


# ---------------------------------------------------------------------------
# Phase 4: Preprocessing + Path A Features
# ---------------------------------------------------------------------------

_phase4_cache: dict = {}   # module-level cache: raw (channel-excluded) arrays for phase4b to reuse


def run_phase4_preprocessing(cfg: PipelineConfig, force: bool = False) -> None:
    cfg.data.filtered_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir = cfg.model.preprocessing_dir(cfg.data.variant_tag)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    train_filtered_path = cfg.data.filtered_dir / "train_filtered.h5"
    val_filtered_path = cfg.data.filtered_dir / "val_filtered.h5"

    if not force and _exists(train_filtered_path) and _exists(val_filtered_path):
        print("Phase 4 (preprocessing): filtered outputs already exist, skipping.")
    else:
        train_path = cfg.data.splits_dir / "train.h5"
        val_path = cfg.data.splits_dir / "val.h5"

        with h5py.File(train_path, "r") as f:
            eeg_train_raw = f["eeg"][:]
            label_binary_train = f["label_binary"][:]
            label_digit_train = f["label_digit"][:]
        with h5py.File(val_path, "r") as f:
            eeg_val_raw = f["eeg"][:]
            label_binary_val = f["label_binary"][:]
            label_digit_val = f["label_digit"][:]

        fc, ac, fs = cfg.filter, cfg.artifact, cfg.data.sample_rate_hz

        print("Phase 4: filtering (diagnostic pass, full channel set)...")
        eeg_train_filtered_full = apply_filters(eeg_train_raw, fs, fc.bandpass_low_hz, fc.bandpass_high_hz,
                                                 fc.filter_order, fc.apply_notch, fc.notch_freq_hz)
        center_full, scale_full = fit_normalization_robust(eeg_train_filtered_full)
        bad_channels = detect_bad_channels(scale_full, ac.bad_channel_scale_floor)
        print(f"Phase 4: bad channels: {bad_channels}")

        eeg_train_raw, good_channels = exclude_channels(eeg_train_raw, bad_channels, cfg.data.n_channels_nominal)
        eeg_val_raw, _ = exclude_channels(eeg_val_raw, bad_channels, cfg.data.n_channels_nominal)
        eeg_train_filtered = eeg_train_filtered_full[:, good_channels, :]
        del eeg_train_filtered_full

        print("Phase 4: filtering val...")
        eeg_val_filtered = apply_filters(eeg_val_raw, fs, fc.bandpass_low_hz, fc.bandpass_high_hz,
                                          fc.filter_order, fc.apply_notch, fc.notch_freq_hz)

        center, scale = fit_normalization_robust(eeg_train_filtered)
        eeg_train_norm = apply_normalization(eeg_train_filtered, center, scale)
        eeg_val_norm = apply_normalization(eeg_val_filtered, center, scale)
        save_normalization_params(preprocessing_dir / "normalization_params.npz", center, scale)

        train_arrays = compute_artifact_arrays(eeg_train_norm)
        val_arrays = compute_artifact_arrays(eeg_val_norm)
        thresholds = derive_thresholds(train_arrays, ac.artifact_percentile, ac.flatline_percentile)
        train_artifact_info = flag_artifacts(train_arrays, thresholds, ac.trial_concern_min_channels)
        val_artifact_info = flag_artifacts(val_arrays, thresholds, ac.trial_concern_min_channels)
        save_artifact_params(preprocessing_dir / "artifact_params.json", bad_channels, thresholds)

        print(f"Phase 4: train trial_concern: {train_artifact_info['trial_concern'].sum()} / {len(eeg_train_norm)}")
        print(f"Phase 4: val trial_concern: {val_artifact_info['trial_concern'].sum()} / {len(eeg_val_norm)}")

        for path, eeg, lb, ld, info in [
            (train_filtered_path, eeg_train_norm, label_binary_train, label_digit_train, train_artifact_info),
            (val_filtered_path, eeg_val_norm, label_binary_val, label_digit_val, val_artifact_info),
        ]:
            with h5py.File(path, "w") as f:
                f.create_dataset("eeg", data=eeg, compression="gzip")
                f.create_dataset("label_binary", data=lb)
                f.create_dataset("label_digit", data=ld)
                f.create_dataset("channel_indices", data=good_channels)
                f.create_dataset("artifact_any", data=info["any"])
                f.create_dataset("n_flagged_channels_per_trial", data=info["n_flagged_channels_per_trial"])
                f.create_dataset("trial_concern", data=info["trial_concern"])
        print(f"Phase 4: wrote {train_filtered_path} and {val_filtered_path}.")

        _phase4_cache[cfg.data.variant_tag] = {
            "train": eeg_train_raw, "val": eeg_val_raw,
            "label_binary_train": label_binary_train, "label_digit_train": label_digit_train,
            "label_binary_val": label_binary_val, "label_digit_val": label_digit_val,
        }

    _run_phase4b_features(cfg, force=force)


def _run_phase4b_features(cfg: PipelineConfig, force: bool = False) -> None:
    cfg.data.features_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = cfg.data.features_dir / "train_features.h5"
    val_features_path = cfg.data.features_dir / "val_features.h5"

    if not force and _exists(train_features_path) and _exists(val_features_path):
        print("Phase 4b (features): already exist, skipping.")
        return

    cached = _phase4_cache.get(cfg.data.variant_tag)
    if cached is None:
        raise RuntimeError(
            f"Phase 4b requires Phase 4 to have run in the same session for "
            f"variant_tag={cfg.data.variant_tag!r}. Re-run run_phase4_preprocessing(cfg, force=True) first."
        )

    fs, bands, nperseg = cfg.data.sample_rate_hz, cfg.feature.eeg_bands, cfg.feature.welch_nperseg
    print("Phase 4b: extracting train features...")
    features_train = extract_features_batch(cached["train"], fs, bands, nperseg)
    print("Phase 4b: extracting val features...")
    features_val = extract_features_batch(cached["val"], fs, bands, nperseg)

    with h5py.File(train_features_path, "w") as f:
        f.create_dataset("features", data=features_train, compression="gzip")
        f.create_dataset("label_binary", data=cached["label_binary_train"])
        f.create_dataset("label_digit", data=cached["label_digit_train"])
    with h5py.File(val_features_path, "w") as f:
        f.create_dataset("features", data=features_val, compression="gzip")
        f.create_dataset("label_binary", data=cached["label_binary_val"])
        f.create_dataset("label_digit", data=cached["label_digit_val"])
    print(f"Phase 4b: wrote {train_features_path} and {val_features_path}.")


# ---------------------------------------------------------------------------
# Phase 5/6: Train + Evaluate (branches on cfg.model.is_deep) + Permutation Test
# ---------------------------------------------------------------------------

def _load_stage_data_classical(cfg: PipelineConfig):
    """Loads Path A features, applies trial_concern exclusion, and filters
    to digit-only trials if stage2. Returns (X_train, y_train, X_val, y_val, average)."""
    with h5py.File(cfg.data.features_dir / "train_features.h5", "r") as f:
        X_train, y_bin_train, y_digit_train = f["features"][:], f["label_binary"][:], f["label_digit"][:]
    with h5py.File(cfg.data.features_dir / "val_features.h5", "r") as f:
        X_val, y_bin_val, y_digit_val = f["features"][:], f["label_binary"][:], f["label_digit"][:]
    with h5py.File(cfg.data.filtered_dir / "train_filtered.h5", "r") as f:
        train_concern = f["trial_concern"][:]
    with h5py.File(cfg.data.filtered_dir / "val_filtered.h5", "r") as f:
        val_concern = f["trial_concern"][:]

    keep_train, keep_val = ~train_concern, ~val_concern

    if cfg.model.stage == "stage2":
        keep_train &= (y_digit_train != -1)
        keep_val &= (y_digit_val != -1)
        y_train, y_val, average = y_digit_train[keep_train], y_digit_val[keep_val], "macro"
    else:
        y_train, y_val, average = y_bin_train[keep_train], y_bin_val[keep_val], "binary"

    return X_train[keep_train], y_train, X_val[keep_val], y_val, average


def _load_stage_data_deep(cfg: PipelineConfig):
    """Loads Path B filtered raw EEG, applies trial_concern exclusion, and
    filters to digit-only trials if stage2. Returns (eeg_train, y_train, eeg_val, y_val)."""
    with h5py.File(cfg.data.filtered_dir / "train_filtered.h5", "r") as f:
        eeg_train, y_bin_train, y_digit_train = f["eeg"][:], f["label_binary"][:], f["label_digit"][:]
        train_concern = f["trial_concern"][:]
    with h5py.File(cfg.data.filtered_dir / "val_filtered.h5", "r") as f:
        eeg_val, y_bin_val, y_digit_val = f["eeg"][:], f["label_binary"][:], f["label_digit"][:]
        val_concern = f["trial_concern"][:]

    keep_train, keep_val = ~train_concern, ~val_concern

    if cfg.model.stage == "stage2":
        keep_train &= (y_digit_train != -1)
        keep_val &= (y_digit_val != -1)
        y_train, y_val = y_digit_train[keep_train], y_digit_val[keep_val]
    else:
        y_train, y_val = y_bin_train[keep_train], y_bin_val[keep_val]

    return eeg_train[keep_train], y_train, eeg_val[keep_val], y_val


def _save_results(cfg: PipelineConfig, results: dict) -> None:
    results_dir = cfg.model.results_dir(cfg.data.variant_tag)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_dir / 'results.json'}")


def run_model_classical(cfg: PipelineConfig) -> dict:
    X_train, y_train, X_val, y_val, average = _load_stage_data_classical(cfg)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_classical_model(cfg.model, cfg.seed)
    print(f"Phase 5 [{cfg.model.stage}/{cfg.model.model_name}]: training...")
    metrics, cm, fitted_model = evaluate_sklearn_model(
        cfg.model.model_name, model, X_train_scaled, y_train, X_val_scaled, y_val, average
    )
    print(metrics)

    checkpoint_path = cfg.model.checkpoint_path(cfg.data.variant_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted_model, "scaler": scaler}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    print("Running permutation test...")
    # Subsample for the permutation test specifically - it only needs a
    # rough noise-floor estimate, not production precision (see
    # PermutationTestConfig docstring).
    n = cfg.permutation.classical_subsample_size
    if n is not None and n < len(X_train_scaled):
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(X_train_scaled), size=n, replace=False)
        X_perm, y_perm = X_train_scaled[idx], y_train[idx]
    else:
        X_perm, y_perm = X_train_scaled, y_train

    model_fn = lambda: build_classical_model_for_permutation(
        cfg.model, cfg.seed, cfg.permutation.svm_permutation_max_iter
    )
    real_acc, shuffled_accs = permutation_test_sklearn(
        model_fn, X_perm, y_perm, X_val_scaled, y_val,
        n_permutations=cfg.permutation.n_permutations_classical, seed=cfg.seed,
    )
    gap = real_acc - shuffled_accs.mean()

    results = {
        "config": {"variant_tag": cfg.data.variant_tag, "stage": cfg.model.stage, "model": cfg.model.model_name},
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "permutation_test": {
            "real_accuracy": real_acc, "shuffled_mean": shuffled_accs.mean(),
            "shuffled_std": shuffled_accs.std(),
            "gap": gap, "gap_over_std": gap / shuffled_accs.std() if shuffled_accs.std() > 0 else None,
        },
    }
    _save_results(cfg, results)
    return results


def run_model_deep(cfg: PipelineConfig) -> dict:
    # Deferred import: torch is only required when a deep model is actually
    # selected, so classical-only usage of this package never needs it installed.
    import torch
    from src.training.loop import make_loaders, train_with_checkpointing, run_epoch, EEGDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn

    eeg_train, y_train, eeg_val, y_val = _load_stage_data_deep(cfg)
    n_channels, n_samples = eeg_train.shape[1], eeg_train.shape[2]
    n_classes = 2 if cfg.model.stage == "stage1" else 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 6 [{cfg.model.stage}/{cfg.model.model_name}]: using device {device}")
    torch.manual_seed(cfg.seed)

    train_loader, val_loader = make_loaders(eeg_train, y_train, eeg_val, y_val, cfg.training.batch_size)
    model = build_eegnet(cfg.model, n_channels, n_samples, n_classes).to(device)

    checkpoint_path = cfg.model.checkpoint_path(cfg.data.variant_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_with_checkpointing(
        model, train_loader, val_loader, checkpoint_path, device,
        cfg.training.n_epochs, cfg.training.learning_rate, cfg.training.early_stop_patience,
    )

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.numpy())

    average = "binary" if cfg.model.stage == "stage1" else "macro"
    metrics = {
        "model": cfg.model.model_name,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
        "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
        "f1": f1_score(all_labels, all_preds, average=average, zero_division=0),
    }
    print(metrics)

    print("Running (reduced-scale) permutation test...")

    def train_fn(eeg_sub, y_sub):
        m = build_eegnet(cfg.model, n_channels, n_samples, n_classes).to(device)
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
        train_fn, eval_fn, eeg_train, y_train, val_loader,
        n_permutations=cfg.permutation.n_permutations_deep,
        subsample_size=cfg.permutation.deep_subsample_size, seed=cfg.seed,
    )
    gap = real_acc - shuffled_accs.mean()

    results = {
        "config": {"variant_tag": cfg.data.variant_tag, "stage": cfg.model.stage, "model": cfg.model.model_name},
        "metrics": metrics,
        "best_val_acc": history["best_val_acc"],
        "permutation_test": {
            "real_accuracy": real_acc, "shuffled_mean": shuffled_accs.mean(),
            "shuffled_std": shuffled_accs.std(),
            "gap": gap, "gap_over_std": gap / shuffled_accs.std() if shuffled_accs.std() > 0 else None,
        },
    }
    _save_results(cfg, results)
    return results


def run_phase5_6_model(cfg: PipelineConfig) -> dict:
    """Single entry point for modeling - branches on cfg.model.is_deep so
    callers never need to know which path they're triggering."""
    if cfg.model.is_deep:
        return run_model_deep(cfg)
    else:
        return run_model_classical(cfg)


# ---------------------------------------------------------------------------
# Phase 7: Test-Set Processing + Evaluation
#
# CRITICAL RULE: nothing here ever re-fits anything. Every parameter used
# (normalization center/scale, bad channels, artifact thresholds) is
# LOADED from what Phase 4 already fit on train - test data only ever
# gets these values APPLIED, never used to derive new ones. This is what
# makes test a genuinely held-out evaluation rather than leaked information.
# ---------------------------------------------------------------------------

def run_phase7_test_processing(cfg: PipelineConfig, force: bool = False) -> None:
    """
    Applies Phase 4's already-fitted preprocessing (bad-channel exclusion,
    filtering, normalization, artifact thresholds) to the test set, and
    extracts Path A features - producing test_filtered.h5 and
    test_features.h5, matching train/val's structure exactly.
    """
    test_filtered_path = cfg.data.filtered_dir / "test_filtered.h5"
    test_features_path = cfg.data.features_dir / "test_features.h5"

    if not force and _exists(test_filtered_path) and _exists(test_features_path):
        print("Phase 7: test outputs already exist, skipping.")
        return

    preprocessing_dir = cfg.model.preprocessing_dir(cfg.data.variant_tag)
    norm_path = preprocessing_dir / "normalization_params.npz"
    artifact_path = preprocessing_dir / "artifact_params.json"
    if not norm_path.exists() or not artifact_path.exists():
        raise RuntimeError(
            f"Fitted preprocessing params not found at {preprocessing_dir} - "
            "run_phase4_preprocessing() must complete for this dataset_variant "
            "before test data can be processed."
        )

    center, scale = load_normalization_params(norm_path)
    saved = load_artifact_params(artifact_path)
    bad_channels, thresholds = saved["bad_channels"], saved["thresholds"]
    print(f"Phase 7: loaded fitted params - bad_channels={bad_channels}, thresholds={thresholds}")

    test_path = cfg.data.splits_dir / "test.h5"
    with h5py.File(test_path, "r") as f:
        eeg_test_raw = f["eeg"][:]
        label_binary_test = f["label_binary"][:]
        label_digit_test = f["label_digit"][:]

    fc, fs = cfg.filter, cfg.data.sample_rate_hz

    # Bad-channel exclusion: APPLY the saved list, never re-detect on test.
    eeg_test_raw, good_channels = exclude_channels(eeg_test_raw, bad_channels, cfg.data.n_channels_nominal)

    print("Phase 7: filtering test set...")
    eeg_test_filtered = apply_filters(eeg_test_raw, fs, fc.bandpass_low_hz, fc.bandpass_high_hz,
                                       fc.filter_order, fc.apply_notch, fc.notch_freq_hz)

    # Normalization: APPLY the saved center/scale, never re-fit.
    eeg_test_norm = apply_normalization(eeg_test_filtered, center, scale)

    # Artifact flagging: APPLY the saved thresholds, never re-derive.
    test_arrays = compute_artifact_arrays(eeg_test_norm)
    test_artifact_info = flag_artifacts(test_arrays, thresholds, cfg.artifact.trial_concern_min_channels)
    print(f"Phase 7: test trial_concern: {test_artifact_info['trial_concern'].sum()} / {len(eeg_test_norm)}")

    cfg.data.filtered_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(test_filtered_path, "w") as f:
        f.create_dataset("eeg", data=eeg_test_norm, compression="gzip")
        f.create_dataset("label_binary", data=label_binary_test)
        f.create_dataset("label_digit", data=label_digit_test)
        f.create_dataset("channel_indices", data=good_channels)
        f.create_dataset("artifact_any", data=test_artifact_info["any"])
        f.create_dataset("n_flagged_channels_per_trial", data=test_artifact_info["n_flagged_channels_per_trial"])
        f.create_dataset("trial_concern", data=test_artifact_info["trial_concern"])
    print(f"Phase 7: wrote {test_filtered_path}")

    # Path A features - computed on RAW (channel-excluded, unfiltered) test
    # signal, same as train/val (see features/extraction.py docstring for why).
    print("Phase 7: extracting test features...")
    features_test = extract_features_batch(eeg_test_raw, fs, cfg.feature.eeg_bands, cfg.feature.welch_nperseg)

    cfg.data.features_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(test_features_path, "w") as f:
        f.create_dataset("features", data=features_test, compression="gzip")
        f.create_dataset("label_binary", data=label_binary_test)
        f.create_dataset("label_digit", data=label_digit_test)
    print(f"Phase 7: wrote {test_features_path}")


def _load_test_data_classical(cfg: PipelineConfig):
    with h5py.File(cfg.data.features_dir / "test_features.h5", "r") as f:
        X_test, y_bin_test, y_digit_test = f["features"][:], f["label_binary"][:], f["label_digit"][:]
    with h5py.File(cfg.data.filtered_dir / "test_filtered.h5", "r") as f:
        test_concern = f["trial_concern"][:]

    keep = ~test_concern
    if cfg.model.stage == "stage2":
        keep &= (y_digit_test != -1)
        y_test, average = y_digit_test[keep], "macro"
    else:
        y_test, average = y_bin_test[keep], "binary"
    return X_test[keep], y_test, average


def _load_test_data_deep(cfg: PipelineConfig):
    with h5py.File(cfg.data.filtered_dir / "test_filtered.h5", "r") as f:
        eeg_test, y_bin_test, y_digit_test = f["eeg"][:], f["label_binary"][:], f["label_digit"][:]
        test_concern = f["trial_concern"][:]

    keep = ~test_concern
    if cfg.model.stage == "stage2":
        keep &= (y_digit_test != -1)
        y_test = y_digit_test[keep]
    else:
        y_test = y_bin_test[keep]
    return eeg_test[keep], y_test


def run_phase7_evaluate_on_test(cfg: PipelineConfig) -> dict:
    """
    Loads the ALREADY-TRAINED checkpoint from Phase 5/6 (does not retrain)
    and evaluates it on the held-out test set. This is the true, final,
    only-look-at-once number for this run.
    """
    from src.evaluation.metrics import compute_classification_metrics
    checkpoint_path = cfg.model.checkpoint_path(cfg.data.variant_tag)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No trained checkpoint found at {checkpoint_path} - "
            "run_phase5_6_model() must complete before test evaluation."
        )

    if cfg.model.is_deep:
        import torch
        eeg_test, y_test = _load_test_data_deep(cfg)
        n_channels, n_samples = eeg_test.shape[1], eeg_test.shape[2]
        n_classes = 2 if cfg.model.stage == "stage1" else 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_eegnet(cfg.model, n_channels, n_samples, n_classes).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        from torch.utils.data import DataLoader
        from src.training.loop import EEGDataset
        loader = DataLoader(EEGDataset(eeg_test, y_test), batch_size=cfg.training.batch_size, shuffle=False)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = model(X_batch.to(device))
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        average = "binary" if cfg.model.stage == "stage1" else "macro"
        metrics = compute_classification_metrics(all_labels, all_preds, cfg.model.model_name, average)
    else:
        X_test, y_test, average = _load_test_data_classical(cfg)
        saved = joblib.load(checkpoint_path)
        model, scaler = saved["model"], saved["scaler"]
        X_test_scaled = scaler.transform(X_test)
        preds = model.predict(X_test_scaled)
        metrics = compute_classification_metrics(y_test, preds, cfg.model.model_name, average)

    print(f"Phase 7 [{cfg.model.stage}/{cfg.model.model_name}] TEST metrics: {metrics}")

    # Compare against val, if we have it - a large val/test gap is a red
    # flag for overfitting to val itself (e.g. via repeated tuning against it).
    results_dir = cfg.model.results_dir(cfg.data.variant_tag)
    val_results_path = results_dir / "results.json"
    comparison = None
    if val_results_path.exists():
        with open(val_results_path) as f:
            val_results = json.load(f)
        val_acc = val_results["metrics"]["accuracy"]
        comparison = {"val_accuracy": val_acc, "test_accuracy": metrics["accuracy"],
                      "gap": val_acc - metrics["accuracy"]}
        print(f"Val vs. test: {comparison}")

    test_results = {"config": {"variant_tag": cfg.data.variant_tag, "stage": cfg.model.stage,
                                "model": cfg.model.model_name},
                     "test_metrics": metrics, "val_vs_test": comparison}
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"Test results saved to {results_dir / 'test_results.json'}")

    return test_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    dataset_variant: str = "2B",
    subsample_fraction: float = 0.20,
    stage: str = "stage1",
    model_name: str = "random_forest",
    seed: int = 42,
    force: bool = False,
) -> dict:
    """
    The intended way to run this pipeline: set these ~5 parameters, get a
    fully-processed dataset and a trained, evaluated model back. Everything
    else - paths, HF repo id, trial counts, which training code path runs -
    is derived automatically from these.
    """
    cfg = build_config(dataset_variant, subsample_fraction, stage, model_name, seed)
    print(f"=== dataset_variant={dataset_variant!r} subsample_fraction={subsample_fraction} "
          f"stage={stage!r} model={model_name!r} ===")

    run_phase2_data_acquisition(cfg, force=force)
    run_phase3_splitting(cfg, force=force)
    run_phase4_preprocessing(cfg, force=force)
    results = run_phase5_6_model(cfg)

    print("\nFinal results:")
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
