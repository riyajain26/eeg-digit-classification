"""
PyTorch Dataset wrapper and training loop with checkpointing/early stopping,
from Notebook 06, Sections 3 and 5. Designed to run identically on CPU,
local GPU, or Colab GPU - device is passed in, never hardcoded.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


class EEGDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray):
        # EEGNet expects (batch, 1, channels, samples) - add the "1 input
        # channel" dimension here (as in a single-channel image).
        self.eeg = torch.tensor(eeg, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]


def make_loaders(eeg_train, y_train, eeg_val, y_val, batch_size: int):
    train_loader = DataLoader(EEGDataset(eeg_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(eeg_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, optimizer, criterion, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_with_checkpointing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    n_epochs: int,
    learning_rate: float,
    early_stop_patience: int,
    initial_best_val_acc: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Returns training history dict; saves best-val-accuracy checkpoint to disk.

    initial_best_val_acc: lets a SECOND call continue tracking "best" from
    where a PRIOR call left off, rather than resetting to 0.0 - used by
    train_stage2_path_b2() so its fine-tune phase can't overwrite a good
    checkpoint from the freeze phase with a worse one just because it's
    the first few epochs of a fresh call.

    Only trains parameters with requires_grad=True - safe for both a
    frozen backbone (Path B1: most params excluded) and a fully trainable
    model (Path A, or Path B2's fine-tune phase: all params included).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = initial_best_val_acc
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print("  -> New best val_acc, checkpoint saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                if verbose:
                    print(f"Early stopping - no improvement for {early_stop_patience} epochs.")
                break

    history["best_val_acc"] = best_val_acc
    return history


def train_stage2_path_b2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Path,
    device: torch.device,
    freeze_epochs: int,
    finetune_epochs: int,
    base_learning_rate: float,
    finetune_lr_multiplier: float,
    early_stop_patience: int,
    verbose: bool = True,
) -> dict:
    """
    Two-phase training for Stage 2 Path B2:
    Phase 1 - train only the new head, backbone frozen (model must already
              have freeze_backbone() applied before this is called).
    Phase 2 - unfreeze the backbone, continue training the whole network
              at a lower learning rate (base_learning_rate * finetune_lr_multiplier).

    Best checkpoint is tracked ACROSS both phases combined - phase 2 loads
    phase 1's best weights before continuing, and can only overwrite the
    checkpoint if it beats phase 1's peak, not just epoch-1-of-phase-2's
    starting point. This guards against fine-tuning regressing a good
    frozen-phase result before it has a chance to improve on it.
    """
    if verbose:
        print("Path B2, Phase 1: training head only (backbone frozen)...")
    history_frozen = train_with_checkpointing(
        model, train_loader, val_loader, checkpoint_path, device,
        n_epochs=freeze_epochs, learning_rate=base_learning_rate,
        early_stop_patience=early_stop_patience, verbose=verbose,
    )
    best_after_freeze = history_frozen["best_val_acc"]

    if verbose:
        print(f"\nPath B2, Phase 2: unfreezing backbone, fine-tuning "
              f"(best so far: {best_after_freeze:.4f})...")
    model.unfreeze_backbone()
    model.load_state_dict(torch.load(checkpoint_path))   # continue from phase 1's best, not wherever training left off

    history_finetune = train_with_checkpointing(
        model, train_loader, val_loader, checkpoint_path, device,
        n_epochs=finetune_epochs, learning_rate=base_learning_rate * finetune_lr_multiplier,
        early_stop_patience=early_stop_patience,
        initial_best_val_acc=best_after_freeze,   # phase 2 can't "improve" just by matching phase 1
        verbose=verbose,
    )

    return {
        "freeze_phase": history_frozen,
        "finetune_phase": history_finetune,
        "best_val_acc": history_finetune["best_val_acc"],
    }
