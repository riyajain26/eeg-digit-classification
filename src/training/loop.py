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
    verbose: bool = True,
) -> dict:
    """Returns training history dict; saves best-val-accuracy checkpoint to disk."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
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
