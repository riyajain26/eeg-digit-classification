"""
Permutation-test (shuffled-label) harnesses - the floor every real model
result must clear before being trusted (Notebook 03, Section 6; extended
for deep models in Notebook 06, Section 7).

Two versions, since classical (sklearn) and deep (PyTorch) models have
different training APIs:
- permutation_test_sklearn: general-purpose, works with any sklearn-style
  estimator (fit/predict), used at full scale.
- permutation_test_torch: deep-model version, deliberately cheaper (fewer
  epochs, smaller subsample, fewer permutations) given per-run training
  cost - a directional check, not a precise one. Bump the parameters up if
  compute allows a more rigorous version.
"""

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score


def permutation_test_sklearn(
    model_fn: Callable, X_train, y_train, X_val, y_val,
    n_permutations: int = 20, seed: int = 42,
) -> tuple[float, np.ndarray]:
    """model_fn: zero-arg callable returning an unfitted sklearn-compatible estimator."""
    rng = np.random.default_rng(seed)

    real_model = model_fn()
    real_model.fit(X_train, y_train)
    real_acc = accuracy_score(y_val, real_model.predict(X_val))

    shuffled_accs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_train)
        model = model_fn()
        model.fit(X_train, y_shuffled)
        shuffled_accs.append(accuracy_score(y_val, model.predict(X_val)))

    return real_acc, np.array(shuffled_accs)


def permutation_test_torch(
    train_fn: Callable, eval_fn: Callable,
    eeg_train, y_train, val_loader,
    n_permutations: int = 5, subsample_size: int = 3000, seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    train_fn(eeg_subset, y_subset) -> trained model (should internally use a
    reduced epoch count for speed - this is a rough directional check, not
    a precise one, given deep-model training cost).
    eval_fn(model, loader) -> accuracy on val_loader.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eeg_train), size=min(subsample_size, len(eeg_train)), replace=False)
    eeg_sub, y_sub = eeg_train[idx], y_train[idx]

    real_model = train_fn(eeg_sub, y_sub)
    real_acc = eval_fn(real_model, val_loader)

    shuffled_accs = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_sub)
        shuffled_model = train_fn(eeg_sub, y_shuffled)
        shuffled_accs.append(eval_fn(shuffled_model, val_loader))

    return real_acc, np.array(shuffled_accs)
