"""
Shared evaluation helpers: classical model fit+eval, and metric
computation kept consistent across every model family (classical and
EEGNet) so results are directly comparable regardless of which model_name
produced them.

No registry here: "how to compute accuracy/precision/recall/F1" isn't a
pluggable strategy - it's a fixed, standard definition every model's
results get held to equally.
"""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)


def compute_classification_metrics(y_true, y_pred, model_name: str, average: str = "binary") -> dict:
    """
    average: "binary" for Stage 1 (blank vs digit, 2 classes) - precision/
        recall/F1 on the positive class.
        "macro" for Stage 2 (digit 0-9, 10 classes) - treats every digit
        class equally regardless of frequency. Passed in by the caller
        based on cfg.model.stage - this function itself has no knowledge
        of "stage", only the average mode, keeping it reusable outside
        this project if needed.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_sklearn_model(name: str, model, X_train, y_train, X_val, y_val, average: str = "binary"):
    """Fits an sklearn-style model and returns (metrics dict, confusion matrix, fitted model)."""
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = compute_classification_metrics(y_val, preds, name, average)
    cm = confusion_matrix(y_val, preds)
    return metrics, cm, model
