"""
Model factory: given a ModelConfig (specifically cfg.model.model_name),
returns the correct model instance with its hyperparameters already
applied. This is the single place "which model to use" gets resolved -
everything downstream (training, evaluation) just receives a ready model,
without needing to know or care which one it is.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import ModelConfig
from src.models.eegnet import EEGNet


def build_classical_model(cfg: ModelConfig, seed: int):
    """
    Returns an unfitted sklearn-compatible estimator for cfg.model_name.
    Raises if cfg.model_name is a deep model - use build_eegnet() for that.
    """
    if cfg.model_name == "lda":
        return LinearDiscriminantAnalysis()
    elif cfg.model_name == "svm":
        return SVC(kernel=cfg.svm.kernel, random_state=seed)
    elif cfg.model_name == "random_forest":
        return RandomForestClassifier(n_estimators=cfg.random_forest.n_estimators,
                                       random_state=seed, n_jobs=-1)
    else:
        raise ValueError(
            f"{cfg.model_name!r} is not a classical model - use build_eegnet() instead, "
            f"or check cfg.is_deep before calling this function."
        )


def build_eegnet(cfg: ModelConfig, n_channels: int, n_samples: int, n_classes: int) -> EEGNet:
    """
    Returns an untrained EEGNet instance, sized for this run's data
    (n_channels/n_samples/n_classes depend on the actual data - e.g.
    n_classes=2 for stage1, 10 for stage2 - so they're passed in rather
    than read from config).
    """
    if cfg.model_name != "eegnet":
        raise ValueError(f"cfg.model_name is {cfg.model_name!r}, not 'eegnet'.")
    p = cfg.eegnet
    return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes,
                  F1=p.F1, D=p.D, F2=p.F2, kernel_length=p.kernel_length, dropout=p.dropout)


def build_classical_model_for_permutation(cfg: ModelConfig, seed: int, svm_max_iter: int = 300):
    """
    Faster surrogate used ONLY inside the permutation-test harness - the
    real, reported model (build_classical_model) is never affected by this.

    SVM specifically swaps the production RBF-kernel SVC for a capped-
    iteration LinearSVC: on shuffled labels there's no real structure to
    find, so RBF SVC's iterative solver often fails to converge at all and
    burns through its full iteration budget every single permutation run -
    observed directly during development (single real fit: ~2 min;
    shuffled fits: 30+ min each, un-capped). LDA and RandomForest don't
    have this problem and use the same builder as production.
    """
    if cfg.model_name == "svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(random_state=seed, max_iter=svm_max_iter, tol=1e-2, dual="auto")
    return build_classical_model(cfg, seed)
