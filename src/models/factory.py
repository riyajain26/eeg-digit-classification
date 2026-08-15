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
