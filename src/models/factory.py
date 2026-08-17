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


# ---------------------------------------------------------------------------
# Stage 2 EEGNet variant registry
#
# Each entry is a standalone function with the same contract:
#   (cfg: ModelConfig, n_channels, n_samples, n_classes, stage1_checkpoint_path) -> EEGNet
# To add a new variant later: write one function matching this contract,
# add one line to STAGE2_MODEL_REGISTRY. No other code needs to change.
# ---------------------------------------------------------------------------

def _load_stage1_backbone(cfg: ModelConfig, n_channels: int, n_samples: int,
                            stage1_checkpoint_path) -> EEGNet:
    """Shared helper for B1/B2: builds a Stage-1-shaped (2-class) EEGNet
    and loads Stage 1's trained weights into it. Raises clearly if the
    checkpoint doesn't exist, rather than silently falling back to a
    fresh/random backbone."""
    import torch
    from pathlib import Path as _Path

    if stage1_checkpoint_path is None or not _Path(stage1_checkpoint_path).exists():
        raise RuntimeError(
            f"Stage 2 variant {cfg.stage2.variant!r} requires a trained Stage 1 EEGNet "
            f"checkpoint, but none was found at {stage1_checkpoint_path!r}. "
            "Run Stage 1 EEGNet (stage='stage1', model_name='eegnet') for the same "
            "dataset_variant first."
        )
    model = build_eegnet(
        ModelConfig(model_name="eegnet", stage="stage1", eegnet=cfg.eegnet),
        n_channels, n_samples, n_classes=2,   # Stage 1's shape - must match the saved checkpoint
    )
    state = torch.load(stage1_checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _build_stage2_path_a(cfg: ModelConfig, n_channels: int, n_samples: int,
                           n_classes: int, stage1_checkpoint_path=None) -> EEGNet:
    """Path A: fresh EEGNet, trained independently on Stage 2 data - no
    reuse from Stage 1. The baseline every reuse variant must beat to
    justify its extra complexity."""
    return build_eegnet(cfg, n_channels, n_samples, n_classes)


def _build_stage2_path_b1(cfg: ModelConfig, n_channels: int, n_samples: int,
                            n_classes: int, stage1_checkpoint_path=None) -> EEGNet:
    """Path B1: Stage 1's backbone reused and FROZEN - only a new 10-class
    head is trained. Backbone acts as a fixed feature extractor."""
    model = _load_stage1_backbone(cfg, n_channels, n_samples, stage1_checkpoint_path)
    model.replace_classifier(n_classes)
    model.freeze_backbone()
    return model


def _build_stage2_path_b2(cfg: ModelConfig, n_channels: int, n_samples: int,
                            n_classes: int, stage1_checkpoint_path=None) -> EEGNet:
    """Path B2: same starting point as B1 (frozen backbone) - the
    unfreeze-and-fine-tune step happens during TRAINING (see
    training/loop.py's train_stage2_path_b2), not at construction time."""
    model = _load_stage1_backbone(cfg, n_channels, n_samples, stage1_checkpoint_path)
    model.replace_classifier(n_classes)
    model.freeze_backbone()
    return model


STAGE2_MODEL_REGISTRY = {
    "path_a": _build_stage2_path_a,
    "path_b1": _build_stage2_path_b1,
    "path_b2": _build_stage2_path_b2,
}


def build_stage2_eegnet(cfg: ModelConfig, n_channels: int, n_samples: int, n_classes: int,
                          stage1_checkpoint_path=None) -> EEGNet:
    """Dispatches to the registered builder for cfg.stage2.variant."""
    if cfg.stage2.variant not in STAGE2_MODEL_REGISTRY:
        raise ValueError(f"Unknown stage2 variant {cfg.stage2.variant!r}. "
                          f"Known: {list(STAGE2_MODEL_REGISTRY)}")
    return STAGE2_MODEL_REGISTRY[cfg.stage2.variant](cfg, n_channels, n_samples, n_classes,
                                                        stage1_checkpoint_path)
