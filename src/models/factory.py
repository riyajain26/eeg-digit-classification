"""
Model registry: given a model_name, resolves how to build it.

Every model - classical (LDA/SVM/RandomForest) and every EEGNet variant,
including the three backbone-reuse strategies - is a single, self-
contained, descriptively-named entry in ONE flat MODEL_REGISTRY. There is
no separate "which model" + "which variant" split: model_name alone tells
you exactly what you're getting, which is the point - someone reading
this file for the first time shouldn't have to cross-reference a second
registry or decode an abbreviation to know what "eegnet_frozen_backbone_reuse"
does.

The four EEGNet entries:
- eegnet_fresh: a fresh, untrained EEGNet - no reuse from anywhere else.
  Works for EITHER task (binary or multiclass); it's the ONLY option for
  binary (nothing exists yet to reuse), and multiclass's own no-reuse
  baseline.
- eegnet_frozen_backbone_reuse: reuses another task's trained eegnet_fresh
  backbone, FROZEN - only a new classification head is trained on top.
- eegnet_finetuned_backbone_reuse: same starting point as
  eegnet_frozen_backbone_reuse, then the backbone is UNFROZEN and
  fine-tuned at a low learning rate (see training/loop.py's
  train_with_two_phase_finetuning - the two-phase part happens during
  training, not at construction time here).
- eegnet_dual_backbone_auxiliary_input: raw EEG fed to BOTH a frozen
  reused backbone AND a fresh trainable one, concatenated before a new
  head - distinct from the two reuse-only variants above, which only ever
  see the reused backbone's learned features, never the raw signal directly.

Every registered build function shares ONE contract:

    build(cfg: PipelineConfig, n_channels: int | None, n_samples: int | None,
          n_classes: int | None) -> model

    cfg is the FULL PipelineConfig (not just cfg.model) - classical models
    only need cfg.model.<their own params> and cfg.seed, but the
    backbone-reuse variants also need to look up ANOTHER model's saved
    checkpoint (cfg.model.eegnet.reuse_source_task, cfg.model.model_root),
    so passing the whole config avoids a narrower, inconsistent signature
    for those cases.

    n_channels/n_samples/n_classes are meaningless to classical models
    (features are already flattened) - they simply ignore them. This is
    what lets src/steps/models.py call
    `MODEL_REGISTRY.get(cfg.model.model_name).build(cfg, n_channels,
    n_samples, n_classes)` identically regardless of which model_name was
    selected, with no branching needed to CONSTRUCT a model (a branch is
    still needed to TRAIN one - sklearn's fit() and PyTorch's training
    loop are genuinely different APIs, not an artifact of how this file
    is organized).

is_deep tells the caller which of those two training APIs to use -
resolved from the registry entry rather than a separate hardcoded set, so
it can never drift out of sync with which build function actually runs.

build_for_permutation (optional, defaults to None = "use build()") is a
cheaper substitute used only inside the permutation-test harness (see
evaluation/permutation_test.py) - e.g. SVM swaps its slow RBF kernel for a
capped-iteration LinearSVC, and every reuse-based EEGNet variant
substitutes a plain eegnet_fresh build rather than replicating the full
reuse procedure for a rough noise-floor check.

To add a new model entirely - classical, a new EEGNet reuse strategy, or
a future architecture (autoencoder, meta-learning, whatever comes next):
write one build function matching the contract above, with a name that
describes what it does, and register it. Nothing else in this file, or in
src/steps/models.py, needs to change.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import CLASSICAL_MODEL_NAMES, DEEP_MODEL_NAMES, ModelConfig, PipelineConfig, TASK_INFO
from src.models.eegnet import EEGNet, EEGNetDualBackbone
from src.utils.registry import Registry

MODEL_REGISTRY = Registry(step_name="model")


@dataclass
class ModelBuilder:
    build: Callable
    is_deep: bool
    build_for_permutation: Callable | None = None


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def build_lda(cfg: PipelineConfig, n_channels, n_samples, n_classes):
    return LinearDiscriminantAnalysis()


def build_svm(cfg: PipelineConfig, n_channels, n_samples, n_classes):
    return SVC(kernel=cfg.model.svm.kernel, random_state=cfg.seed)


def build_svm_for_permutation(cfg: PipelineConfig, n_channels, n_samples, n_classes):
    """
    Faster surrogate used ONLY inside the permutation-test harness - the
    real, reported model (build_svm) is never affected by this.

    Swaps the production RBF-kernel SVC for a capped-iteration LinearSVC:
    on shuffled labels there's no real structure to find, so RBF SVC's
    iterative solver often fails to converge at all and burns through its
    full iteration budget every single permutation run - observed
    directly during development (single real fit: ~2 min; shuffled fits:
    30+ min each, un-capped).
    """
    from sklearn.svm import LinearSVC
    return LinearSVC(random_state=cfg.seed, max_iter=cfg.permutation.svm_permutation_max_iter,
                      tol=1e-2, dual="auto")


def build_random_forest(cfg: PipelineConfig, n_channels, n_samples, n_classes):
    return RandomForestClassifier(n_estimators=cfg.model.random_forest.n_estimators,
                                   random_state=cfg.seed, n_jobs=-1)


MODEL_REGISTRY.register("lda")(ModelBuilder(build=build_lda, is_deep=False))
MODEL_REGISTRY.register("svm")(ModelBuilder(build=build_svm, is_deep=False,
                                             build_for_permutation=build_svm_for_permutation))
MODEL_REGISTRY.register("random_forest")(ModelBuilder(build=build_random_forest, is_deep=False))


# ---------------------------------------------------------------------------
# EEGNet - fresh (no reuse)
# ---------------------------------------------------------------------------

def build_eegnet_fresh(cfg: PipelineConfig, n_channels: int, n_samples: int, n_classes: int):
    """A fresh, untrained EEGNet sized for this run's task - no reuse from
    any other model. Works for EITHER task: it's the only option for
    task="binary" (nothing exists yet to reuse), and it's multiclass's own
    no-reuse baseline every reuse-based variant needs to beat to justify
    its extra complexity. Also used as the permutation-test substitute for
    every reuse-based variant below - see their build_for_permutation."""
    p = cfg.model.eegnet
    return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes,
                  F1=p.F1, D=p.D, F2=p.F2, kernel_length=p.kernel_length, dropout=p.dropout)


MODEL_REGISTRY.register("eegnet_fresh")(ModelBuilder(build=build_eegnet_fresh, is_deep=True))


# ---------------------------------------------------------------------------
# EEGNet - backbone-reuse variants
# ---------------------------------------------------------------------------

def _load_reuse_source_backbone(cfg: PipelineConfig, n_channels: int, n_samples: int) -> EEGNet:
    """
    Shared by every backbone-reuse variant below: loads a trained
    eegnet_fresh checkpoint for cfg.model.eegnet.reuse_source_task (e.g.
    the binary task's trained backbone, to reuse for multiclass).

    Raises clearly if that checkpoint doesn't exist yet, rather than
    silently falling back to a fresh/random backbone.
    """
    source_task = cfg.model.eegnet.reuse_source_task
    source_model_cfg = ModelConfig(model_name="eegnet_fresh", task=source_task, model_root=cfg.model.model_root)
    checkpoint_path = source_model_cfg.checkpoint_path(cfg.data.variant_tag)

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"{cfg.model.model_name!r} requires a trained eegnet_fresh checkpoint for "
            f"task={source_task!r}, but none was found at {checkpoint_path}. Run "
            f"`python scripts/run_models.py --model-name eegnet_fresh --task {source_task}` first."
        )

    p = cfg.model.eegnet
    n_source_classes = TASK_INFO[source_task].n_classes   # the SOURCE task's class count, not this run's
    model = EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_source_classes,
                   F1=p.F1, D=p.D, F2=p.F2, kernel_length=p.kernel_length, dropout=p.dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def build_eegnet_frozen_backbone_reuse(cfg: PipelineConfig, n_channels: int, n_samples: int, n_classes: int):
    """Reuses another task's trained backbone, FROZEN - only a new
    n_classes-size classification head is trained on top. The backbone
    acts purely as a fixed feature extractor."""
    model = _load_reuse_source_backbone(cfg, n_channels, n_samples)
    model.replace_classifier(n_classes)
    model.freeze_backbone()
    return model


def build_eegnet_finetuned_backbone_reuse(cfg: PipelineConfig, n_channels: int, n_samples: int, n_classes: int):
    """Same starting point as build_eegnet_frozen_backbone_reuse (frozen
    backbone) - the unfreeze-and-fine-tune step happens during TRAINING
    (see training/loop.py's train_with_two_phase_finetuning), not here at
    construction time. This function only sets up phase 1's starting state."""
    model = _load_reuse_source_backbone(cfg, n_channels, n_samples)
    model.replace_classifier(n_classes)
    model.freeze_backbone()
    return model


def build_eegnet_dual_backbone_auxiliary_input(cfg: PipelineConfig, n_channels: int, n_samples: int, n_classes: int):
    """Raw EEG fed to BOTH a frozen reused backbone AND a fresh trainable
    backbone, concatenated before a new classifier head. Distinct from the
    two reuse-only variants above, which only ever see the reused
    backbone's learned FEATURES - this variant always has direct access to
    the raw signal too, alongside whatever the reused backbone learned."""
    frozen_backbone = _load_reuse_source_backbone(cfg, n_channels, n_samples)
    # This fresh backbone's own classifier head is unused/discarded - only
    # its _forward_features() output feeds into EEGNetDualBackbone's
    # classifier. Kept as a full EEGNet instance for simplicity rather
    # than a stripped-down feature-extractor-only variant.
    fresh_backbone = build_eegnet_fresh(cfg, n_channels, n_samples, n_classes)
    return EEGNetDualBackbone(frozen_backbone, fresh_backbone, n_classes, n_channels, n_samples)


MODEL_REGISTRY.register("eegnet_frozen_backbone_reuse")(
    ModelBuilder(build=build_eegnet_frozen_backbone_reuse, is_deep=True,
                 build_for_permutation=build_eegnet_fresh)
)
MODEL_REGISTRY.register("eegnet_finetuned_backbone_reuse")(
    ModelBuilder(build=build_eegnet_finetuned_backbone_reuse, is_deep=True,
                 build_for_permutation=build_eegnet_fresh)
)
MODEL_REGISTRY.register("eegnet_dual_backbone_auxiliary_input")(
    ModelBuilder(build=build_eegnet_dual_backbone_auxiliary_input, is_deep=True,
                 build_for_permutation=build_eegnet_fresh)
)


# ---------------------------------------------------------------------------
# Sanity check: config.py's CLASSICAL_MODEL_NAMES/DEEP_MODEL_NAMES are kept
# in sync with this registry BY HAND (see config.py's comment on why they
# aren't derived from it directly, to avoid a circular import). This check
# catches the two drifting apart immediately at import time instead of
# letting them silently disagree.
# ---------------------------------------------------------------------------

_expected_names = CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES
if set(MODEL_REGISTRY.names()) != _expected_names:
    raise RuntimeError(
        f"MODEL_REGISTRY names {set(MODEL_REGISTRY.names())} don't match config.py's "
        f"CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES {_expected_names} - update one to match "
        "the other (src/models/factory.py registrations vs src/config.py's name sets)."
    )
