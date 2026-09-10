"""
Full pipeline: chains Steps 1-4 into one entry point.

run_pipeline(cfg) calls, in order:
  1. run_data_preparation(cfg)   [src/steps/data_preparation.py]
  2. run_preprocessing(cfg)      [src/steps/preprocessing.py]
  3. run_features(cfg)           [src/steps/features.py] - ONLY when
     cfg.model.is_deep is False. Deep models read filtered EEG directly
     and never touch features (see src/steps/features.py's module
     docstring) - running this step for a deep-only run would just burn
     time computing something nothing downstream reads.
  4. run_model_training(cfg)     [src/steps/models.py]

Plus one genuinely new piece: if the selected model needs a backbone
reused from another task (any *_backbone_reuse/_auxiliary_input model)
and that source checkpoint doesn't exist yet, it's trained first,
automatically, as a nested run_pipeline() call for the source task's
eegnet_fresh - so running e.g. eegnet_frozen_backbone_reuse cold, with no
prior binary-task run, just works rather than raising and telling you to
go run a separate command first.

Also provides prompt_for_variants(cfg), the interactive-selection piece.
This lives HERE, not in config.py, because it needs to read the actual
registries (FILTER_REGISTRY, MODEL_REGISTRY, etc.) to show valid options -
and config.py deliberately never imports from src.preprocessing/src.models/
etc (see config.py's own comment on why: avoiding a circular import).
build_config() itself never prompts; only scripts/run_pipeline.py's
--interactive flag calls this.
"""

import copy

from src.config import CLASSICAL_MODEL_NAMES, DEEP_MODEL_NAMES, ModelConfig, PipelineConfig, TASK_INFO
from src.steps.data_preparation import run_data_preparation
from src.steps.features import run_features
from src.steps.models import run_model_training
from src.steps.preprocessing import run_preprocessing

REUSE_MODEL_NAMES = {
    "eegnet_frozen_backbone_reuse",
    "eegnet_finetuned_backbone_reuse",
    "eegnet_dual_backbone_auxiliary_input",
}


def _ensure_reuse_source_trained(cfg: PipelineConfig, force: bool, run_permutation: bool) -> None:
    """
    If cfg.model.model_name is one of the backbone-reuse variants and its
    source task's eegnet_fresh checkpoint doesn't exist yet, trains it
    first via a nested run_pipeline() call.

    Can only recurse one level deep: the nested call always uses
    model_name="eegnet_fresh", which is never itself in REUSE_MODEL_NAMES -
    so its own _ensure_reuse_source_trained() call immediately returns,
    with no explicit recursion guard needed.
    """
    if cfg.model.model_name not in REUSE_MODEL_NAMES:
        return

    source_task = cfg.model.eegnet.reuse_source_task
    source_model_cfg = ModelConfig(model_name="eegnet_fresh", task=source_task, model_root=cfg.model.model_root)
    checkpoint_path = source_model_cfg.checkpoint_path(cfg.data.variant_tag)

    if checkpoint_path.exists() and not force:
        return

    print(f"\nPipeline: {cfg.model.model_name!r} needs a trained eegnet_fresh checkpoint for "
          f"task={source_task!r} (not found at {checkpoint_path}) - training it first.\n")

    # Deep copy so the prerequisite run can't mutate the caller's cfg -
    # everything except model_name/task carries over unchanged, so the
    # SAME acquired/preprocessed data is reused, not redone.
    prereq_cfg = copy.deepcopy(cfg)
    prereq_cfg.model.model_name = "eegnet_fresh"
    prereq_cfg.model.task = source_task
    run_pipeline(prereq_cfg, force=force, run_permutation=run_permutation)

    print(f"\nPipeline: prerequisite {source_task!r} eegnet_fresh training complete - "
          f"resuming {cfg.model.model_name!r}.\n")


def run_pipeline(cfg: PipelineConfig, force: bool = False, run_permutation: bool = True) -> dict:
    """
    Runs every step needed to go from nothing to a trained, evaluated
    model, for whatever cfg.model.task/model_name selects. This is the
    ONE function scripts/run_pipeline.py calls.
    """
    _ensure_reuse_source_trained(cfg, force=force, run_permutation=run_permutation)

    run_data_preparation(cfg, force=force)
    run_preprocessing(cfg, force=force)
    if not cfg.model.is_deep:
        run_features(cfg, force=force)
    return run_model_training(cfg, run_permutation=run_permutation)


def prompt_for_variants(cfg: PipelineConfig) -> PipelineConfig:
    """
    Interactively confirms/overrides each pluggable step's variant at the
    terminal - shows the current value in brackets, Enter keeps it.
    Mutates and returns the same cfg. Only called when
    scripts/run_pipeline.py is run with --interactive.
    """
    from src.features.extraction import FEATURE_REGISTRY
    from src.preprocessing.artifacts import ARTIFACT_REGISTRY
    from src.preprocessing.filters import FILTER_REGISTRY
    from src.preprocessing.normalization import NORMALIZATION_REGISTRY

    def ask(label: str, current: str, options) -> str:
        raw = input(f"{label} [{current}]  (options: {', '.join(sorted(options))}): ").strip()
        return raw or current

    cfg.filter.variant = ask("Filter variant", cfg.filter.variant, FILTER_REGISTRY.names())
    cfg.normalization.variant = ask("Normalization variant", cfg.normalization.variant, NORMALIZATION_REGISTRY.names())
    cfg.artifact.variant = ask("Artifact variant", cfg.artifact.variant, ARTIFACT_REGISTRY.names())
    cfg.model.task = ask("Task", cfg.model.task, TASK_INFO.keys())
    cfg.model.model_name = ask("Model", cfg.model.model_name, CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES)
    cfg.model.__post_init__()   # re-validate after manual overrides

    if not cfg.model.is_deep:
        cfg.feature.variant = ask("Feature variant", cfg.feature.variant, FEATURE_REGISTRY.names())

    return cfg
