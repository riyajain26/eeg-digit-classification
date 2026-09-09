"""
Central configuration.

Design principle (unchanged from the original pipeline, kept deliberately):
the user should only ever need to set a handful of top-level parameters,
and every path, count, and derived value follows automatically. Nothing
below should ever need hand-editing to change dataset scale.

BUILD STATUS - Steps 1-4 of modularization (data acquisition + split,
preprocessing, features, models) done. This file currently defines:
  - DATASET_VARIANTS / DataConfig: dataset scale + derived paths
  - SplitConfig: train/val split parameters
  - FilterConfig / ArtifactConfig / NormalizationConfig: preprocessing
    variant selection + that variant's parameters
  - FeatureConfig: feature-extraction variant selection + parameters
  - LDAParams / SVMParams / RandomForestParams / EEGNetParams / TaskInfo
    / TrainingConfig / PermutationTestConfig / ModelConfig: model selection,
    training, and evaluation parameters
  - PipelineConfig / build_config(): wiring for all of the above

This is every step of the original pipeline - no further sections are
currently planned. If a new step is added later, it follows the same
shape as everything here: a dataclass for that step's parameters, a
`<step>.variant` field naming which registered function to use (see
src/utils/registry.py) where the step is pluggable, and a slot in
PipelineConfig.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset variant registry
#
# Not a src.utils.registry.Registry (that's for swapping FUNCTIONS - filter
# methods, models, etc). This is a plain lookup table of DATA describing
# each known dataset release, since there's no "alternative implementation"
# to select between - just different releases with different known sizes.
# ---------------------------------------------------------------------------

@dataclass
class DatasetVariantInfo:
    """Everything about one MindBigData release that has to be known up
    front (not derivable from the data itself, since we stream it rather
    than reading a local file with a header)."""
    hf_dataset_name: str | None          # Hugging Face repo id; None = not yet confirmed, see "8B" below
    total_train_digits_per_class: int | None
    total_test_digits_per_class: int | None
    n_samples: int | None                # samples/trial differs by variant (256 for 2B, 500 for 8B)
    description: str = ""


DATASET_VARIANTS: dict[str, DatasetVariantInfo] = {
    "2B": DatasetVariantInfo(
        hf_dataset_name="DavidVivancos/MindBigData2023_MNIST-2B",
        # Directly from the HF dataset card: train.csv = 120,000 rows,
        # test.csv = 20,000 rows. Rows are 50/50 blank/digit (pairing
        # structure - see data/acquisition.py), so digit trials/class =
        # rows / 2 / 10.
        total_train_digits_per_class=6000,    # 120,000 / 2 / 10
        total_test_digits_per_class=1000,       # 20,000 / 2 / 10
        n_samples=256,                            # reduced from 500 (per HF card)
        description="Reduced 2-billion-datapoint MindBigData2023 MNIST release (Hugging Face)",
    ),
    "8B": DatasetVariantInfo(
        hf_dataset_name=None,   # STILL NOT CONFIRMED - repo id not yet located in HF documentation
        # Row counts per the same HF documentation, stated to apply to both
        # the 2B and 8B releases - trial counts are known even though the
        # repo id isn't.
        total_train_digits_per_class=6000,
        total_test_digits_per_class=1000,
        n_samples=500,   # original (un-reduced) sample count per trial - documented, not reduced
        description="Full un-reduced 8-billion-datapoint release - repo id not yet confirmed, "
                     "trial counts inferred from shared documentation with the 2B release",
    ),
}


@dataclass
class DataConfig:
    """
    Controls dataset acquisition scale and where acquired/processed data
    lives on disk. The only two fields a user should ever set directly are
    dataset_variant and subsample_fraction - everything else on this class
    is either a fixed constant or a derived @property.
    """
    # --- The only two params a user sets to control dataset scale ---
    dataset_variant: str = "2B"          # key into DATASET_VARIANTS
    subsample_fraction: float = 0.20     # 1.0 = full dataset; 0.20 = 20% subsample

    n_channels_nominal: int = 128
    sample_rate_hz: float = 250.0
    max_stream_multiplier: int = 20      # safety cap multiplier on rows SCANNED during acquisition (not rows kept)

    data_root: Path = field(default_factory=lambda: Path("data"))

    # --- Derived from the two params above - never set these directly ---

    @property
    def variant_info(self) -> DatasetVariantInfo:
        if self.dataset_variant not in DATASET_VARIANTS:
            raise ValueError(f"Unknown dataset_variant {self.dataset_variant!r}. "
                              f"Known variants: {list(DATASET_VARIANTS)}")
        return DATASET_VARIANTS[self.dataset_variant]

    @property
    def hf_dataset_name(self) -> str:
        """Hugging Face repo id for acquisition.acquire_split(). Raises
        clearly (rather than silently guessing a repo id) if the selected
        variant doesn't have one confirmed yet - currently true for '8B'."""
        info = self.variant_info
        if info.hf_dataset_name is None:
            raise NotImplementedError(
                f"dataset_variant={self.dataset_variant!r} has no confirmed Hugging Face "
                "repo id yet - fill in DATASET_VARIANTS before using this variant."
            )
        return info.hf_dataset_name

    @property
    def target_per_class(self) -> int | None:
        """Digit trials per class to acquire for TRAIN. None means "no cap
        - collect everything available" (used when subsample_fraction=1.0)."""
        info = self.variant_info
        if self.subsample_fraction >= 1.0:
            return None
        if info.total_train_digits_per_class is None:
            raise NotImplementedError(
                f"dataset_variant={self.dataset_variant!r} has no known total trial count - "
                "cannot compute a fractional subsample. Use subsample_fraction=1.0, or fill "
                "in DATASET_VARIANTS."
            )
        return int(info.total_train_digits_per_class * self.subsample_fraction)

    @property
    def test_target_per_class(self) -> int | None:
        """Same idea as target_per_class, but for the held-out TEST split."""
        info = self.variant_info
        if self.subsample_fraction >= 1.0:
            return None
        if info.total_test_digits_per_class is None:
            raise NotImplementedError(
                f"dataset_variant={self.dataset_variant!r} has no known test trial count."
            )
        return int(info.total_test_digits_per_class * self.subsample_fraction)

    @property
    def n_samples(self) -> int:
        info = self.variant_info
        if info.n_samples is None:
            raise NotImplementedError(
                f"dataset_variant={self.dataset_variant!r} has no known n_samples."
            )
        return info.n_samples

    @property
    def variant_tag(self) -> str:
        """Folder-safe tag encoding variant + scale - used for every derived
        data path below, so different scales NEVER collide or overwrite
        each other on disk (e.g. a 20% run and a 100% run stay in separate
        folders)."""
        frac_str = "full" if self.subsample_fraction >= 1.0 else f"frac{self.subsample_fraction:.2f}"
        return f"{self.dataset_variant}_{frac_str}"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"   # intentionally empty - data is streamed, never stored raw locally

    @property
    def interim_dir(self) -> Path:
        """Where the acquired (but not yet split) train pool lives."""
        return self.data_root / "interim" / self.variant_tag

    @property
    def splits_dir(self) -> Path:
        """Where train/val/test HDF5 files live after splitting. NOT
        scoped by anything downstream of splitting (preprocessing stage,
        model choice, etc) - later steps read from here and write their
        own outputs to their own directories."""
        return self.data_root / "processed" / self.variant_tag / "splits"

    @property
    def filtered_dir(self) -> Path:
        """Where preprocessing (filter + normalize + artifact-flag)
        writes its train/val output. Added in Step 2."""
        return self.data_root / "processed" / self.variant_tag / "filtered"

    @property
    def preprocessing_params_dir(self) -> Path:
        """Where fitted preprocessing params (normalization center/scale,
        bad channels, artifact thresholds) are saved.

        Deliberately scoped by data variant + preprocessing variant choice
        ONLY - not by model. The original codebase nested this under model
        config (`cfg.model.preprocessing_dir`), which doesn't reflect
        reality: these fitted params don't depend on which model will
        later consume the resulting features, so a model-scoped path
        would need needlessly re-fitting the same preprocessing per model."""
        return self.data_root / "processed" / self.variant_tag / "preprocessing_params"

    @property
    def features_dir(self) -> Path:
        """Where extracted features (train_features.h5 / val_features.h5)
        live. Added in Step 3. Computed from Step 1's RAW split output
        (splits_dir), not Step 2's filtered output - see
        src/features/extraction.py's module docstring for why."""
        return self.data_root / "processed" / self.variant_tag / "features"


@dataclass
class SplitConfig:
    """Controls the train/val split (see data/train_val_split.py)."""
    val_fraction: float = 0.20   # target fraction of BLOCKS (not trials) assigned to val
    seed: int = 42


@dataclass
class FilterConfig:
    """Selects + parameterizes a filtering variant (see
    src/preprocessing/filters.py). variant must be a name registered in
    FILTER_REGISTRY; the rest of the fields are that variant's own
    parameters - a different variant might not use all (or any) of them."""
    variant: str = "butterworth_bandpass"
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    notch_freq_hz: float = 60.0
    apply_notch: bool = False
    filter_order: int = 4


@dataclass
class ArtifactConfig:
    """Selects + parameterizes an artifact-detection variant (see
    src/preprocessing/artifacts.py). variant must be a name registered in
    ARTIFACT_REGISTRY."""
    variant: str = "percentile_multi_criteria"
    artifact_percentile: float = 99.5
    flatline_percentile: float = 0.5
    trial_concern_min_channels: int = 3
    bad_channel_scale_floor: float = 1e-7


@dataclass
class NormalizationConfig:
    """Selects a normalization variant (see
    src/preprocessing/normalization.py). variant must be a name registered
    in NORMALIZATION_REGISTRY. No tunable parameters yet - the current
    variant (median/MAD) is parameterless; a future variant might add some."""
    variant: str = "robust_median_mad"


@dataclass
class FeatureConfig:
    """Selects + parameterizes a feature-extraction variant (see
    src/features/extraction.py). variant must be a name registered in
    FEATURE_REGISTRY; the rest of the fields are that variant's own
    parameters."""
    variant: str = "band_power_stat_freq"
    eeg_bands: dict = field(default_factory=lambda: {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 40), "high_gamma": (40, 80),
    })
    welch_nperseg: int = 128


@dataclass
class LDAParams:
    pass   # LDA has no hyperparameters currently tuned


@dataclass
class SVMParams:
    kernel: str = "rbf"


@dataclass
class RandomForestParams:
    n_estimators: int = 200


@dataclass
class EEGNetParams:
    # F1=16/F2=32 empirically beat F1=8/F2=16 (EEGNet's more common default)
    # on the harder 10-way multiclass task, evaluated with the real
    # block-aware split. Tripling further (F1=24/F2=48) clearly overfit -
    # this is the validated sweet spot, not an arbitrary choice.
    F1: int = 16
    D: int = 2
    F2: int = 32
    kernel_length: int = 64
    dropout: float = 0.5

    # --- Only relevant to the three backbone-reuse model variants below
    # (eegnet_frozen_backbone_reuse / eegnet_finetuned_backbone_reuse /
    # eegnet_dual_backbone_auxiliary_input) - ignored by eegnet_fresh. ---

    # Which task's trained eegnet_fresh checkpoint to reuse. In every run
    # so far this has been "binary" (multiclass reusing a binary-trained
    # backbone) - kept as a config field rather than hardcoded so a future
    # task could reuse from a different source task without a code change.
    reuse_source_task: str = "binary"
    # eegnet_finetuned_backbone_reuse only: epochs spent training just the
    # new head with the backbone frozen, before unfreezing.
    reuse_freeze_epochs: int = 15
    # eegnet_finetuned_backbone_reuse only: fine-tune phase learning rate,
    # as a fraction of the base training LR - kept low to avoid
    # catastrophic forgetting of the reused backbone's learned features.
    reuse_finetune_lr_multiplier: float = 0.1
    # eegnet_finetuned_backbone_reuse only: epochs after unfreezing.
    reuse_finetune_epochs: int = 15


# ---------------------------------------------------------------------------
# Task registry - a plain lookup table (like DATASET_VARIANTS above), not a
# src.utils.registry.Registry: there's no alternative IMPLEMENTATION to
# choose between per task, just data about what each task is (how many
# classes). Add a third task here (n_classes + description) if one ever
# comes up - nothing elsewhere needs to change beyond that.
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    n_classes: int
    description: str


TASK_INFO: dict[str, TaskInfo] = {
    "binary": TaskInfo(n_classes=2, description="blank vs. digit"),
    "multiclass": TaskInfo(n_classes=10, description="digit 0-9"),
}


# Kept here as a plain set, deliberately NOT derived from MODEL_REGISTRY:
# every domain module in this project imports FROM config.py, never the
# reverse - config.py has zero src.* imports anywhere. Deriving this from
# the registry would create a circular import (config -> models.factory ->
# config). Keep this set in sync by hand when a new model_name is
# registered in src/models/factory.py.
#
# Every name here is fully self-contained - "which model, and which
# strategy" in one string, not split across model_name + a separate
# variant field. eegnet_fresh works for EITHER task (no reuse involved);
# the other three eegnet_* names are multiclass-only reuse strategies (see
# EEGNetParams.reuse_source_task above and models/factory.py for what each
# one actually does).
CLASSICAL_MODEL_NAMES = {"lda", "svm", "random_forest"}
DEEP_MODEL_NAMES = {
    "eegnet_fresh",
    "eegnet_frozen_backbone_reuse",
    "eegnet_finetuned_backbone_reuse",
    "eegnet_dual_backbone_auxiliary_input",
}


@dataclass
class TrainingConfig:
    """Only used when model_name is a deep model (any eegnet_* name)."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_epochs: int = 30
    early_stop_patience: int = 7


@dataclass
class PermutationTestConfig:
    n_permutations_classical: int = 10
    n_permutations_deep: int = 5
    deep_subsample_size: int = 3000
    deep_quick_epochs: int = 10

    # Permutation testing only ever needs a rough noise-floor estimate, not
    # production-quality precision - these settings trade accuracy for
    # speed SPECIFICALLY for the permutation harness. The real model
    # evaluation (src/steps/models.py's main training call) never uses these.
    classical_subsample_size: int | None = 5000   # None = use full train set
    svm_permutation_max_iter: int = 300             # RBF SVC on shuffled labels often
                                                        # fails to converge, grinding through
                                                        # the full iteration budget every run -
                                                        # capped LinearSVC avoids this (see
                                                        # models/factory.py's build_for_permutation).


@dataclass
class ModelConfig:
    """
    Selects + parameterizes model training. model_name must be a name
    registered in src/models/factory.py's MODEL_REGISTRY - it fully
    identifies both WHICH model and, for EEGNet, which backbone-reuse
    strategy (if any); there is no separate "variant" field to also set.
    """
    # --- The two params a user sets to control modeling ---
    model_name: str = "random_forest"   # any name in CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES (see above)
    task: str = "binary"                 # "binary" | "multiclass" - see TASK_INFO

    lda: LDAParams = field(default_factory=LDAParams)
    svm: SVMParams = field(default_factory=SVMParams)
    random_forest: RandomForestParams = field(default_factory=RandomForestParams)
    eegnet: EEGNetParams = field(default_factory=EEGNetParams)

    model_root: Path = field(default_factory=lambda: Path("models"))

    def __post_init__(self):
        if self.model_name not in CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. "
                              f"Known: {sorted(CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES)}")
        if self.task not in TASK_INFO:
            raise ValueError(f"Unknown task {self.task!r}. Known: {sorted(TASK_INFO)}")

    @property
    def is_deep(self) -> bool:
        return self.model_name in DEEP_MODEL_NAMES

    @property
    def n_classes(self) -> int:
        return TASK_INFO[self.task].n_classes

    def run_tag(self, dataset_variant_tag: str) -> str:
        """Unique tag for this exact (dataset scale, task, model)
        combination - used for checkpoints and results. model_name alone
        distinguishes e.g. eegnet_fresh from eegnet_frozen_backbone_reuse,
        so no separate variant suffix is needed the way the old
        stage-based naming required."""
        return f"{dataset_variant_tag}__{self.task}__{self.model_name}"

    def checkpoint_path(self, dataset_variant_tag: str) -> Path:
        ext = "pt" if self.is_deep else "pkl"
        d = self.model_root / "checkpoints" / self.run_tag(dataset_variant_tag)
        return d / f"{self.model_name}.{ext}"

    def results_dir(self, dataset_variant_tag: str) -> Path:
        return self.model_root / "results" / self.run_tag(dataset_variant_tag)


@dataclass
class PipelineConfig:
    """
    Top-level config object, built by build_config(). Wires together every
    step built so far: data/split/filter/artifact/normalization/feature
    (Steps 1-3) plus model/training/permutation (Step 4).
    """
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    permutation: PermutationTestConfig = field(default_factory=PermutationTestConfig)
    seed: int = 42   # top-level seed; split.seed mirrors this by default via build_config()


def build_config(
    dataset_variant: str = "2B",
    subsample_fraction: float = 0.20,
    seed: int = 42,
    filter_variant: str = "butterworth_bandpass",
    artifact_variant: str = "percentile_multi_criteria",
    normalization_variant: str = "robust_median_mad",
    feature_variant: str = "band_power_stat_freq",
    task: str = "binary",
    model_name: str = "random_forest",
) -> PipelineConfig:
    """
    The intended entry point for users: set these parameters, get back a
    fully-wired config. Everything else (paths, trial counts, HF repo id,
    which training path runs) derives automatically.

    task: "binary" (blank vs. digit) or "multiclass" (digit 0-9).
    model_name: any name in CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES (see
        above) - e.g. "lda", "svm", "random_forest", "eegnet_fresh",
        "eegnet_frozen_backbone_reuse", "eegnet_finetuned_backbone_reuse",
        "eegnet_dual_backbone_auxiliary_input". The three backbone-reuse
        names only make sense with task="multiclass" (see
        EEGNetParams.reuse_source_task) - eegnet_fresh works for either task.

    The *_variant parameters select which registered function/strategy
    each preprocessing/feature step uses - each defaults to the variant
    that was already in place before it had alternatives, so existing
    calls to build_config() don't need to change as more variants are
    added. Finer per-variant parameters (e.g. cfg.filter.bandpass_low_hz,
    cfg.model.eegnet.F1) aren't build_config() arguments - set them
    directly on the returned config if you need non-default values.
    """
    cfg = PipelineConfig()
    cfg.data.dataset_variant = dataset_variant
    cfg.data.subsample_fraction = subsample_fraction
    cfg.seed = seed
    cfg.split.seed = seed
    cfg.filter.variant = filter_variant
    cfg.artifact.variant = artifact_variant
    cfg.normalization.variant = normalization_variant
    cfg.feature.variant = feature_variant
    cfg.model.model_name = model_name
    cfg.model.task = task
    cfg.model.__post_init__()   # re-validate after manual assignment
    return cfg


# Default instance - `from src.config import CFG` for quick/interactive use;
# call build_config() directly when you need non-default parameters.
CFG = build_config()
