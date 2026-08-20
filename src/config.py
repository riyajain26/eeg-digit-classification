"""
Central configuration.

Design principle (per project decision): the user should only ever need to
set a handful of top-level parameters - dataset_variant, subsample_fraction,
stage, model_name - and every path, count, and derived value follows
automatically. Nothing below should ever need hand-editing to change scale,
stage, or model; that's what build_config() is for.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset variant registry
# ---------------------------------------------------------------------------

@dataclass
class DatasetVariantInfo:
    hf_dataset_name: str | None
    total_train_digits_per_class: int | None
    total_test_digits_per_class: int | None
    n_samples: int | None            # NEW: samples/trial differs by variant (256 vs 500)
    description: str = ""


DATASET_VARIANTS: dict[str, DatasetVariantInfo] = {
    "2B": DatasetVariantInfo(
        hf_dataset_name="DavidVivancos/MindBigData2023_MNIST-2B",
        # Directly from the HF dataset card: train.csv = 120,000 rows,
        # test.csv = 20,000 rows. Rows are 50/50 blank/digit (pairing
        # structure), so digit trials/class = rows / 2 / 10.
        total_train_digits_per_class=6000,    # 120,000 / 2 / 10
        total_test_digits_per_class=1000,       # 20,000 / 2 / 10
        n_samples=256,                            # reduced from 500 (per HF card)
        description="Reduced 2-billion-datapoint MindBigData2023 MNIST release (Hugging Face)",
    ),
    "8B": DatasetVariantInfo(
        hf_dataset_name=None,   # STILL NOT CONFIRMED - row-count docs don't confirm a repo exists
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
    # --- The only two params a user sets to control dataset scale ---
    dataset_variant: str = "2B"          # key into DATASET_VARIANTS
    subsample_fraction: float = 0.20     # 1.0 = full dataset; 0.20 = 20% subsample

    n_channels_nominal: int = 128
    sample_rate_hz: float = 250.0
    max_stream_multiplier: int = 20      # safety cap multiplier on rows SCANNED (not kept)

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
        info = self.variant_info
        if info.hf_dataset_name is None:
            raise NotImplementedError(
                f"dataset_variant={self.dataset_variant!r} has no confirmed Hugging Face "
                "repo id yet - fill in DATASET_VARIANTS before using this variant."
            )
        return info.hf_dataset_name

    @property
    def target_per_class(self) -> int | None:
        """None means 'no cap - collect everything available' (full dataset)."""
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
        each other on disk."""
        frac_str = "full" if self.subsample_fraction >= 1.0 else f"frac{self.subsample_fraction:.2f}"
        return f"{self.dataset_variant}_{frac_str}"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"   # intentionally empty - streamed, not stored locally

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim" / self.variant_tag

    @property
    def splits_dir(self) -> Path:
        # NOT stage-scoped: Stage 2 is a filtered VIEW of this same data
        # (digit-only trials), not a separate copy - see splitting/features
        # loading logic, which branches on stage at READ time.
        return self.data_root / "processed" / self.variant_tag / "splits"

    @property
    def filtered_dir(self) -> Path:
        return self.data_root / "processed" / self.variant_tag / "filtered"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "processed" / self.variant_tag / "features"


@dataclass
class SplitConfig:
    val_fraction: float = 0.20
    seed: int = 42


@dataclass
class FilterConfig:
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    notch_freq_hz: float = 60.0
    apply_notch: bool = False
    filter_order: int = 4
    approx_edge_samples: int = 27


@dataclass
class ArtifactConfig:
    artifact_percentile: float = 99.5
    flatline_percentile: float = 0.5
    trial_concern_min_channels: int = 3
    bad_channel_scale_floor: float = 1e-7


@dataclass
class FeatureConfig:
    eeg_bands: dict = field(default_factory=lambda: {
        "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
        "beta": (13, 30), "gamma": (30, 40), "high_gamma": (40, 80),
    })
    welch_nperseg: int = 128


# ---------------------------------------------------------------------------
# Model selection + per-model hyperparameters
# ---------------------------------------------------------------------------

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
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout: float = 0.5


CLASSICAL_MODEL_NAMES = {"lda", "svm", "random_forest"}
DEEP_MODEL_NAMES = {"eegnet"}


@dataclass
class TrainingConfig:
    # Only used when model_name is a deep model (currently: eegnet).
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
    # (Section 5's actual evaluation) never uses these.
    classical_subsample_size: int | None = 5000   # None = use full train set
    svm_permutation_max_iter: int = 300             # RBF SVC on shuffled labels often
                                                        # fails to converge, grinding through
                                                        # the full iteration budget every run -
                                                        # capped LinearSVC avoids this.


@dataclass
class Stage2Config:
    # Only relevant when stage="stage2" AND model_name="eegnet". Registry
    # key into STAGE2_MODEL_REGISTRY (models/factory.py) - "path_a" (fresh,
    # no reuse), "path_b1" (frozen backbone), "path_b2" (frozen then
    # fine-tuned). Adding a new variant later: write one function, add one
    # registry entry - no changes needed here or in pipeline.py.
    variant: str = "path_a"

    # Path B2 only: epochs spent training just the new head with the
    # backbone frozen, before unfreezing for the fine-tune phase.
    freeze_epochs: int = 15
    # Path B2 only: fine-tune phase learning rate, as a fraction of the
    # base training LR - kept low to avoid catastrophic forgetting of
    # Stage 1's learned features.
    finetune_lr_multiplier: float = 0.1
    finetune_epochs: int = 15


@dataclass
class ModelConfig:
    # --- The two params a user sets to control modeling ---
    model_name: str = "random_forest"    # "lda" | "svm" | "random_forest" | "eegnet"
    stage: str = "stage1"                 # "stage1" (binary) | "stage2" (digit, 0-9)

    lda: LDAParams = field(default_factory=LDAParams)
    svm: SVMParams = field(default_factory=SVMParams)
    random_forest: RandomForestParams = field(default_factory=RandomForestParams)
    eegnet: EEGNetParams = field(default_factory=EEGNetParams)
    stage2: Stage2Config = field(default_factory=Stage2Config)

    model_root: Path = field(default_factory=lambda: Path("models"))

    def __post_init__(self):
        if self.model_name not in CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES:
            raise ValueError(f"Unknown model_name {self.model_name!r}. "
                              f"Known: {CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES}")
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unknown stage {self.stage!r}. Expected 'stage1' or 'stage2'.")

    @property
    def is_deep(self) -> bool:
        return self.model_name in DEEP_MODEL_NAMES

    def run_tag(self, dataset_variant_tag: str) -> str:
        """Unique tag for this exact (dataset scale, stage, model[, stage2
        variant]) combination - used for checkpoints and results, so
        different Stage 2 EEGNet variants (A/B1/B2) never overwrite each
        other on disk."""
        tag = f"{dataset_variant_tag}__{self.stage}__{self.model_name}"
        if self.stage == "stage2" and self.model_name == "eegnet":
            tag += f"__{self.stage2.variant}"
        return tag

    def preprocessing_dir(self, dataset_variant_tag: str) -> Path:
        # Fitted normalization/artifact params depend on dataset scale ONLY
        # (not stage or model - preprocessing is identical regardless of
        # which model consumes its output).
        return self.model_root / "preprocessing" / dataset_variant_tag

    def checkpoint_path(self, dataset_variant_tag: str) -> Path:
        ext = "pt" if self.is_deep else "pkl"
        d = self.model_root / "checkpoints" / self.run_tag(dataset_variant_tag)
        return d / f"{self.model_name}.{ext}"

    def results_dir(self, dataset_variant_tag: str) -> Path:
        return self.model_root / "results" / self.run_tag(dataset_variant_tag)


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    permutation: PermutationTestConfig = field(default_factory=PermutationTestConfig)
    seed: int = 42   # top-level seed; split.seed mirrors this by default via build_config()


def build_config(
    dataset_variant: str = "2B",
    subsample_fraction: float = 0.20,
    stage: str = "stage1",
    model_name: str = "random_forest",
    stage2_variant: str = "path_a",
    seed: int = 42,
) -> PipelineConfig:
    """
    The intended entry point for users: set these ~6 parameters, get back a
    fully-wired config. Everything else (paths, trial counts, HF repo id,
    which training path runs) derives automatically.

    stage2_variant only matters when stage="stage2" and model_name="eegnet" -
    "path_a" (fresh), "path_b1" (frozen Stage 1 backbone), "path_b2"
    (frozen then fine-tuned).
    """
    cfg = PipelineConfig()
    cfg.data.dataset_variant = dataset_variant
    cfg.data.subsample_fraction = subsample_fraction
    cfg.model.stage = stage
    cfg.model.model_name = model_name
    cfg.model.stage2.variant = stage2_variant
    cfg.model.__post_init__()   # re-validate after manual assignment
    cfg.seed = seed
    cfg.split.seed = seed
    return cfg


# Default instance - `from src.config import CFG` for quick/interactive use;
# call build_config() directly when you need non-default parameters.
CFG = build_config()
