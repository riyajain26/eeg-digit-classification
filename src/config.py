"""
Central configuration.

Design principle (unchanged from the original pipeline, kept deliberately):
the user should only ever need to set a handful of top-level parameters,
and every path, count, and derived value follows automatically. Nothing
below should ever need hand-editing to change dataset scale.

BUILD STATUS - Step 1 of modularization (data acquisition + train/val
split) only. This file currently defines:
  - DATASET_VARIANTS / DataConfig: dataset scale + derived paths
  - SplitConfig: train/val split parameters
  - PipelineConfig / build_config(): wiring for the above two

Later steps will ADD to this same file (not replace it):
  - Step 2 (preprocessing): FilterConfig, ArtifactConfig, plus variant
    fields for filter/artifact-detection strategy
  - Step 3 (features): FeatureConfig, plus a feature-extraction variant field
  - Step 4 (models): ModelConfig and everything under it
  - Step 5+ (training/evaluation): TrainingConfig, PermutationTestConfig
Each addition follows the same shape as what's here: a dataclass for that
step's parameters, a `<step>.variant` field naming which registered
function to use (see src/utils/registry.py), and a slot in PipelineConfig.
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


@dataclass
class SplitConfig:
    """Controls the train/val split (see data/train_val_split.py)."""
    val_fraction: float = 0.20   # target fraction of BLOCKS (not trials) assigned to val
    seed: int = 42


@dataclass
class PipelineConfig:
    """
    Top-level config object, built by build_config(). Currently wires
    together DataConfig and SplitConfig only (Step 1 of modularization) -
    later steps will add their own dataclass fields here (filter, artifact,
    feature, model, training, permutation), matching the pattern already
    used for data/split.
    """
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 42   # top-level seed; split.seed mirrors this by default via build_config()


def build_config(
    dataset_variant: str = "2B",
    subsample_fraction: float = 0.20,
    seed: int = 42,
) -> PipelineConfig:
    """
    The intended entry point for users: set these parameters, get back a
    fully-wired config. Everything else (paths, trial counts, HF repo id)
    derives automatically.

    NOTE: this signature will grow as later modularization steps add their
    own config sections (preprocessing/feature/model variants) - each new
    parameter will have a sensible default, so existing calls to
    build_config() keep working unchanged as the pipeline grows.
    """
    cfg = PipelineConfig()
    cfg.data.dataset_variant = dataset_variant
    cfg.data.subsample_fraction = subsample_fraction
    cfg.seed = seed
    cfg.split.seed = seed
    return cfg


# Default instance - `from src.config import CFG` for quick/interactive use;
# call build_config() directly when you need non-default parameters.
CFG = build_config()
