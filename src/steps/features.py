"""
Step 3 orchestration: feature extraction.

Renamed/rebuilt from pipeline.py's fused Phase 4 feature-extraction path
(previously computed inline inside the same chunked pass as filtering -
see src/steps/preprocessing.py's docstring for why that's been split
apart). Resolves the extraction method by name from FEATURE_REGISTRY
(see src/features/extraction.py) instead of calling one hardcoded function.

READS FROM cfg.data.splits_dir (Step 1's RAW output), NOT
cfg.data.filtered_dir (Step 2's filtered/normalized output). This is
intentional, not an oversight: Path A-style band-power features need the
unfiltered, wide-band signal to be meaningful (e.g. high_gamma, 40-80Hz,
would be zeroed out by Step 2's default 1-40Hz bandpass before these
features ever saw it). See src/features/extraction.py's module docstring
for the full explanation.

Still requires Step 2 to have run first, though - not for its filtered
EEG, but for the bad_channels list it saved (artifact_params.json),
so features are computed on the same reduced channel set Step 2 used
(keeping trial arrays comparable/poolable downstream regardless of
which step's output a later step reads).

Features are extracted for EVERY trial here - trial_concern-based
exclusion (from Step 2) is applied later, when a model-training step
loads this data, not here. That keeps this step's output a complete,
reusable record rather than one baked-in exclusion decision.
"""

import h5py
import numpy as np

from src.config import PipelineConfig
from src.features.extraction import FEATURE_REGISTRY
from src.preprocessing.artifacts import exclude_channels, load_artifact_params


def _append_chunk(dset, chunk) -> None:
    old_size = dset.shape[0]
    dset.resize(old_size + chunk.shape[0], axis=0)
    dset[old_size:old_size + chunk.shape[0]] = chunk


def _feature_kwargs(cfg: PipelineConfig) -> dict:
    """Picks FeatureConfig's fields relevant to calling the registered
    feature function as keyword arguments (`variant` itself is used to
    SELECT the function, not passed to it)."""
    return dict(bands=cfg.feature.eeg_bands, welch_nperseg=cfg.feature.welch_nperseg)


def _extract_split_chunked(cfg: PipelineConfig, split_path, out_features_path,
                             bad_channels, chunk_size: int) -> None:
    """Streams one split (train or val) through channel-exclusion +
    feature extraction in chunks, writing each chunk to disk immediately.

    Output datasets are created UPFRONT using a dummy zero-filled trial to
    determine feature dimensionality - not lazily from the first real
    chunk. This matters for splits that legitimately end up with zero
    trials (e.g. an unlucky random train/val block assignment on a small
    subsample): without this, such a split would silently produce a
    features file with no "features" dataset at all, breaking anything
    downstream that expects one. Mirrors src/steps/preprocessing.py's
    pattern, where output datasets are likewise created before the loop."""
    feature_fn = FEATURE_REGISTRY.get(cfg.feature.variant)
    fs = cfg.data.sample_rate_hz
    n_good_channels = cfg.data.n_channels_nominal - len(bad_channels)

    dummy_eeg = np.zeros((1, n_good_channels, cfg.data.n_samples), dtype=np.float32)
    n_feat = feature_fn(dummy_eeg, fs, **_feature_kwargs(cfg)).shape[1]

    with h5py.File(split_path, "r") as src:
        n_total = src["eeg"].shape[0]
        label_binary_all = src["label_binary"][:]
        label_digit_all = src["label_digit"][:]

        with h5py.File(out_features_path, "w") as f_feat:
            feat_ds = f_feat.create_dataset("features", shape=(0, n_feat), maxshape=(None, n_feat),
                                             compression="gzip", dtype="float32")
            feat_lb_ds = f_feat.create_dataset("label_binary", shape=(0,), maxshape=(None,), dtype="int8")
            feat_ld_ds = f_feat.create_dataset("label_digit", shape=(0,), maxshape=(None,), dtype="int8")

            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                eeg_chunk = src["eeg"][start:end]
                eeg_chunk, _ = exclude_channels(eeg_chunk, bad_channels, cfg.data.n_channels_nominal)

                features_chunk = feature_fn(eeg_chunk, fs, **_feature_kwargs(cfg))

                _append_chunk(feat_ds, features_chunk)
                _append_chunk(feat_lb_ds, label_binary_all[start:end])
                _append_chunk(feat_ld_ds, label_digit_all[start:end])

                print(f"  processed {end}/{n_total} trials...")


def run_features(cfg: PipelineConfig, force: bool = False, chunk_size: int = 2000) -> None:
    """
    Full Step 3: extracts features for train and val from Step 1's raw
    split output, using the bad-channel list Step 2 fitted. Requires
    Step 1 (run_data_preparation) and Step 2 (run_preprocessing) to have
    completed first.

    Args:
        cfg: must have cfg.feature.variant set to a registered name
            (build_config()'s default is fine if you don't need to change it).
        force: re-run even if outputs already exist.
        chunk_size: trials processed at once - keeps peak memory roughly
            constant regardless of total dataset size.
    """
    cfg.data.features_dir.mkdir(parents=True, exist_ok=True)

    train_features_path = cfg.data.features_dir / "train_features.h5"
    val_features_path = cfg.data.features_dir / "val_features.h5"

    if not force and train_features_path.exists() and val_features_path.exists():
        print("Features: outputs already exist, skipping.")
        return

    train_path = cfg.data.splits_dir / "train.h5"
    val_path = cfg.data.splits_dir / "val.h5"
    artifact_params_path = cfg.data.preprocessing_params_dir / "artifact_params.json"

    if not train_path.exists() or not val_path.exists():
        raise RuntimeError(
            "Features requires run_data_preparation() (Step 1) to have completed first - "
            f"missing train.h5 or val.h5 under {cfg.data.splits_dir}."
        )
    if not artifact_params_path.exists():
        raise RuntimeError(
            "Features requires run_preprocessing() (Step 2) to have completed first - "
            f"missing {artifact_params_path} (need its bad_channels list)."
        )

    bad_channels = load_artifact_params(artifact_params_path)["bad_channels"]
    print(f"Features [{cfg.feature.variant}]: using bad_channels={bad_channels} from Step 2.")

    print(f"\nFeatures: extracting train (chunk_size={chunk_size})...")
    _extract_split_chunked(cfg, train_path, train_features_path, bad_channels, chunk_size)
    print(f"Features: wrote {train_features_path}")

    print(f"\nFeatures: extracting val (chunk_size={chunk_size})...")
    _extract_split_chunked(cfg, val_path, val_features_path, bad_channels, chunk_size)
    print(f"Features: wrote {val_features_path}")
