"""
Step 2 orchestration: preprocessing (filtering, normalization, artifact
detection), applied to train and val.

Renamed/rebuilt from pipeline.py's run_phase4_preprocessing() - same
two-pass structure (small diagnostic subsample to FIT everything, then a
memory-safe chunked pass to APPLY it to the full train/val splits), but
now resolves each of filter/normalization/artifact-detection by name from
its registry (see src/preprocessing/*.py) instead of calling one hardcoded
function - swapping any of the three to a different registered variant
needs a one-line config change here, not a code change.

INTENTIONAL BEHAVIOR CHANGE from the original: feature extraction is NOT
done in the same pass anymore. The original fused it in here purely for
I/O efficiency (avoid reading each chunk from disk twice). Splitting it
into its own step (coming next) trades one extra chunked read of already-
on-disk filtered data for a much cleaner separation - each step reads the
previous step's output and writes its own, the same as every other step
in this pipeline. That extra read is still fully chunked, so it doesn't
reintroduce the full-array-in-memory crash the original chunking was
built to avoid.

test.h5 is deliberately NOT processed here - preprocessing params are fit
on train only, and test should only ever be touched once, at final
evaluation time, with these same saved params (see save_normalization_params
/ save_artifact_params below). That happens in a later evaluation step,
not this one.
"""

import h5py
import numpy as np

from src.config import PipelineConfig
from src.preprocessing.artifacts import ARTIFACT_REGISTRY, exclude_channels, save_artifact_params
from src.preprocessing.filters import FILTER_REGISTRY
from src.preprocessing.normalization import (
    NORMALIZATION_REGISTRY, save_normalization_params,
)


def _append_chunk(dset, chunk) -> None:
    """Grows a resizable HDF5 dataset by chunk.shape[0] rows and writes
    `chunk` into it - the chunked-processing equivalent of
    data/acquisition.py's _append_row (one row at a time there; here,
    a whole chunk at once)."""
    old_size = dset.shape[0]
    dset.resize(old_size + chunk.shape[0], axis=0)
    dset[old_size:old_size + chunk.shape[0]] = chunk


def _filter_kwargs(cfg: PipelineConfig) -> dict:
    """Picks FilterConfig's fields relevant to calling the registered
    filter function as keyword arguments (`variant` itself is used to
    SELECT the function, not passed to it)."""
    fc = cfg.filter
    return dict(bandpass_low_hz=fc.bandpass_low_hz, bandpass_high_hz=fc.bandpass_high_hz,
                filter_order=fc.filter_order, apply_notch=fc.apply_notch, notch_freq_hz=fc.notch_freq_hz)


def _process_split_chunked(
    cfg: PipelineConfig, split_path, out_filtered_path,
    bad_channels, good_channels, center, scale, thresholds,
    chunk_size: int,
) -> None:
    """
    Streams one split (train or val) through filter -> normalize ->
    artifact-flag in chunks, writing each chunk to disk immediately -
    never holding the full split in memory. Required at 100% data scale,
    where loading a full split's EEG array at once previously crashed
    (~13GB for the filtered train array alone).
    """
    filter_fn = FILTER_REGISTRY.get(cfg.filter.variant)
    norm_strategy = NORMALIZATION_REGISTRY.get(cfg.normalization.variant)
    artifact_strategy = ARTIFACT_REGISTRY.get(cfg.artifact.variant)
    ac = cfg.artifact
    n_good = len(good_channels)

    with h5py.File(split_path, "r") as src:
        n_total = src["eeg"].shape[0]
        label_binary_all = src["label_binary"][:]
        label_digit_all = src["label_digit"][:]

        with h5py.File(out_filtered_path, "w") as f_filt:
            eeg_ds = f_filt.create_dataset("eeg", shape=(0, n_good, cfg.data.n_samples),
                                            maxshape=(None, n_good, cfg.data.n_samples),
                                            chunks=(1, n_good, cfg.data.n_samples),
                                            compression="gzip", dtype="float32")
            lb_ds = f_filt.create_dataset("label_binary", shape=(0,), maxshape=(None,), dtype="int8")
            ld_ds = f_filt.create_dataset("label_digit", shape=(0,), maxshape=(None,), dtype="int8")
            any_ds = f_filt.create_dataset("artifact_any", shape=(0, n_good), maxshape=(None, n_good), dtype="bool")
            nflag_ds = f_filt.create_dataset("n_flagged_channels_per_trial", shape=(0,), maxshape=(None,), dtype="int64")
            concern_ds = f_filt.create_dataset("trial_concern", shape=(0,), maxshape=(None,), dtype="bool")

            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                eeg_chunk = src["eeg"][start:end]
                eeg_chunk, _ = exclude_channels(eeg_chunk, bad_channels, cfg.data.n_channels_nominal)

                filtered_chunk = filter_fn(eeg_chunk, cfg.data.sample_rate_hz, **_filter_kwargs(cfg))
                norm_chunk = norm_strategy.apply(filtered_chunk, center, scale)
                arrays_chunk = artifact_strategy.compute_arrays(norm_chunk)
                flags_chunk = artifact_strategy.flag(arrays_chunk, thresholds, ac.trial_concern_min_channels)

                _append_chunk(eeg_ds, norm_chunk)
                _append_chunk(lb_ds, label_binary_all[start:end])
                _append_chunk(ld_ds, label_digit_all[start:end])
                _append_chunk(any_ds, flags_chunk["any"])
                _append_chunk(nflag_ds, flags_chunk["n_flagged_channels_per_trial"])
                _append_chunk(concern_ds, flags_chunk["trial_concern"])

                print(f"  processed {end}/{n_total} trials...")

            f_filt.create_dataset("channel_indices", data=good_channels)


def run_preprocessing(
    cfg: PipelineConfig,
    force: bool = False,
    diagnostic_subsample_size: int = 25000,
    chunk_size: int = 2000,
) -> None:
    """
    Full Step 2: fits filtering/normalization/artifact-detection on a
    train subsample, then applies the fitted result to the full train and
    val splits in memory-safe chunks. Requires Step 1
    (run_data_preparation) to have completed first.

    Args:
        cfg: must have cfg.filter.variant / cfg.artifact.variant /
            cfg.normalization.variant set to registered names (build_config()
            defaults are fine if you don't need to change them).
        force: re-run even if outputs already exist.
        diagnostic_subsample_size: trials used to fit bad-channel
            detection, normalization, and artifact thresholds. Kept small
            relative to the full dataset purely for speed - large enough
            for good percentile-threshold coverage, small enough to fit
            in memory comfortably.
        chunk_size: trials processed at once during the full pass - keeps
            peak memory roughly constant regardless of total dataset size.
    """
    cfg.data.filtered_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.preprocessing_params_dir.mkdir(parents=True, exist_ok=True)

    train_filtered_path = cfg.data.filtered_dir / "train_filtered.h5"
    val_filtered_path = cfg.data.filtered_dir / "val_filtered.h5"

    if not force and train_filtered_path.exists() and val_filtered_path.exists():
        print("Preprocessing: outputs already exist, skipping.")
        return

    train_path = cfg.data.splits_dir / "train.h5"
    val_path = cfg.data.splits_dir / "val.h5"
    if not train_path.exists() or not val_path.exists():
        raise RuntimeError(
            "Preprocessing requires run_data_preparation() to have completed first - "
            f"missing train.h5 or val.h5 under {cfg.data.splits_dir}."
        )

    filter_fn = FILTER_REGISTRY.get(cfg.filter.variant)
    norm_strategy = NORMALIZATION_REGISTRY.get(cfg.normalization.variant)
    artifact_strategy = ARTIFACT_REGISTRY.get(cfg.artifact.variant)
    ac = cfg.artifact

    # --- Diagnostic pass: fit everything on a train subsample ---
    with h5py.File(train_path, "r") as f:
        n_train = f["eeg"].shape[0]
        if n_train > diagnostic_subsample_size:
            rng = np.random.default_rng(cfg.seed)
            diag_idx = np.sort(rng.choice(n_train, size=diagnostic_subsample_size, replace=False))
            eeg_diag = f["eeg"][diag_idx]
        else:
            eeg_diag = f["eeg"][:]

    print(f"Preprocessing [{cfg.filter.variant}/{cfg.artifact.variant}/{cfg.normalization.variant}]: "
          f"diagnostic pass on {len(eeg_diag)} of {n_train} train trials...")

    eeg_diag_filtered_full = filter_fn(eeg_diag, cfg.data.sample_rate_hz, **_filter_kwargs(cfg))
    center_full, scale_full = norm_strategy.fit(eeg_diag_filtered_full)
    bad_channels = artifact_strategy.detect_bad_channels(scale_full, ac.bad_channel_scale_floor)
    print(f"Preprocessing: bad channels: {bad_channels}")

    eeg_diag, good_channels = exclude_channels(eeg_diag, bad_channels, cfg.data.n_channels_nominal)
    eeg_diag_filtered = eeg_diag_filtered_full[:, good_channels, :]
    del eeg_diag_filtered_full

    # Re-fit on good channels only - center_full/scale_full above was only
    # ever needed to DETECT which channels are bad in the first place.
    center, scale = norm_strategy.fit(eeg_diag_filtered)
    save_normalization_params(cfg.data.preprocessing_params_dir / "normalization_params.npz", center, scale)

    diag_norm = norm_strategy.apply(eeg_diag_filtered, center, scale)
    diag_arrays = artifact_strategy.compute_arrays(diag_norm)
    thresholds = artifact_strategy.derive_thresholds(diag_arrays, ac.artifact_percentile, ac.flatline_percentile)

    save_artifact_params(cfg.data.preprocessing_params_dir / "artifact_params.json", bad_channels, thresholds)
    del eeg_diag, eeg_diag_filtered, diag_norm, diag_arrays

    # --- Full pass: apply the fitted result to train, then val ---
    print(f"\nPreprocessing: processing train (chunk_size={chunk_size})...")
    _process_split_chunked(cfg, train_path, train_filtered_path,
                            bad_channels, good_channels, center, scale, thresholds, chunk_size)
    print(f"Preprocessing: wrote {train_filtered_path}")

    print(f"\nPreprocessing: processing val (chunk_size={chunk_size})...")
    _process_split_chunked(cfg, val_path, val_filtered_path,
                            bad_channels, good_channels, center, scale, thresholds, chunk_size)
    print(f"Preprocessing: wrote {val_filtered_path}")

    with h5py.File(train_filtered_path, "r") as f:
        print(f"\nPreprocessing: train trial_concern: {f['trial_concern'][:].sum()} / {f['trial_concern'].shape[0]}")
    with h5py.File(val_filtered_path, "r") as f:
        print(f"Preprocessing: val trial_concern: {f['trial_concern'][:].sum()} / {f['trial_concern'].shape[0]}")
