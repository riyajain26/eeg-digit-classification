"""
Run and sanity-check Step 2 of the pipeline: preprocessing (filter,
normalize, artifact-flag). Requires Step 1 to have already been run for
the same dataset_variant/subsample_fraction (scripts/run_data_preparation.py).

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --subsample-fraction 0.01
    python scripts/run_preprocessing.py --filter-variant butterworth_bandpass
    python scripts/run_preprocessing.py --force
"""

import argparse

import h5py

import _bootstrap  # noqa: F401  (adds project root to sys.path - must come before the `src` imports below)
from src.config import build_config
from src.steps.preprocessing import run_preprocessing


def _summarize(cfg) -> None:
    """Prints shapes/flag rates from the Step 2 outputs, so a human can
    eyeball that things look right. Not part of the pipeline - purely a
    convenience for running this script interactively."""
    print("\n=== Step 2 output summary ===")
    for label, path in [("train", cfg.data.filtered_dir / "train_filtered.h5"),
                         ("val", cfg.data.filtered_dir / "val_filtered.h5")]:
        with h5py.File(path, "r") as f:
            n_trials = f["eeg"].shape[0]
            n_concern = int(f["trial_concern"][:].sum())
            print(f"  {label:8s} {path}")
            print(f"    trials={n_trials}  trial_concern={n_concern} ({n_concern / max(n_trials, 1):.1%})  "
                  f"channels kept={f['channel_indices'].shape[0]}  eeg shape per trial={f['eeg'].shape[1:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 2: preprocessing.")
    parser.add_argument("--dataset-variant", default="2B", choices=["2B", "8B"])
    parser.add_argument("--subsample-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-variant", default="butterworth_bandpass")
    parser.add_argument("--artifact-variant", default="percentile_multi_criteria")
    parser.add_argument("--normalization-variant", default="robust_median_mad")
    parser.add_argument("--diagnostic-subsample-size", type=int, default=25000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = build_config(
        dataset_variant=args.dataset_variant,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
        filter_variant=args.filter_variant,
        artifact_variant=args.artifact_variant,
        normalization_variant=args.normalization_variant,
    )
    print(f"=== Step 2: preprocessing === filter={args.filter_variant!r} "
          f"artifact={args.artifact_variant!r} normalization={args.normalization_variant!r} force={args.force}")

    run_preprocessing(cfg, force=args.force,
                       diagnostic_subsample_size=args.diagnostic_subsample_size,
                       chunk_size=args.chunk_size)
    _summarize(cfg)


if __name__ == "__main__":
    main()
