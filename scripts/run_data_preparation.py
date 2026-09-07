"""
Run and sanity-check Step 1 of the pipeline: data acquisition + train/val
split. Nothing beyond src/config.py, src/data/, and src/steps/data_preparation.py
is needed for this script to work - preprocessing, features, and models
don't exist yet in the modular rebuild.

Usage:
    python scripts/run_data_preparation.py
    python scripts/run_data_preparation.py --subsample-fraction 0.01   # quick smoke test
    python scripts/run_data_preparation.py --force                     # re-run even if outputs exist

Run from the project root (so the "data/" output directory lands next to
this script's parent, not somewhere unexpected) - i.e. `cd` to the
directory containing `src/` and `scripts/` before running.
"""

import argparse

import h5py

from src.config import build_config
from src.steps.data_preparation import run_data_preparation


def _summarize(cfg) -> None:
    """Prints shapes/counts from the Step 1 outputs, so a human can
    eyeball that things look right without opening the HDF5 files by
    hand. Not part of the pipeline itself - purely a convenience for
    running this script interactively."""
    train_pool_path = cfg.data.interim_dir / "train_pool.h5"
    train_path = cfg.data.splits_dir / "train.h5"
    val_path = cfg.data.splits_dir / "val.h5"
    test_path = cfg.data.splits_dir / "test.h5"

    print("\n=== Step 1 output summary ===")
    for label, path in [("train_pool (pre-split)", train_pool_path),
                         ("train", train_path), ("val", val_path), ("test", test_path)]:
        with h5py.File(path, "r") as f:
            n_trials = f["eeg"].shape[0]
            n_digit_trials = int((f["label_digit"][:] != -1).sum())
            print(f"  {label:24s} {path}")
            print(f"    trials={n_trials}  (digit={n_digit_trials}, blank={n_trials - n_digit_trials})  "
                  f"eeg shape per trial={f['eeg'].shape[1:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 1: data acquisition + train/val split.")
    parser.add_argument("--dataset-variant", default="2B", choices=["2B", "8B"])
    parser.add_argument("--subsample-fraction", type=float, default=0.20,
                         help="1.0 = full dataset. Use a small value (e.g. 0.01) for a quick smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                         help="Re-run acquisition/splitting even if outputs already exist on disk.")
    args = parser.parse_args()

    cfg = build_config(
        dataset_variant=args.dataset_variant,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
    )
    print(f"=== Step 1: data preparation === dataset_variant={args.dataset_variant!r} "
          f"subsample_fraction={args.subsample_fraction} seed={args.seed} force={args.force}")

    run_data_preparation(cfg, force=args.force)
    _summarize(cfg)


if __name__ == "__main__":
    main()
