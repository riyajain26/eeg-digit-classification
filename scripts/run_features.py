"""
Run and sanity-check Step 3 of the pipeline: feature extraction. Requires
Steps 1 and 2 to have already been run for the same dataset_variant/
subsample_fraction (run_data_preparation.py, then run_preprocessing.py).

Usage:
    python scripts/run_features.py
    python scripts/run_features.py --subsample-fraction 0.01
    python scripts/run_features.py --force
"""

import argparse

import h5py

import _bootstrap  # noqa: F401  (adds project root to sys.path - must come before the `src` imports below)
from src.config import build_config
from src.steps.features import run_features


def _summarize(cfg) -> None:
    print("\n=== Step 3 output summary ===")
    for label, path in [("train", cfg.data.features_dir / "train_features.h5"),
                         ("val", cfg.data.features_dir / "val_features.h5")]:
        with h5py.File(path, "r") as f:
            print(f"  {label:8s} {path}")
            print(f"    trials={f['features'].shape[0]}  feature_dim={f['features'].shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 3: feature extraction.")
    parser.add_argument("--dataset-variant", default="2B", choices=["2B", "8B"])
    parser.add_argument("--subsample-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-variant", default="band_power_stat_freq")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = build_config(
        dataset_variant=args.dataset_variant,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
        feature_variant=args.feature_variant,
    )
    print(f"=== Step 3: features === feature_variant={args.feature_variant!r} force={args.force}")

    run_features(cfg, force=args.force, chunk_size=args.chunk_size)
    _summarize(cfg)


if __name__ == "__main__":
    main()
