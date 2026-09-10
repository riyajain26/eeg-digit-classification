"""
Run the FULL pipeline end-to-end: data preparation -> preprocessing ->
features (if needed) -> model training/evaluation. This is what chains
Steps 1-4 (scripts/run_data_preparation.py through scripts/run_models.py)
into one script - those four still work standalone if you want to inspect
or re-run a single step in isolation.

Usage:
    python scripts/run_pipeline.py --model-name random_forest --task multiclass
    python scripts/run_pipeline.py --model-name eegnet_frozen_backbone_reuse --task multiclass
        (auto-trains a binary eegnet_fresh prerequisite first if it doesn't exist yet)
    python scripts/run_pipeline.py --interactive
        (prompts for each pluggable step's variant at the terminal, Enter keeps the default)
"""

import argparse

import _bootstrap  # noqa: F401  (adds project root to sys.path - must come before the `src` imports below)
from src.config import CLASSICAL_MODEL_NAMES, DEEP_MODEL_NAMES, build_config
from src.pipeline import prompt_for_variants, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline, Steps 1-4.")
    parser.add_argument("--dataset-variant", default="2B", choices=["2B", "8B"])
    parser.add_argument("--subsample-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-variant", default="butterworth_bandpass")
    parser.add_argument("--artifact-variant", default="percentile_multi_criteria")
    parser.add_argument("--normalization-variant", default="robust_median_mad")
    parser.add_argument("--feature-variant", default="band_power_stat_freq")
    parser.add_argument("--model-name", default="random_forest",
                         choices=sorted(CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES))
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--reuse-source-task", default="binary", choices=["binary", "multiclass"],
                         help="Only used by *_backbone_reuse / *_auxiliary_input model names.")
    parser.add_argument("--interactive", action="store_true",
                         help="Prompt for each pluggable step's variant at the terminal.")
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Re-run every step even if outputs already exist.")
    args = parser.parse_args()

    cfg = build_config(
        dataset_variant=args.dataset_variant,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
        filter_variant=args.filter_variant,
        artifact_variant=args.artifact_variant,
        normalization_variant=args.normalization_variant,
        feature_variant=args.feature_variant,
        model_name=args.model_name,
        task=args.task,
    )
    cfg.model.eegnet.reuse_source_task = args.reuse_source_task

    if args.interactive:
        cfg = prompt_for_variants(cfg)

    print(f"=== Full pipeline === task={cfg.model.task!r} model={cfg.model.model_name!r} "
          f"dataset={cfg.data.variant_tag!r} run_permutation={not args.no_permutation}")

    results = run_pipeline(cfg, force=args.force, run_permutation=not args.no_permutation)

    print("\n=== Pipeline complete ===")
    print(results["metrics"])
    if results.get("permutation_test"):
        print(results["permutation_test"])


if __name__ == "__main__":
    main()
