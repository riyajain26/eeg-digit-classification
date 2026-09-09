"""
Run and sanity-check Step 4 of the pipeline: model training + evaluation
+ permutation testing. Requires Steps 1-3 to have already been run for
the same dataset_variant/subsample_fraction (Step 3 only needed for
classical models - see src/features/extraction.py's module docstring).

Usage:
    python scripts/run_models.py --model-name random_forest
    python scripts/run_models.py --model-name eegnet_fresh --task binary
    python scripts/run_models.py --model-name eegnet_frozen_backbone_reuse --task multiclass
    python scripts/run_models.py --model-name eegnet_fresh --task binary --no-permutation

Any *_backbone_reuse or *_auxiliary_input model requires a trained
eegnet_fresh checkpoint for the source task to already exist - e.g. before
running eegnet_frozen_backbone_reuse (task=multiclass), first run:
    python scripts/run_models.py --model-name eegnet_fresh --task binary
"""

import argparse

import _bootstrap  # noqa: F401  (adds project root to sys.path - must come before the `src` imports below)
from src.config import CLASSICAL_MODEL_NAMES, DEEP_MODEL_NAMES, build_config
from src.steps.models import run_model_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 4: model training + evaluation.")
    parser.add_argument("--dataset-variant", default="2B", choices=["2B", "8B"])
    parser.add_argument("--subsample-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="random_forest",
                         choices=sorted(CLASSICAL_MODEL_NAMES | DEEP_MODEL_NAMES))
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--reuse-source-task", default="binary", choices=["binary", "multiclass"],
                         help="Only used by *_backbone_reuse / *_auxiliary_input model names: "
                              "which task's trained eegnet_fresh checkpoint to reuse.")
    parser.add_argument("--no-permutation", action="store_true",
                         help="Skip the permutation (shuffled-label) test - useful for a quick run.")
    args = parser.parse_args()

    cfg = build_config(
        dataset_variant=args.dataset_variant,
        subsample_fraction=args.subsample_fraction,
        seed=args.seed,
        model_name=args.model_name,
        task=args.task,
    )
    cfg.model.eegnet.reuse_source_task = args.reuse_source_task

    print(f"=== Step 4: models === model={args.model_name!r} task={args.task!r} "
          f"run_permutation={not args.no_permutation}")

    results = run_model_training(cfg, run_permutation=not args.no_permutation)
    print("\n=== Final results ===")
    print(results["metrics"])
    if results.get("permutation_test"):
        print(results["permutation_test"])


if __name__ == "__main__":
    main()
