# src/ — Parameterized EEG Digit Classification Pipeline

## The 5 parameters that control everything

```python
from src.pipeline import main

main(
    dataset_variant="2B",        # "2B" | "8B" (8B not yet available - see below)
    subsample_fraction=0.20,     # 1.0 = full dataset; 0.20 = 20% subsample
    stage="stage1",              # "stage1" (binary) | "stage2" (digit 0-9)
    model_name="random_forest",  # "lda" | "svm" | "random_forest" | "eegnet"
    seed=42,
)
```

Every path, trial count, HF repo id, training code path, and output
location is derived from these. Nothing else needs editing to change
scale, stage, or model.

- **Scaling the dataset**: change `subsample_fraction` only. `1400` (the
  old hardcoded 20%-subsample count) is now `int(7000 * 0.20)`, derived in
  `config.py`'s `DataConfig.target_per_class` property — `7000` is the
  known total digits/class for the 2B release, sourced from prior research,
  not guessed.
- **Adding Stage 2**: change `stage` only. Every phase before modeling
  (acquisition, splitting, preprocessing, features) is stage-agnostic —
  only `pipeline.py`'s `_load_stage_data_*` functions branch on it, at read
  time, filtering to digit-only trials for stage2. Verified empirically:
  running stage2 after stage1 triggers **zero** re-processing of Phase 4's
  output — same filtered/feature files, just a different label column and
  trial subset.
- **Switching models**: change `model_name` only. `models/factory.py` is
  the single place this gets resolved into an actual model instance;
  `pipeline.py`'s `run_phase5_6_model()` branches on `cfg.model.is_deep` to
  pick the classical (sklearn) or deep (PyTorch/EEGNet) training path.

## On the 8B variant

`DATASET_VARIANTS["8B"]` in `config.py` is a **deliberate placeholder** —
`hf_dataset_name=None`. Using it raises `NotImplementedError` immediately
rather than silently guessing a repo id or trial count. Fill in the real
values in that registry entry once confirmed, and nothing else in the
codebase needs to change.

## Path scoping — what's shared vs. what isn't

- `data/processed/<variant_tag>/{splits,filtered,features}/` — scoped by
  **dataset scale only** (e.g. `2B_frac0.20` vs `2B_full`). NOT scoped by
  stage or model, since Stage 2 and every model type consume the exact same
  preprocessed data, just filtered/interpreted differently at read time.
- `models/preprocessing/<variant_tag>/` — fitted normalization/artifact
  params. Same scoping as data (shared across stage/model) — verified this
  is written exactly once regardless of how many stage/model combinations
  run against the same `variant_tag`.
- `models/checkpoints/<variant_tag>__<stage>__<model_name>/` and
  `models/results/<variant_tag>__<stage>__<model_name>/` — scoped by all
  three, since these genuinely differ per combination. Verified: 3 runs
  (stage1/random_forest, stage2/lda, stage1/eegnet) produce 3 distinct
  checkpoint+results folders, never colliding.

## Structure

```
src/
├── config.py                    # dataset registry, all params, build_config()
├── pipeline.py                   # orchestration: Phase 2 -> 3 -> 4 -> 5/6
├── data/
│   ├── conversion.py              # HF streaming (target_per_class=None = full dataset)
│   └── splitting.py                # block-aware leakage-safe split
├── preprocessing/
│   ├── filters.py, normalization.py, artifacts.py, frequency_audit.py
├── features/
│   └── extraction.py               # Path A: band power / stats / frequency
├── models/
│   ├── eegnet.py                    # architecture only
│   └── factory.py                    # NEW: resolves model_name -> model instance
├── training/
│   └── loop.py                       # generic training loop (any model)
├── evaluation/
│   ├── metrics.py                     # stage-aware averaging (binary vs macro)
│   └── permutation_test.py             # sklearn + torch versions
└── utils/
    └── reproducibility.py
```

## Verified end-to-end (this session, synthetic data)

- Config derivation: `target_per_class`, `variant_tag`, path collisions
  avoided across scales, 8B placeholder fails loudly, invalid
  `model_name`/`stage` rejected at construction.
- Model factory: all 3 classical models build correctly; EEGNet rejected
  by the classical builder; EEGNet builds correctly via `build_eegnet()`
  with the right output class count for stage2 (10) vs stage1 (2).
- Full pipeline, classical path: Phase 3 → 4 → 5 ran end-to-end
  (synthetic random data, ~0.50 accuracy — correct, since there's no real
  signal in noise).
- Stage 2 (LDA) reused Phase 4's existing output — no re-processing
  triggered.
- EEGNet (Phase 6): trained with checkpointing, evaluated, ran its
  permutation test, saved results — all on the shared preprocessing output.
- On-disk scoping confirmed exactly as designed (see above).

## Known gaps / deliberately deferred

- **Test-set preprocessing** (Phase 7) isn't wired into `pipeline.py` yet —
  it needs to load `models/preprocessing/<variant_tag>/*` and apply
  (never re-fit) to `splits_dir/test.h5`. Straightforward given the saved
  params, but not yet implemented.
- **8B variant** needs real `hf_dataset_name` and trial-count numbers filled
  into `DATASET_VARIANTS` before it can be used.
- **Notebook 04's exploratory visualizations** (frequency heatmap, filter
  edge-effect plot) remain notebook-only, not wrapped as pipeline functions
  — they're for visual inspection, not automated logic.
