# Pipeline Walkthrough (Steps 1-4)

This covers everything built so far, as four separate runnable steps.
`pipeline.py` (not yet built) will chain these into one script - this doc
is what should make that integration easy to follow rather than a black box.

## The pattern, once, so it doesn't need repeating per step

Every pluggable step follows the same shape:

1. A **registry** (`src/utils/registry.py`'s `Registry` class) mapping a
   name -> a function (or small bundle of functions). Defined in the
   domain module the step belongs to (e.g. `src/preprocessing/filters.py`
   defines `FILTER_REGISTRY`).
2. A **config section** in `src/config.py` with a `variant` field (or, for
   models, just `model_name`) naming which registered entry to use, plus
   that variant's own parameters.
3. An **orchestrator** in `src/steps/<step>.py` - the only place that
   reads config, decides what's cached vs. needs (re)running, and calls
   the registry-resolved function(s) in the right order. This is what a
   future `pipeline.py` will call directly, one function per step.
4. A **script** in `scripts/run_<step>.py` - a thin CLI wrapper around the
   orchestrator, for running/testing that one step in isolation (which is
   what you've been doing).

Adding a new variant to any step = write one function, register it, set
the config string to select it. No other file changes.

## Directory structure

```
project/
├── src/
│   ├── config.py                 # every config dataclass, build_config()
│   ├── utils/registry.py         # the generic Registry class
│   ├── data/
│   │   ├── acquisition.py        # low-level: stream HF -> HDF5
│   │   └── train_val_split.py    # low-level: leakage-safe block split
│   ├── preprocessing/
│   │   ├── filters.py            # FILTER_REGISTRY
│   │   ├── normalization.py      # NORMALIZATION_REGISTRY
│   │   ├── artifacts.py          # ARTIFACT_REGISTRY
│   │   └── frequency_audit.py    # standalone diagnostic tool, no registry
│   ├── features/
│   │   └── extraction.py         # FEATURE_REGISTRY
│   ├── models/
│   │   ├── eegnet.py             # EEGNet, EEGNetDualBackbone architectures
│   │   └── factory.py            # MODEL_REGISTRY (classical + every EEGNet variant)
│   ├── training/
│   │   └── loop.py               # PyTorch training loop, no registry
│   ├── evaluation/
│   │   ├── metrics.py            # shared metric computation, no registry
│   │   └── permutation_test.py   # shuffled-label harnesses, no registry
│   └── steps/                    # ORCHESTRATORS - one per pipeline step
│       ├── data_preparation.py   # Step 1
│       ├── preprocessing.py      # Step 2
│       ├── features.py           # Step 3
│       └── models.py             # Step 4
└── scripts/
    ├── _bootstrap.py             # sys.path fix, imported first by every script below
    ├── run_data_preparation.py
    ├── run_preprocessing.py
    ├── run_features.py
    └── run_models.py
```

## Data flow across steps

```
Hugging Face (streamed)
       |
       v
[Step 1: data_preparation]
  data/interim/<tag>/train_pool.h5  (acquired, not yet split)
       |
       v  (block-aware split)
  data/processed/<tag>/splits/{train,val,test}.h5   <-- RAW, channel-complete
       |
       +---------------------------------------------+
       |                                              |
       v  (Step 2 reads train+val only;                v  (Step 3 reads train+val,
       |   test is untouched until final eval)          |   RAW - not Step 2's output)
       v                                                v
[Step 2: preprocessing]                        [Step 3: features]
  fit filter/normalize/artifact-detect            uses Step 2's saved bad_channels
  on a train subsample, apply to train+val          (artifact_params.json) only -
       |                                              NOT Step 2's filtered signal
       v                                                     |
  data/processed/<tag>/filtered/{train,val}_filtered.h5      v
  data/processed/<tag>/preprocessing_params/           data/processed/<tag>/features/
    normalization_params.npz, artifact_params.json       {train,val}_features.h5
       |                                                     |
       v  (deep models read this)                            v  (classical models read this)
       |                                                     |
       +---------------------------------------------+-------+
                                                       |
                                                       v
                                            [Step 4: models]
                                     models/checkpoints/<run_tag>/*.pt|.pkl
                                     models/results/<run_tag>/results.json
```

**The one non-obvious edge in this diagram**: Step 3 branches off Step 1's
*raw* output, not Step 2's filtered output - band power in the
`high_gamma` band needs the unfiltered wide-band signal, which Step 2's
default 1-40Hz bandpass would zero out. Step 3 still *depends* on Step 2
having run first, though, since it borrows Step 2's `bad_channels` list
to stay on the same reduced channel set. Both files' docstrings call this
out for exactly this reason - it's the kind of thing that looks like a bug
to "fix" later if you don't already know it's intentional.

`test.h5` is untouched by both Step 2 and Step 3 in their normal run -
preprocessing params are fit on train only, and test only gets processed
once, at final held-out evaluation (a step we haven't built yet - it'll
reuse Step 2's saved `normalization_params.npz`/`artifact_params.json`
rather than re-fitting).

## Step 1: Data Preparation

**Run:** `python scripts/run_data_preparation.py --subsample-fraction 0.20`

**Orchestrator:** `src/steps/data_preparation.py`
- `run_data_acquisition(cfg)` - streams train pool + test set from Hugging
  Face (`src/data/acquisition.py`'s `acquire_split()`), pairing each digit
  trial with its preceding blank so binary classes come out balanced by
  construction.
- `run_train_val_split(cfg)` - checks for session overlap between the
  train pool and test set, assigns whole `(session, block)` blocks to
  train or val (never splitting a block across both - `src/data/
  train_val_split.py`'s `assign_block_splits()`), and writes the result.
- `run_data_preparation(cfg)` - calls both of the above in order. This is
  the one function a future `pipeline.py` needs to call for this step.

**No registry** - there's one way to acquire this dataset, so nothing to
select between yet.

**Idempotent by default** - re-running with `force=False` (the default)
skips any output file that already exists on disk.

## Step 2: Preprocessing

**Run:** `python scripts/run_preprocessing.py --subsample-fraction 0.20`

**Orchestrator:** `src/steps/preprocessing.py`'s `run_preprocessing(cfg)`:
1. **Diagnostic pass**: takes a subsample of train (`diagnostic_subsample_size`,
   default 25,000 trials), filters it, fits normalization on it to detect
   dead/bad channels, excludes those channels, re-fits normalization on
   the reduced channel set, derives artifact-flagging thresholds from
   *that* data's own percentiles. Saves the fitted result (`normalization_params.npz`,
   `artifact_params.json`) to `preprocessing_params_dir` - this is what
   makes it safe to apply the SAME fit to val (and later, test) without
   ever re-fitting on them.
2. **Full chunked pass**: streams train, then val, through filter ->
   normalize -> artifact-flag in fixed-size chunks (`chunk_size`, default
   2000), writing each chunk to disk immediately. Chunking exists because
   loading a full split into memory at once previously crashed at 100%
   scale (~13GB for one array).

**Registries used:**
- `FILTER_REGISTRY` (`src/preprocessing/filters.py`) - currently one
  variant, `butterworth_bandpass`.
- `NORMALIZATION_REGISTRY` (`src/preprocessing/normalization.py`) -
  currently one variant, `robust_median_mad` (median/MAD, resistant to
  the outlier trials artifact detection is specifically looking for).
- `ARTIFACT_REGISTRY` (`src/preprocessing/artifacts.py`) - currently one
  variant, `percentile_multi_criteria`, bundling four functions (detect
  bad channels, compute diagnostic arrays, derive thresholds, flag) since
  a different strategy's internals wouldn't decompose the same way.

**Not part of the automatic run:** `src/preprocessing/frequency_audit.py` -
a standalone tool for re-validating whether a bandpass cutoff is still
appropriate. This is what Stage 3's planned "frequency audit re-run at
100%" diagnostic refers to - run it by hand, not via this step.

## Step 3: Features

**Run:** `python scripts/run_features.py --subsample-fraction 0.20`

**Orchestrator:** `src/steps/features.py`'s `run_features(cfg)`:
- Loads `bad_channels` from Step 2's saved `artifact_params.json`.
- Streams train, then val, from Step 1's **raw** split files (not Step
  2's filtered output - see the data-flow note above), excludes the same
  bad channels, extracts features per trial in chunks, writes to
  `features_dir`.

**Registry:** `FEATURE_REGISTRY` (`src/features/extraction.py`) -
currently one variant, `band_power_stat_freq` (band power across 6 EEG
bands, plus per-channel statistical and spectral summary stats).

**Worth remembering:** nothing you've trained so far actually reads this
step's output - only classical models do (`src/steps/models.py`'s
`_load_classical_features`), and every real run so far has been EEGNet.
The step is fully built and tested, just unused until a classical
baseline or permutation-floor comparison is actually run.

## Step 4: Models

**Run:** `python scripts/run_models.py --model-name eegnet_fresh --task binary`

**Orchestrator:** `src/steps/models.py`'s `run_model_training(cfg)` -
branches on `cfg.model.is_deep` (derived from the registry entry, not a
separate hardcoded flag):
- `run_classical_model(cfg)` - loads Step 3's features, excludes
  `trial_concern`-flagged trials (from Step 2), scales, fits, evaluates,
  checkpoints, permutation-tests.
- `run_deep_model(cfg)` - loads Step 2's filtered output *lazily* (reads
  labels/flags into memory, but individual trials are read from disk
  on-demand during training via `LazyEEGDataset` - required at 100% scale),
  trains (single-phase, or two-phase for `eegnet_finetuned_backbone_reuse`),
  evaluates, permutation-tests.

**Registry:** `MODEL_REGISTRY` (`src/models/factory.py`) - one flat list,
`model_name` alone identifies exactly what's built:
- `lda`, `svm`, `random_forest` - classical.
- `eegnet_fresh` - no reuse; the only option for `task="binary"`, and
  multiclass's own no-reuse baseline.
- `eegnet_frozen_backbone_reuse` - loads a trained `eegnet_fresh`
  checkpoint from another task (`cfg.model.eegnet.reuse_source_task`,
  default `"binary"`), swaps in a fresh head, freezes everything else.
- `eegnet_finetuned_backbone_reuse` - same start as frozen, then
  unfreezes and fine-tunes at a lower LR (`src/training/loop.py`'s
  `train_with_two_phase_finetuning` - the two-phase logic lives in
  training, not in how the model is built).
- `eegnet_dual_backbone_auxiliary_input` - loads that same reused
  backbone (frozen) AND builds a second, fresh backbone trained from
  scratch, concatenates both feature sets before the classifier.

Each entry optionally declares `build_for_permutation` - a cheaper
substitute used only inside permutation testing (SVM -> capped
`LinearSVC`; every reuse-based EEGNet variant -> plain `eegnet_fresh`,
since a rough noise-floor check doesn't need to replicate the full reuse
procedure).

**Outputs:** `models/checkpoints/<run_tag>/` (the trained model) and
`models/results/<run_tag>/results.json` (metrics + permutation test),
where `run_tag` = `<dataset_variant_tag>__<task>__<model_name>` - unique
per exact combination, so nothing overwrites anything else.

## What `pipeline.py` will actually need to do

Given the above, the integration is mechanical rather than a redesign:
read one `PipelineConfig`, call `run_data_preparation(cfg)` ->
`run_preprocessing(cfg)` -> `run_features(cfg)` (only if a classical model
is selected, or always for consistency - worth deciding) ->
`run_model_training(cfg)`, in order, and expose the same CLI arguments
the four scripts already expose individually. The only genuinely new
logic will be the interactive-prompt piece for variant selection.

---

## Why Step 3 (features) looks unused

It isn't unused - it's unexercised so far. There's exactly one consumer:
`src/steps/models.py`'s `_load_classical_features`, called only when
`run_classical_model` runs - which only happens when `cfg.model.model_name`
is `lda`, `svm`, or `random_forest`. Every real run so far has trained an
EEGNet variant, and EEGNet reads Step 2's filtered signal directly
(`_load_deep_lazy_indices`) - it learns its own features via convolutions,
so it has no use for hand-engineered band power/statistical features. The
moment a classical model is trained, Step 3's output gets read for real.

---

## File-by-file function reference

Every function/class in every file built so far, in the order you'd
naturally read them (low-level building blocks first, orchestrators last).
`__init__`/`__len__`/`__getitem__`/dunder methods on Dataset classes are
grouped under their class rather than listed separately.

### `src/utils/registry.py`
| Function | What it does |
|---|---|
| `class Registry` | Name -> function/object lookup table. `register(name)` (decorator, rejects duplicate names), `get(name)` (raises with the full valid-names list if missing), `names()` (sorted list of registered names). |

### `src/data/acquisition.py` (Step 1 building block)
| Function | What it does |
|---|---|
| `detect_channel_names(row)` | Infers channel names from one streamed row's column names (runs once, cached by the caller). |
| `extract_eeg_trial(row, channel_names, n_samples)` | Reshapes one row's flat per-sample columns into a `(n_channels, n_samples)` array. |
| `_append_row(dset, value)` | Grows a resizable HDF5 dataset by one row. |
| `acquire_split(...)` | The main entry point: streams one HF split, pairs each digit trial with its preceding blank, writes both to an output HDF5 file until every class hits its target count (or the stream runs out). |

### `src/data/train_val_split.py` (Step 1 building block)
| Function | What it does |
|---|---|
| `find_session_overlap(path_a, path_b)` | Returns which `sessionnum`s appear in both files (train pool vs. test). |
| `assign_block_splits(sessionnum, blocknum, val_fraction, seed)` | Randomly assigns each unique `(session, block)` to train/val, maps every trial to its block's assignment. |
| `check_split_has_no_block_overlap(sessionnum, blocknum, trial_splits)` | Hard check that zero block appears in both train and val. |
| `materialize_split(source_path, trial_splits, output_paths)` | Writes each split's rows to its own HDF5 file, reading only those rows from disk (not the full source array). |

### `src/steps/data_preparation.py` (Step 1 orchestrator)
| Function | What it does |
|---|---|
| `_exists(path)` | Trivial existence check, used for the idempotent-skip logic below. |
| `run_data_acquisition(cfg, force)` | Calls `acquire_split()` for train pool + test, skipping if outputs already exist. |
| `run_train_val_split(cfg, force)` | Calls `find_session_overlap` -> `assign_block_splits` -> `check_split_has_no_block_overlap` -> `materialize_split`, in order; excludes any overlapping-session trials first. |
| `run_data_preparation(cfg, force)` | Calls the two functions above, in order. **The function `pipeline.py` will call for this whole step.** |

### `src/preprocessing/filters.py` (Step 2 building block)
| Function | What it does |
|---|---|
| `design_bandpass_sos(low_hz, high_hz, fs, order)` | Builds Butterworth bandpass filter coefficients. |
| `design_notch_sos(freq_hz, fs, quality)` | Builds notch filter coefficients (mains hum removal), off by default. |
| `apply_butterworth_bandpass(eeg, fs, ...)` | **The registered `FILTER_REGISTRY["butterworth_bandpass"]` variant.** Applies zero-phase bandpass (+ optional notch) per trial per channel. |
| `check_filter_stability(low_hz, high_hz, fs, order)` | Design-time diagnostic - are these filter coefficients numerically stable? Not called by the pipeline itself. |

### `src/preprocessing/normalization.py` (Step 2 building block)
| Function | What it does |
|---|---|
| `class NormalizationStrategy` | Bundles one variant's `fit`+`apply` pair together. |
| `_fit_robust_median_mad(eeg_train)` | Computes per-channel median/MAD-based center+scale from train data. |
| `_apply_robust_median_mad(eeg, center, scale)` | Applies a previously-fit center/scale to any split. |
| *(the two above registered together as `NORMALIZATION_REGISTRY["robust_median_mad"]`)* | |
| `save_normalization_params(path, center, scale)` / `load_normalization_params(path)` | Persist/reload a fitted result, so it's never accidentally re-fit on val/test. |

### `src/preprocessing/artifacts.py` (Step 2 building block)
| Function | What it does |
|---|---|
| `class ArtifactStrategy` | Bundles one variant's 4 functions together (see below). |
| `_detect_bad_channels_percentile(scale, floor)` | Flags channels with near-zero normalization scale (dead/faulty electrode). |
| `exclude_channels(eeg, bad_channels, n_channels_nominal)` | Drops bad channels from the channel axis - shared utility, not part of the strategy bundle (every variant needs this exact operation). |
| `_compute_arrays_percentile(eeg_normalized)` | Computes raw per-trial/per-channel amplitude, jump, std, kurtosis arrays. |
| `_derive_thresholds_percentile(train_arrays, artifact_percentile, flatline_percentile)` | Turns train's diagnostic arrays into flagging thresholds (data's own percentiles, not fixed assumptions). |
| `_flag_artifacts_percentile(arrays, thresholds, trial_concern_min_channels)` | Applies thresholds to any split's diagnostic arrays, producing per-trial flags. |
| *(the four `_..._percentile` functions above registered together as `ARTIFACT_REGISTRY["percentile_multi_criteria"]`)* | |
| `save_artifact_params(path, bad_channels, thresholds)` / `load_artifact_params(path)` | Persist/reload bad channels + thresholds. |

### `src/preprocessing/frequency_audit.py` (standalone tool, not part of the automatic run)
| Function | What it does |
|---|---|
| `compute_class_psd(eeg_a, eeg_b, fs, nperseg)` | Compares power spectral density between two classes. |
| `audit_cutoff(freqs, log_diff, cutoff_hz)` | Checks whether a candidate bandpass cutoff is discarding real class-divergent signal. |

### `src/steps/preprocessing.py` (Step 2 orchestrator)
| Function | What it does |
|---|---|
| `_append_chunk(dset, chunk)` | Grows a resizable HDF5 dataset by a whole chunk at once (vs. acquisition's one-row-at-a-time). |
| `_filter_kwargs(cfg)` | Picks `FilterConfig`'s fields out as kwargs for the registered filter function. |
| `_process_split_chunked(...)` | Streams one split through filter -> normalize -> artifact-flag in chunks, writing each chunk immediately. |
| `run_preprocessing(cfg, force, diagnostic_subsample_size, chunk_size)` | **The step's entry point.** Diagnostic-fit pass (bad channels, normalization, thresholds) on a train subsample, then calls `_process_split_chunked` for train, then val. |

### `src/features/extraction.py` (Step 3 building block)
| Function | What it does |
|---|---|
| `compute_band_power(sig, fs, bands, nperseg)` | Power in each of 6 EEG frequency bands, for one channel's signal. |
| `compute_statistical_features(sig)` | Mean/std/skew/kurtosis etc. for one channel's signal. |
| `compute_frequency_features(sig, fs, nperseg)` | Spectral summary stats (e.g. peak frequency) for one channel's signal. |
| `extract_features_for_trial(eeg_trial_raw, fs, bands, welch_nperseg)` | Runs the three functions above across every channel of one trial, concatenates into one feature vector. |
| `extract_features_batch(eeg_batch_raw, fs, bands, welch_nperseg)` | **The registered `FEATURE_REGISTRY["band_power_stat_freq"]` variant.** Calls `extract_features_for_trial` across a whole batch. |
| `feature_names(bands)` | Returns the ordered list of feature names matching a feature vector's columns - for interpreting results, not used in extraction itself. |

### `src/steps/features.py` (Step 3 orchestrator)
| Function | What it does |
|---|---|
| `_append_chunk(dset, chunk)` | Same idea as preprocessing's version. |
| `_feature_kwargs(cfg)` | Picks `FeatureConfig`'s fields out as kwargs for the registered feature function. |
| `_extract_split_chunked(...)` | Streams one split's **raw** trials (Step 1's output, not Step 2's), excludes Step 2's bad channels, extracts features per chunk. |
| `run_features(cfg, force, chunk_size)` | **The step's entry point.** Loads `bad_channels` from Step 2's saved params, calls `_extract_split_chunked` for train, then val. |

### `src/models/eegnet.py` (Step 4 building block)
| Function | What it does |
|---|---|
| `class EEGNet` | The architecture. `forward`/`_forward_features` (the actual network), `num_parameters`, `freeze_backbone`/`unfreeze_backbone` (lock/unlock everything except the classifier head), `replace_classifier` (swap the output layer for a new class count). |
| `class EEGNetDualBackbone` | Combines a frozen backbone + a fresh trainable backbone. `forward` (runs both, concatenates, classifies), `train` (overridden so the frozen half never leaves eval mode, protecting its BatchNorm stats), `num_parameters`. |

### `src/models/factory.py` (Step 4 building block - the model registry)
| Function | What it does |
|---|---|
| `class ModelBuilder` | Bundles `build` + `is_deep` + optional `build_for_permutation` for one registered model. |
| `build_lda` / `build_svm` / `build_random_forest` | Build an unfitted sklearn estimator. |
| `build_svm_for_permutation` | Cheaper `LinearSVC` substitute for SVM's permutation test. |
| `build_eegnet_fresh` | Fresh, untrained EEGNet - works for either task; also every reuse variant's permutation substitute. |
| `_load_reuse_source_backbone(cfg, n_channels, n_samples)` | Shared by every reuse variant: loads a trained `eegnet_fresh` checkpoint from `cfg.model.eegnet.reuse_source_task`. |
| `build_eegnet_frozen_backbone_reuse` | Loads the reuse-source backbone, swaps in a new head, freezes the backbone. |
| `build_eegnet_finetuned_backbone_reuse` | Same starting state as frozen - the unfreeze+fine-tune happens later, in training. |
| `build_eegnet_dual_backbone_auxiliary_input` | Loads the reuse-source backbone (frozen) AND builds a fresh second backbone; combined in `EEGNetDualBackbone`. |
| *(all seven `build_*` functions above registered under `MODEL_REGISTRY`)* | |

### `src/training/loop.py` (Step 4 building block)
| Function | What it does |
|---|---|
| `class EEGDataset` | Wraps a full in-memory EEG array + labels as a PyTorch `Dataset`. |
| `class LazyEEGDataset` | Same idea, but reads each trial from an HDF5 file on demand (never loads the full array) - needed at 100% scale. |
| `make_loaders(...)` / `make_lazy_loaders(...)` | Wrap the two Dataset classes above into `DataLoader`s. |
| `run_epoch(model, loader, optimizer, criterion, device, train)` | One pass over a loader - training or eval, depending on `train`. |
| `train_with_checkpointing(...)` | Runs `run_epoch` in a loop with early stopping, saving the best-val-accuracy checkpoint. Used by `eegnet_fresh`, `eegnet_frozen_backbone_reuse`, and `eegnet_dual_backbone_auxiliary_input`. |
| `train_with_two_phase_finetuning(...)` | Calls `train_with_checkpointing` twice (frozen phase, then fine-tune phase), tracking the best checkpoint across both. Used only by `eegnet_finetuned_backbone_reuse`. |

### `src/evaluation/metrics.py` (Step 4 building block)
| Function | What it does |
|---|---|
| `compute_classification_metrics(y_true, y_pred, model_name, average)` | Accuracy/precision/recall/F1, in the same shape regardless of model. |
| `evaluate_sklearn_model(name, model, X_train, y_train, X_val, y_val, average)` | Fits an sklearn model, evaluates it on val, returns metrics + confusion matrix. |

### `src/evaluation/permutation_test.py` (Step 4 building block)
| Function | What it does |
|---|---|
| `permutation_test_sklearn(model_fn, X_train, y_train, X_val, y_val, ...)` | Fits the real model, then N models on shuffled labels, compares val accuracy. |
| `permutation_test_torch(train_fn, eval_fn, eeg_train, y_train, val_loader, ...)` | Same idea, for a PyTorch model (caller supplies how to train/evaluate one). |

### `src/steps/models.py` (Step 4 orchestrator)
| Function | What it does |
|---|---|
| `_load_classical_features(cfg)` | Loads Step 3's features, excludes `trial_concern` trials, filters to digit-only if multiclass. |
| `_load_deep_lazy_indices(cfg)` | Same idea but returns indices into Step 2's filtered HDF5 file, not full arrays (for lazy loading). |
| `_save_results(cfg, results)` | Writes `results.json` to `cfg.model.results_dir(...)`. |
| `run_classical_model(cfg, run_permutation)` | Full classical path: load features -> scale -> build+fit -> evaluate -> checkpoint -> permutation test. |
| `run_deep_model(cfg, run_permutation)` | Full deep path: load lazy indices -> build model -> train -> evaluate -> permutation test. |
| `run_model_training(cfg, run_permutation)` | **The step's entry point.** Branches to one of the two functions above based on `cfg.model.is_deep`. |

---

## Dry run: the full call trace, function by function

This is what actually executes, in order, for a complete run. Read this
alongside the file reference above rather than the source itself - it's
the same information, arranged as "what calls what" instead of "what's in
each file."

### Step 1 - `python scripts/run_data_preparation.py`

```
main()
 └─ build_config(...)                                  [src/config.py]
 └─ run_data_preparation(cfg)                           [steps/data_preparation.py]
     ├─ run_data_acquisition(cfg)
     │   ├─ acquire_split(... hf_split="train" ...)     [data/acquisition.py]
     │   │   ├─ detect_channel_names(row)                (once)
     │   │   ├─ extract_eeg_trial(row, ...)               (per trial)
     │   │   └─ _append_row(dset, value)                   (per trial written)
     │   └─ acquire_split(... hf_split="test" ...)       (same, for test.h5)
     └─ run_train_val_split(cfg)
         ├─ find_session_overlap(train_pool, test)
         ├─ assign_block_splits(sessionnum, blocknum, ...)
         ├─ check_split_has_no_block_overlap(...)
         └─ materialize_split(train_pool, splits, {train, val})
 └─ _summarize(cfg)                                     [run_data_preparation.py itself, printing only]
```

### Step 2 - `python scripts/run_preprocessing.py`

```
main()
 └─ build_config(..., filter_variant=..., artifact_variant=..., normalization_variant=...)
 └─ run_preprocessing(cfg, diagnostic_subsample_size, chunk_size)  [steps/preprocessing.py]
     ├─ FILTER_REGISTRY.get(cfg.filter.variant)          -> apply_butterworth_bandpass
     ├─ NORMALIZATION_REGISTRY.get(...)                  -> NormalizationStrategy(fit=_fit_robust_median_mad, apply=_apply_robust_median_mad)
     ├─ ARTIFACT_REGISTRY.get(...)                       -> ArtifactStrategy(detect_bad_channels=..., compute_arrays=..., derive_thresholds=..., flag=...)
     │
     ├─ [diagnostic pass, on a train subsample]
     │   ├─ apply_butterworth_bandpass(eeg_diag, fs, ...)
     │   ├─ norm_strategy.fit(...)              -> _fit_robust_median_mad
     │   ├─ artifact_strategy.detect_bad_channels(scale, ...)   -> _detect_bad_channels_percentile
     │   ├─ exclude_channels(eeg_diag, bad_channels, ...)
     │   ├─ norm_strategy.fit(...) again, on good channels only
     │   ├─ save_normalization_params(...)
     │   ├─ norm_strategy.apply(...)             -> _apply_robust_median_mad
     │   ├─ artifact_strategy.compute_arrays(...) -> _compute_arrays_percentile
     │   ├─ artifact_strategy.derive_thresholds(...) -> _derive_thresholds_percentile
     │   └─ save_artifact_params(...)
     │
     └─ [full pass, train then val]
         └─ _process_split_chunked(...)  (once per split)
             per chunk:
             ├─ exclude_channels(chunk, bad_channels, ...)
             ├─ apply_butterworth_bandpass(chunk, fs, **_filter_kwargs(cfg))
             ├─ norm_strategy.apply(...)
             ├─ artifact_strategy.compute_arrays(...)
             ├─ artifact_strategy.flag(...)        -> _flag_artifacts_percentile
             └─ _append_chunk(...)  x6 (eeg, label_binary, label_digit, artifact_any, n_flagged, trial_concern)
```

### Step 3 - `python scripts/run_features.py`

```
main()
 └─ build_config(..., feature_variant=...)
 └─ run_features(cfg, chunk_size)                       [steps/features.py]
     ├─ load_artifact_params(...)                       [preprocessing/artifacts.py - reads Step 2's saved bad_channels]
     ├─ FEATURE_REGISTRY.get(cfg.feature.variant)        -> extract_features_batch
     └─ _extract_split_chunked(...)  (train, then val - reads Step 1's RAW split, not Step 2's output)
         per chunk:
         ├─ exclude_channels(chunk, bad_channels, ...)
         ├─ extract_features_batch(chunk, fs, bands, nperseg)
         │   └─ extract_features_for_trial(...)  (per trial in the chunk)
         │       ├─ compute_band_power(...)       (per channel)
         │       ├─ compute_statistical_features(...) (per channel)
         │       └─ compute_frequency_features(...)   (per channel)
         └─ _append_chunk(...)
```

### Step 4 - `python scripts/run_models.py --model-name eegnet_frozen_backbone_reuse --task multiclass`

```
main()
 └─ build_config(..., model_name="eegnet_frozen_backbone_reuse", task="multiclass")
 └─ run_model_training(cfg)                              [steps/models.py]
     └─ cfg.model.is_deep == True  -> run_deep_model(cfg)
         ├─ _load_deep_lazy_indices(cfg)                 (reads labels/flags from Step 2's filtered files)
         ├─ make_lazy_loaders(...)                       [training/loop.py]
         ├─ MODEL_REGISTRY.get("eegnet_frozen_backbone_reuse")  -> ModelBuilder(build=build_eegnet_frozen_backbone_reuse, ...)
         ├─ builder.build(cfg, n_channels, n_samples, n_classes)
         │   └─ build_eegnet_frozen_backbone_reuse(...)  [models/factory.py]
         │       ├─ _load_reuse_source_backbone(cfg, ...)   (loads eegnet_fresh/binary's checkpoint)
         │       ├─ model.replace_classifier(n_classes)     [models/eegnet.py]
         │       └─ model.freeze_backbone()                 [models/eegnet.py]
         ├─ train_with_checkpointing(model, train_loader, val_loader, ...)   [training/loop.py]
         │   └─ run_epoch(...)  (repeated, train then eval, each epoch)
         ├─ [evaluation on val_loader]
         │   └─ accuracy_score / precision_score / recall_score / f1_score
         ├─ [permutation test, if enabled]
         │   ├─ builder.build_for_permutation  -> build_eegnet_fresh   (NOT the reuse variant - see factory.py docstring)
         │   └─ permutation_test_torch(train_fn, eval_fn, ...)          [evaluation/permutation_test.py]
         │       └─ train_fn/eval_fn call run_epoch(...) internally, N+1 times
         └─ _save_results(cfg, results)
```

For a classical model (`run_classical_model`), the shape is the same idea,
simpler: `_load_classical_features` -> `MODEL_REGISTRY.get(...)` ->
`builder.build(...)` -> `evaluate_sklearn_model(...)` (which internally
calls `.fit()`/`.predict()` and `compute_classification_metrics(...)`) ->
`joblib.dump(...)` -> `permutation_test_sklearn(...)` -> `_save_results(...)`.

---

# File-by-file function reference

Every function in the project, grouped by file, in the order they appear.
`_leading_underscore` names are private helpers (not meant to be called
from outside their own file).

## `src/config.py`
Mostly dataclasses (data, not functions) - the functions/methods worth knowing:
- `DataConfig.variant_info` / `.hf_dataset_name` / `.target_per_class` / `.test_target_per_class` / `.n_samples` / `.variant_tag` / `.raw_dir` / `.interim_dir` / `.splits_dir` / `.filtered_dir` / `.preprocessing_params_dir` / `.features_dir` - all `@property`, all derived automatically from `dataset_variant` + `subsample_fraction`.
- `ModelConfig.__post_init__` - validates `model_name`/`task` are known values.
- `ModelConfig.is_deep` / `.n_classes` / `.run_tag()` / `.checkpoint_path()` / `.results_dir()` - derived model-run properties.
- `build_config(...)` - the one function you actually call; builds a fully-wired `PipelineConfig`.

## `src/utils/registry.py`
- `Registry.register(name)` - decorator; adds a function/object to the registry under `name`.
- `Registry.get(name)` - looks up what's registered under `name`, raises with a list of valid names if not found.
- `Registry.names()` - all registered names, sorted.

## `src/data/acquisition.py` (Step 1)
- `extract_eeg_trial(row, channel_names, n_samples)` - reshapes one streamed HF row into a `(channels, samples)` array.
- `detect_channel_names(row)` - infers the channel name list from one row's columns.
- `_append_row(dset, value)` - grows an HDF5 dataset by one row.
- `acquire_split(...)` - the real work: streams one HF split, pairs each digit trial with its preceding blank, writes both to HDF5.

## `src/data/train_val_split.py` (Step 1)
- `find_session_overlap(path_a, path_b)` - which sessions exist in both files (should be empty).
- `assign_block_splits(sessionnum, blocknum, val_fraction, seed)` - randomly assigns whole blocks to train/val.
- `check_split_has_no_block_overlap(...)` - hard verification that no block ended up in both splits.
- `materialize_split(source_path, trial_splits, output_paths)` - writes each split to its own HDF5 file.

## `src/preprocessing/filters.py` (Step 2)
- `design_bandpass_sos(low_hz, high_hz, fs, order)` - builds the bandpass filter coefficients.
- `design_notch_sos(freq_hz, fs, quality)` - builds notch filter coefficients (mains hum).
- `apply_butterworth_bandpass(...)` - **registered as `"butterworth_bandpass"`**; the actual filtering.
- `check_filter_stability(...)` - design-time diagnostic, not called by the pipeline.

## `src/preprocessing/normalization.py` (Step 2)
- `_fit_robust_median_mad(eeg_train)` - computes median/MAD center+scale from train.
- `_apply_robust_median_mad(eeg, center, scale)` - applies a previously-fit center/scale.
- Both bundled and **registered as `"robust_median_mad"`**.
- `save_normalization_params(path, center, scale)` / `load_normalization_params(path)` - persist/reload the fit.

## `src/preprocessing/artifacts.py` (Step 2, also used by Step 3)
- `_detect_bad_channels_percentile(scale, bad_channel_scale_floor)` - flags dead/faulty channels.
- `exclude_channels(eeg, bad_channels, n_channels_nominal)` - drops those channels (shared utility, used by Steps 2 *and* 3).
- `_compute_arrays_percentile(eeg_normalized)` - raw per-trial diagnostic arrays (amplitude, jump, std, kurtosis).
- `_derive_thresholds_percentile(train_arrays, artifact_percentile, flatline_percentile)` - turns train's arrays into thresholds.
- `_flag_artifacts_percentile(arrays, thresholds, trial_concern_min_channels)` - applies thresholds to flag trials.
- The four above bundled and **registered as `"percentile_multi_criteria"`**.
- `save_artifact_params(path, bad_channels, thresholds)` / `load_artifact_params(path)` - persist/reload (Step 3 reloads this).

## `src/preprocessing/frequency_audit.py`
Standalone diagnostic, not called by any step - `compute_class_psd(...)`, `audit_cutoff(...)`. Run by hand when re-checking a cutoff choice.

## `src/features/extraction.py` (Step 3)
- `compute_band_power(sig, fs, bands, nperseg)` - power per named frequency band.
- `compute_statistical_features(sig)` - mean/variance/skew/kurtosis/zero-crossing-rate.
- `compute_frequency_features(sig, fs, nperseg)` - dominant frequency + total power.
- `extract_features_for_trial(eeg_trial_raw, fs, bands, welch_nperseg)` - runs all three above per channel, concatenates.
- `extract_features_batch(...)` - **registered as `"band_power_stat_freq"`**; applies the above across a batch of trials.
- `feature_names(bands)` - ordered name list matching the feature vector's columns.

## `src/models/eegnet.py`
- `EEGNet.__init__/._forward_features/.forward` - the architecture itself.
- `EEGNet.num_parameters()` - trainable+frozen parameter count.
- `EEGNet.freeze_backbone()` / `.unfreeze_backbone()` - lock/unlock everything except the classifier.
- `EEGNet.replace_classifier(n_classes)` - swap the final layer for a new class count.
- `EEGNetDualBackbone.__init__/.forward` - combines a frozen + a trainable `EEGNet`.
- `EEGNetDualBackbone.train(mode)` - overridden so the frozen half never leaves eval mode.

## `src/models/factory.py` (Step 4)
- `build_lda`, `build_svm`, `build_random_forest` - **registered as `"lda"`/`"svm"`/`"random_forest"`**.
- `build_svm_for_permutation` - SVM's cheap substitute for the permutation harness.
- `build_eegnet_fresh` - **registered as `"eegnet_fresh"`**; also used as every reuse variant's permutation substitute.
- `_load_reuse_source_backbone(cfg, n_channels, n_samples)` - loads another task's trained `eegnet_fresh` checkpoint (shared by the three functions below).
- `build_eegnet_frozen_backbone_reuse` - **registered as `"eegnet_frozen_backbone_reuse"`**.
- `build_eegnet_finetuned_backbone_reuse` - **registered as `"eegnet_finetuned_backbone_reuse"`**.
- `build_eegnet_dual_backbone_auxiliary_input` - **registered as `"eegnet_dual_backbone_auxiliary_input"`**.

## `src/training/loop.py` (Step 4, deep models only)
- `EEGDataset` - wraps an in-memory EEG array as a PyTorch `Dataset`.
- `LazyEEGDataset` - same, but reads each trial from HDF5 on demand (used at scale).
- `make_loaders` / `make_lazy_loaders` - build train+val `DataLoader`s from either of the above.
- `run_epoch(model, loader, optimizer, criterion, device, train)` - one train or eval pass.
- `train_with_checkpointing(...)` - the standard training loop (checkpointing + early stopping).
- `train_with_two_phase_finetuning(...)` - freeze-phase then fine-tune-phase, for `eegnet_finetuned_backbone_reuse`.

## `src/evaluation/metrics.py`
- `compute_classification_metrics(y_true, y_pred, model_name, average)` - accuracy/precision/recall/F1.
- `evaluate_sklearn_model(name, model, X_train, y_train, X_val, y_val, average)` - fits + evaluates a classical model.

## `src/evaluation/permutation_test.py`
- `permutation_test_sklearn(model_fn, X_train, y_train, X_val, y_val, ...)` - shuffled-label test for classical models.
- `permutation_test_torch(train_fn, eval_fn, eeg_train, y_train, val_loader, ...)` - same idea for deep models.

## `src/steps/data_preparation.py` (Step 1 orchestrator)
- `_exists(path)` - trivial existence check.
- `run_data_acquisition(cfg, force)` - calls `acquire_split()` for train pool + test.
- `run_train_val_split(cfg, force)` - calls the `train_val_split.py` functions in order.
- `run_data_preparation(cfg, force)` - calls both of the above; **this is what a future `pipeline.py` calls**.

## `src/steps/preprocessing.py` (Step 2 orchestrator)
- `_append_chunk(dset, chunk)` - grows an HDF5 dataset by a whole chunk at once.
- `_filter_kwargs(cfg)` - picks `FilterConfig`'s fields to pass into the registered filter function.
- `_process_split_chunked(...)` - streams one split through filter → normalize → flag, in chunks.
- `run_preprocessing(cfg, force, diagnostic_subsample_size, chunk_size)` - diagnostic fit, then calls `_process_split_chunked` for train and val. **This is what a future `pipeline.py` calls.**

## `src/steps/features.py` (Step 3 orchestrator)
- `_append_chunk(dset, chunk)` - same idea as preprocessing's.
- `_feature_kwargs(cfg)` - picks `FeatureConfig`'s fields for the registered feature function.
- `_extract_split_chunked(...)` - streams one split through channel-exclusion + feature extraction.
- `run_features(cfg, force, chunk_size)` - loads Step 2's `bad_channels`, calls `_extract_split_chunked` for train and val. **This is what a future `pipeline.py` calls.**

## `src/steps/models.py` (Step 4 orchestrator)
- `_load_classical_features(cfg)` - reads Step 3's features + Step 2's `trial_concern` flags, applies exclusions.
- `_load_deep_lazy_indices(cfg)` - same idea, but returns indices into Step 2's filtered HDF5 rather than loaded arrays.
- `_save_results(cfg, results)` - writes `results.json`.
- `run_classical_model(cfg, run_permutation)` - the sklearn path, full call chain below.
- `run_deep_model(cfg, run_permutation)` - the PyTorch path, full call chain below.
- `run_model_training(cfg, run_permutation)` - branches on `cfg.model.is_deep`. **This is what a future `pipeline.py` calls.**

## `scripts/*.py`
Each has `main()` (parses CLI args, builds config, calls its step's orchestrator) and, for Steps 1-3, a `_summarize()` that prints shapes/counts afterward - convenience only, not part of the pipeline itself.

---

# How the functions connect, per step

Arrows show what calls what, top to bottom.

**Step 1** - `run_data_preparation.py:main()`
→ `build_config()`
→ `run_data_preparation(cfg)`
&nbsp;&nbsp;→ `run_data_acquisition(cfg)` → `acquire_split()` ×2 (train pool, test) → `detect_channel_names()`, `extract_eeg_trial()`, `_append_row()`
&nbsp;&nbsp;→ `run_train_val_split(cfg)` → `find_session_overlap()` → `assign_block_splits()` → `check_split_has_no_block_overlap()` → `materialize_split()`
→ `_summarize(cfg)`

**Step 2** - `run_preprocessing.py:main()`
→ `run_preprocessing(cfg)`
&nbsp;&nbsp;→ (diagnostic pass) `FILTER_REGISTRY.get(...)` → `apply_butterworth_bandpass()`
&nbsp;&nbsp;→ `NORMALIZATION_REGISTRY.get(...).fit` → `_fit_robust_median_mad()`
&nbsp;&nbsp;→ `ARTIFACT_REGISTRY.get(...).detect_bad_channels` → `_detect_bad_channels_percentile()`
&nbsp;&nbsp;→ `exclude_channels()`, re-fit normalization, `.compute_arrays` → `_compute_arrays_percentile()`, `.derive_thresholds` → `_derive_thresholds_percentile()`
&nbsp;&nbsp;→ `save_normalization_params()`, `save_artifact_params()`
&nbsp;&nbsp;→ (full pass, ×2 for train/val) `_process_split_chunked()` → per chunk: `exclude_channels()` → filter → `.apply` normalization → `.compute_arrays` → `.flag` → `_flag_artifacts_percentile()` → `_append_chunk()`

**Step 3** - `run_features.py:main()`
→ `run_features(cfg)`
&nbsp;&nbsp;→ `load_artifact_params()` (Step 2's saved file)
&nbsp;&nbsp;→ (×2 for train/val) `_extract_split_chunked()` → per chunk: `exclude_channels()` → `FEATURE_REGISTRY.get(...)` → `extract_features_batch()` → `extract_features_for_trial()` (per trial) → `compute_band_power()`, `compute_statistical_features()`, `compute_frequency_features()` → `_append_chunk()`

**Step 4** - `run_models.py:main()`
→ `run_model_training(cfg)` — branches on `cfg.model.is_deep`:

*Classical branch* → `run_classical_model(cfg)`
&nbsp;&nbsp;→ `_load_classical_features(cfg)` (reads Step 3 + Step 2's flags)
&nbsp;&nbsp;→ `MODEL_REGISTRY.get(...).build` → `build_lda` / `build_svm` / `build_random_forest`
&nbsp;&nbsp;→ `evaluate_sklearn_model()` → `compute_classification_metrics()`
&nbsp;&nbsp;→ `permutation_test_sklearn()` (using `.build_for_permutation` if set)
&nbsp;&nbsp;→ `_save_results()`

*Deep branch* → `run_deep_model(cfg)`
&nbsp;&nbsp;→ `_load_deep_lazy_indices(cfg)` (reads Step 2's filtered output, lazily)
&nbsp;&nbsp;→ `MODEL_REGISTRY.get(...).build` → `build_eegnet_fresh` / `build_eegnet_frozen_backbone_reuse` / `build_eegnet_finetuned_backbone_reuse` / `build_eegnet_dual_backbone_auxiliary_input` (the last three call `_load_reuse_source_backbone()` internally)
&nbsp;&nbsp;→ `make_lazy_loaders()`
&nbsp;&nbsp;→ `train_with_checkpointing()` (or `train_with_two_phase_finetuning()` if `model_name == "eegnet_finetuned_backbone_reuse"`) → `run_epoch()` per epoch
&nbsp;&nbsp;→ `permutation_test_torch()` (using `.build_for_permutation`)
&nbsp;&nbsp;→ `_save_results()`

---

# Dry run: a complete pass, function by function

Say you ran all four scripts in order, for `task="multiclass"`, once with
`--model-name random_forest` and once with `--model-name
eegnet_frozen_backbone_reuse` (after first training `eegnet_fresh` on
`task="binary"`). Here's what actually executes, in order:

1. **Step 1 runs once** (shared by everything after it): `acquire_split()`
   streams train+test from Hugging Face into `train_pool.h5`/`test.h5`.
   `assign_block_splits()` + `materialize_split()` turn the train pool
   into `train.h5`/`val.h5`. Nothing here knows or cares what task or
   model comes later.

2. **Step 2 runs once**: fits filter/normalization/artifact thresholds on
   a train subsample, writes `train_filtered.h5`/`val_filtered.h5` plus
   the saved `bad_channels`/thresholds. Still task/model-agnostic.

3. **Step 3 runs once**: loads Step 2's `bad_channels`, extracts
   band-power/statistical/frequency features from Step 1's *raw* split,
   writes `train_features.h5`/`val_features.h5`.

4. **First model run - `eegnet_fresh`, `task="binary"`**:
   `_load_deep_lazy_indices()` reads `label_binary` + `trial_concern` from
   Step 2's filtered output (Step 3's features are never touched - deep
   models don't use them). `build_eegnet_fresh()` builds a 2-class EEGNet.
   `train_with_checkpointing()` trains it, saving the best checkpoint to
   `models/checkpoints/<tag>__binary__eegnet_fresh/eegnet_fresh.pt`.
   `permutation_test_torch()` runs a reduced-scale shuffled-label check.
   Results saved.

5. **Second model run - `eegnet_frozen_backbone_reuse`,
   `task="multiclass"`**: same `_load_deep_lazy_indices()`, but now
   reading `label_digit` (multiclass) instead of `label_binary`.
   `MODEL_REGISTRY.get("eegnet_frozen_backbone_reuse").build` calls
   `_load_reuse_source_backbone()`, which builds `ModelConfig(model_name=
   "eegnet_fresh", task="binary")` purely to compute the checkpoint path
   from step 4, finds the `.pt` file saved in step 4, loads its weights
   into a fresh `EEGNet`, then `replace_classifier(10)` + `freeze_backbone()`.
   Training proceeds exactly like step 4, but only the new 10-class head
   has `requires_grad=True`, so `train_with_checkpointing()`'s optimizer
   only ever updates that head.

6. **Third model run - `random_forest`, `task="multiclass"`**: totally
   different path. `_load_classical_features()` reads Step 3's
   `train_features.h5`/`val_features.h5` (finally used!), applies Step
   2's `trial_concern` mask, filters to digit-only rows. `StandardScaler`
   fits on train, transforms both. `build_random_forest()` builds a plain
   `RandomForestClassifier`. `evaluate_sklearn_model()` fits it directly
   (no epochs, no loader - one `.fit()` call) and evaluates on val.
   `permutation_test_sklearn()` refits it ~10 more times on
   label-shuffled data. Results saved to a separate `results.json` under
   its own `run_tag`, so it never collides with either EEGNet run above.

Every one of these six steps is independently re-runnable, cached
(`force=False` skips anything already on disk), and none of them know
about each other except through the files on disk each one reads - which
is exactly what will let `pipeline.py` just call them in sequence.
