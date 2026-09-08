## Stage 3 Roadmap — Modularize, Add Alternatives, Sweep

**Status**: In Progress (modularization underway)
**Depends on**: Stage 1 (complete) and Stage 2 (complete — Path A only; B1/B2/C not
yet built). Current results going into Stage 3: Stage 1 binary ≈70-75% accuracy;
Stage 2 multiclass ≈11% at 20% data, ≈14-15% at 100% data.

### Why this stage exists

Stage 1 and 2 established a working end-to-end pipeline and a first honest
result. Stage 2's multiclass accuracy is low enough (barely above the 10%
chance floor) that before spending further effort on any one preprocessing
choice or model architecture, the pipeline needs to make it cheap to try many
of each and compare — hence: modularize first, then explore alternatives
systematically, rather than hand-tuning one fixed pipeline further.

### Part 1 — Modularization

Restructure the codebase so every pluggable step (preprocessing, artifact
detection, feature extraction, model architecture) follows the same pattern
already used for Stage 2's EEGNet variants (`models/factory.py`'s
`STAGE2_MODEL_REGISTRY`): a name → function registry, resolved from a config
field. Adding a new variant to any step means writing one function and
registering it — no other file changes.

Sub-steps (each reviewed before moving to the next):
- [x] Foundation: generic `Registry` helper (`src/utils/registry.py`)
- [x] Data layer renamed/re-documented: `data/acquisition.py`, `data/train_val_split.py`
- [x] Preprocessing made variant-pluggable: filtering, normalization, artifact detection
- [ ] Feature extraction made variant-pluggable
- [ ] Model selection made variant-pluggable (generalizing beyond Stage 2 EEGNet only)
- [ ] Training loop, evaluation: renamed/re-documented (no registry needed — these
      stay generic across whatever model variant is selected)
- [ ] `pipeline.py`: rebuilt as the final integration point, with descriptive
      (non-phase-numbered) function names, dispatching to each step's registry
- [ ] `config.py`: final wiring — one variant field per pluggable step, plus an
      optional interactive prompt (config value is the default; a runtime prompt
      can override it) for selecting each step's variant

### Part 2 — Alternatives to add, once modularization is in place

Not yet enumerated in detail — to be filled in per step as Part 1 reaches each
module. Known candidates from earlier discussion:
- Preprocessing: alternative filter designs, artifact-detection strategies
- Features: alternative feature sets beyond the current band-power/statistical/
  frequency set
- Models: alternative architectures beyond EEGNet, in addition to Stage 2's
  existing A/B1/B2/C backbone-reuse variants

### Part 3 — Diagnostics (deferred to run alongside the alternatives pass)

- Re-log the artifact detection rate correctly (earlier local-vs-Colab
  discrepancy suspected to be a logging error, not a real environment
  difference) and re-tune thresholds once confirmed
- Shuffled-label sanity check on the multiclass task, to check whether the
  model is learning digit-related signal versus session/order structure
- Run both of the above on Colab, once modularization + alternatives are ready,
  so diagnostics are checked against the same pipeline everything else runs on

### Part 4 — Train + evaluate every alternative

- Run every preprocessing × feature × model combination at both 20% and 100%
  of the 2B dataset
- Compare results to identify which combination(s) are worth carrying forward

### Part 5 — Sweeps / fine-tuning on the strongest candidates

- Extend epochs / early-stopping patience
- Capacity sweep at 100% data scale
- Re-run the frequency-content audit (`preprocessing/frequency_audit.py`) at
  100% scale to confirm the current bandpass cutoff still holds at full data

### Definition of Done

- [ ] Every pluggable step (preprocessing, features, model) supports variant
      selection via config, with at least one alternative beyond the current
      default implemented for each
- [ ] Diagnostics (artifact-rate re-log, shuffled-label check) completed and
      resolved
- [ ] Full comparison across variants run at both 20% and 100% data
- [ ] Sweep results recorded for the strongest candidate(s)
- [ ] Findings written up, feeding into Stage 4 scope/priority
