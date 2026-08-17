## Stage 2 Roadmap — Multiclass Digit Classification (0-9)

**Status**: In Progress
**Depends on**: Stage 1 (complete) — reuses its data pipeline, split, and
(for Path B) trained EEGNet backbone.

### Architecture Principle

Stage 2 is built as a **pluggable model registry**, not hardcoded logic —
each architecture variant is a standalone function with a common contract
(`(cfg, seed) → model`), registered under a name. Adding a new variant
later means writing one function + one registry line, no changes to
pipeline orchestration. This generalizes the pattern already used for
Stage 1's classical model selection (`models/factory.py`).

### Variants to Build & Compare

| Variant | Status | Description | Backbone |
|---|---|---|---|
| **Path A** | Not Started | Fresh EEGNet, trained independently on digit-only data | None (from scratch) |
| **Path B1** | Not Started | Stage 1's EEGNet backbone reused, **frozen** — only a new 10-class head trains | Frozen |
| **Path B2** | Not Started | Same as B1, then backbone **unfrozen and fine-tuned** at low LR | Frozen → fine-tuned |
| **Path C**  | *(deferred)* | Raw 128-channel input **+** Stage 1's output/embedding, concatenated | Auxiliary input |

**Deliverable**: head-to-head comparison (accuracy, macro-F1, permutation
test) across A / B1 / B2 — resolves whether backbone reuse helps at all,
and whether freezing vs. fine-tuning matters, rather than assuming either
from reasoning alone.

### Explicitly Deferred (not blocking Stage 2 completion)

- Classical Stage 2 baselines (LDA/SVM/RF) — already implemented
  (`stage="stage2"`), skipped given Stage 1's classical results were weak;
  cheap to run later if a permutation-tested floor is needed
- Path C (auxiliary-input stacking)
- Alternate architectures (One-vs-Rest, multi-stream fusion, self-supervised
  pretraining, transformers) — see main roadmap's Fine-Tune Phase

### Evaluation

Same standard as Stage 1: permutation test (shuffled-label floor) for
every variant, val + held-out test evaluation, no variant trusted on raw
accuracy alone.

### Definition of Done

- [ ] Path A built, trained, evaluated (val + test + permutation)
- [ ] Path B1 built, trained, evaluated
- [ ] Path B2 built, trained, evaluated
- [ ] A / B1 / B2 comparison written up
- [ ] Model registry pattern in place for Stage 2 (supports future variants
      via new function + registration, no pipeline changes)