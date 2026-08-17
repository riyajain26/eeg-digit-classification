# EEG Digit Classification

An end-to-end machine learning pipeline for classifying imagined handwritten digits from EEG signals using the MindBigData 2023 dataset.

## Features

- EEG preprocessing
- Classical ML baselines
- Deep learning models
- Reproducible training pipeline
- Meta learning models
- Experiment tracking

## Dataset

MindBigData 2023 (MNIST-2B)

## Project Roadmap

### Stage 1 — Binary (blank vs. digit) — ✅ Complete
Classical baselines (LDA/SVM/RandomForest) + EEGNet, permutation-tested,
evaluated on held-out test. EEGNet best (test acc. 0.598 vs. RF 0.527).
Open finding: EEGNet's val-test gap (17pp) far exceeds classical models'
(1-2.5pp) — parked for the fine-tune phase.

### Stage 2 — Multiclass (digit 0-9) — In Progress
Three EEGNet architectures, built and compared (not chosen upfront):
- **Path A** — fresh EEGNet, trained independently on digit-only data, no
  reuse from Stage 1.
- **Path B1** — Stage 1's backbone reused and **frozen**; only a new
  10-class head is trained on top.
- **Path B2** — same as B1, then the backbone is **unfrozen and
  fine-tuned** at a low learning rate.

Classical Stage 2 baselines (LDA/SVM/RF) — already implemented
(`stage="stage2"`), deliberately skipped for now given Stage 1's classical
results were weak; available cheaply later if a permutation-tested floor
is ever needed for comparison.

**Deliverable**: A vs. B1 vs. B2 comparison — resolves whether/how backbone
reuse actually helps, rather than assuming it from reasoning alone.

### Fine-Tune Phase (after Stage 1 + 2 both complete)
- Investigate EEGNet's val-test generalization gap (session-level
  breakdown, regularization)
- Root-cause artifact-rate variance across runs (~13% local vs. ~55% Colab)
- **Path C** — auxiliary-input stacking: feed Stage 2 both the raw
  128-channel signal *and* Stage 1's output/embedding, concatenated
  (distinct from B — B only sees Stage 1's learned features, C sees both)
- **Alternate strategies to explore**:
  - One-vs-Rest: 10 independent binary classifiers (one per digit),
    combined via argmax confidence
  - Multi-stream fusion: separate sub-networks per channel region (e.g.
    frontal vs. occipital), merged at the representation level rather
    than the input level
  - Self-supervised pretraining: autoencoder reconstruction pretraining,
    encoder reused as a feature extractor
  - Transformer architecture — likely more promising once data scale
    increases (see below)
- Session/block-level meta-learning — adapting to signal drift within the
  single subject over time (distinct from cross-subject meta-learning)

### Dataset Scaling (after fine-tuning, applied gradually)
1. **Current**: 20% subsample, MindBigData2023 MNIST-2B
2. **Next**: 100% MNIST-2B (`subsample_fraction=1.0`)
3. **Future**: full MNIST-8B (placeholder in `config.py` — HF repo id not
   yet confirmed)

...

## Results

(To be added)

## References

...
