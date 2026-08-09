# EEG Digit Classification Roadmap

## Project Goal

Develop an end-to-end EEG classification pipeline using the **MindBigData2023 MNIST-2B** dataset.

The project will follow a hierarchical classification approach:

### Stage 1 — Stimulus Detection

**Binary classification**

* Blank/rest EEG (`label = -1`)
* Digit stimulus EEG (`label = 0–9`)

**Goal**

Determine whether EEG contains detectable information about visual stimulus presence.

### Stage 2 — Digit Decoding

**Multi-class classification**

* Digit classes: `0–9`

**Goal**

Determine whether EEG responses contain enough information to identify the presented digit.

The initial milestone is to build a **clean, reproducible end-to-end pipeline** with honest baseline results. Model performance optimization will come later.

---

## Scope Note: Stage 1 First, Stage 2 Incremental

This project builds Stage 1 (binary: blank vs. digit) completely first,
including a working baseline and deep learning model, before starting
Stage 2 (digit 0-9).

Stage 2 is **not a separate track** — it reuses Stage 1's trained backbone
(feature-reuse design) and repeats several of the same pipeline steps on
digit-only data:

| Step                          | Stage 1 status | Stage 2 status |
|--------------------------------|----------------|----------------|
| Data acquisition & storage     | Done           | Done (same files, filtered by `label_digit != -1`) |
| Leakage-safe splitting         | Done           | Done (same split, digit-only subset) |
| Permutation-test harness       | Done           | Deferred — needs stratified sampling |
| Preprocessing pipeline         | In progress    | Deferred |
| Baseline models (Phase 5)      | Not started    | Deferred |
| Deep learning model (Phase 6)  | Not started    | Deferred — depends on Stage 1's trained backbone |
| Evaluation (Phase 7)           | Not started    | Deferred |

Rule of thumb: any phase marked "Deferred" for Stage 2 gets revisited only
after the same phase is complete for Stage 1.

## Scope Note: Dataset Scale-Up Plan

Data volume scales independently of the Stage 1/2 axis above, in three
planned steps:

1. **Current**: ~20% stratified subsample of **MindBigData2023 MNIST-2B**
   (the reduced Hugging Face release) — used to validate the full pipeline
   end-to-end cheaply before committing to larger runs.
2. **Next**: scale to **100% of MNIST-2B** — once the pipeline (data prep,
   splitting, preprocessing, baseline models) is proven correct on the
   subsample, re-run against the full 2B release.
3. **Final**: scale to **100% of the original MindBigData2023 MNIST-8B**
   dataset — the full, un-reduced release — once the pipeline has been
   validated at the 2B scale.

Rule of thumb: don't scale up until the current scale's results (and
leakage/permutation checks) look correct and trustworthy. Chasing bigger
data on top of an unverified pipeline just means bigger, slower mistakes.

---

# Phase 0 — Scoping Decisions

**Status:** Completed (S1, reduced 2B)

## Dataset Selection

**Dataset:** MindBigData2023 MNIST-2B

* Hugging Face version
* Initial development on a **10–20% stratified subset**
* Scale to the complete dataset after validating the pipeline

## Classification Strategy

Rather than directly attempting 10-class digit classification, the project will progress through two stages.

### Stage 1 — Binary Classification

**Task**

* Blank (`-1`) vs Digit (`0–9`)

**Goal**

Determine whether EEG contains sufficient information to detect the presence of a visual stimulus.

### Stage 2 — Multi-class Classification

**Task**

* Digit (`0–9`)

**Goal**

Determine whether EEG contains enough information to identify which digit was presented.

## Success Criteria

The objective is **not** achieving a fixed accuracy target.

Success means:

* Working end-to-end pipeline
* Reproducible experiments
* Honest baseline performance
* Clean and modular codebase
* Easily extensible architecture

---

# Phase 1 — Project & Environment Setup

**Status:** Completed (S1, reduced 2B)

## Repository Structure

```text
eeg-digit-classification/

├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
├── models/
├── notebooks/
├── results/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── README.md
├── roadmap.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Development Workflow

### Development

* VS Code
* Python virtual environment
* Git & GitHub

### Training

* Local CPU for development and debugging
* Google Colab GPU for deep learning experiments

## Experiment Tracking

### Initial

* Local CSV/log-based experiment tracking

### Future

* Integrate Weights & Biases after establishing baseline models

---

# Phase 2 — Data Acquisition & Storage

**Status:** Completed (S1, reduced 2B)

## Objectives

Understand the dataset completely before writing preprocessing or training code.

## Dataset Structure

Each row represents **one EEG trial**.

### EEG Data

* 128 EEG channels
* 256 samples per channel
* 250 Hz sampling rate
* ~1-second recording

```
128 × 500 = 64,000 EEG values
```

### Labels

* `-1` → Blank screen
* `0–9` → Presented digit

Additional label information:

* `label_source`
* `label_pos`
* `label_imgpix_0` ... `label_imgpix_783`

### Metadata

* `timestamp`
* `sessionnum`
* `blocknum`
* `blockpos`

## Data Validation

Completed in Notebook 02, Sections 4 & 6:

* Verified EEG array shape consistency
* Checked for NaN/corrupt values
* Verified binary and digit-class distribution
* Confirmed metadata present on every trial

(Deferred to Phase 3: session distribution and channel-ordering-across-trials
checks, since they're more naturally part of split validation.)

## Data Conversion

Converted via Hugging Face streaming (no full download) into array-based HDF5
— not per-trial groups, to avoid HDF5 metadata overhead at scale.

Target representation:

```
eeg (N, 128, 256) float32
label_binary (N,) 0=blank, 1=digit
label_digit (N,) -1=blank, 0-9=digit
```

metadata arrays (sessionnum, blocknum, blockpos, timestamp), shape (N,) each

**Two separate files produced:**

```
data/processed/mindbigdata2023_train.h5 # train/val pool (~28,000 trials target)
data/processed/mindbigdata2023_test.h5 # held-out test set (~6,000 trials target)
```

Storage format: HDF5 (chosen for partial/indexed reads without loading the
full file into memory, plus built-in per-chunk compression).

---

# Phase 3 — Leakage-Safe Dataset Splitting

**Status:** Completed (S1, reduced 2B)

## Objectives

Create reproducible train/validation/test datasets while preventing information leakage.

## Test Set Sanity Check (new — first step)

Before splitting, verify Hugging Face's official train/test boundary is
itself leakage-safe: check for `sessionnum` overlap between
`mindbigdata2023_train.h5` and `mindbigdata2023_test.h5`. If clean, adopt
their test set as final and only split train/val ourselves. If overlap is
found, fall back to deriving a custom 3-way split.

## Dataset Splitting

* Trial-level split (never window-level) — applies to the train/val boundary
  only, since test is (pending the check above) already separated
* Stratified sampling
* Fixed random seed

### Stage 1
Blank (`-1`) vs Digit (`0–9`)

### Stage 2
Digits only (`0–9`)

## Leakage Prevention

Investigate different splitting strategies for train/val:

* Random trial split
* Session-aware split
* Chronological split

Use metadata:

* `sessionnum`
* `blocknum`
* `timestamp`

## Validation

Implement a permutation-test baseline (shuffled labels) before training any real models.

---

# Phase 4 — EEG Preprocessing Pipeline

**Status:** Completed (S1, reduced 2B)

## Objectives

Develop a reusable preprocessing pipeline.

## Signal Processing

* Bandpass filtering
* Notch filtering
* Signal normalization
* Basic artifact detection

## Feature Paths

### Path A — Classical Machine Learning

Extract features such as:

* Band power
* Statistical features
* Frequency-domain features

### Path B — Deep Learning

Use filtered raw EEG directly.

Input format:

```
(channels, time)
```

The preprocessing pipeline should be configurable for future experiments.

---

# Phase 5 — Baseline Models

**Status:** Not Started

## Stage 1 — Binary Classification

### Task

Blank (`-1`) vs Digit (`0–9`)

### Models

* Linear Discriminant Analysis (LDA)
* Support Vector Machine (SVM)
* Random Forest

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Goal

Verify that EEG signals contain detectable information about visual stimulus presence.

---

## Stage 2 — Multi-class Classification

### Task

Digit (`0–9`)

### Models

* Linear Discriminant Analysis (LDA)
* Support Vector Machine (SVM)
* Random Forest

### Evaluation

* Accuracy
* Macro F1-score
* Per-digit confusion matrix
* Permutation baseline comparison

---

# Phase 6 — Deep Learning Models

**Status:** Not Started

## Primary Model

**EEGNet**

Train separate models for:

* Binary stimulus detection
* Multi-class digit classification

## Optional Models

* 1D CNN
* LSTM

Track:

* Training loss
* Validation loss
* Accuracy
* Learning curves
* Saved checkpoints

---

# Phase 7 — Evaluation & Analysis

**Status:** Not Started

## Binary Classification

Evaluate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC (optional)

Research Question:

> Can EEG reliably distinguish between blank and visual stimulus trials?

---

## Multi-class Classification

Evaluate:

* Accuracy
* Macro F1-score
* Per-digit confusion matrix

Analyze:

* Most confused digit pairs
* Failure cases
* Error distribution

Research Question:

> Which digits are most distinguishable from EEG, and which are consistently confused?

---

## Two-Week Milestone

Deliverables:

* Complete preprocessing pipeline
* ML-ready dataset
* Leakage-safe data split
* Baseline classical ML models
* Initial EEGNet implementation
* Reproducible experiments
* Results notebook
* Initial findings report

---

# Future Extensions

Potential research directions:

* Advanced artifact removal (ICA)
* Transfer learning with pretrained EEG models
* Cross-session generalization
* Cross-subject evaluation
* Transformer-based architectures
* Self-supervised representation learning
* Real-time EEG decoding
* Brain–Computer Interface (BCI) applications

