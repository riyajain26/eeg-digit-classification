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

# Phase 0 — Scoping Decisions

**Status:** Completed

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

**Status:** Completed

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

**Status:** In Progress

## Objectives

Understand the dataset completely before writing preprocessing or training code.

## Dataset Structure

Each row represents **one EEG trial**.

### EEG Data

* 128 EEG channels
* 500 samples per channel
* 250 Hz sampling rate
* 2-second recording

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

Perform sanity checks:

* Verify channel count
* Verify sample count per channel
* Check missing values
* Check corrupt trials
* Check duplicate trials
* Verify class distribution
* Measure blank (`-1`) distribution
* Verify session distribution
* Confirm consistent channel ordering

### Questions

* What is the exact file size after conversion?
* Are there any incomplete or corrupted trials?
* Are there any irregular trial lengths?
* Is channel ordering identical across every trial?

## Data Conversion

Convert the Hugging Face dataset into an ML-ready format.

Target representation:

```text
X

(number_of_trials,
128 channels,
500 samples)
```

```text
y

(number_of_trials,)
```

Store metadata separately.

Preferred storage format:

* HDF5

---

# Phase 3 — Leakage-Safe Dataset Splitting

**Status:** Not Started

## Objectives

Create reproducible train/validation/test datasets while preventing information leakage.

## Dataset Splitting

* Trial-level split (never window-level)
* Train / Validation / Test
* Stratified sampling
* Fixed random seed

### Stage 1

Blank (`-1`) vs Digit (`0–9`)

### Stage 2

Digits only (`0–9`)

## Leakage Prevention

Investigate different splitting strategies:

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

**Status:** Not Started

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

