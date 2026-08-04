# EEG Digit Classification — Development Roadmap

## Project Timeline

**Target timeline:** 1–2 months

**Initial milestone:** Build a complete working EEG classification pipeline within 2 weeks.

The first milestone focuses on creating a reliable end-to-end framework using a smaller dataset subset. Future iterations will scale the dataset size, improve preprocessing, and explore more advanced models.

---

# Phase 0 — Project Scoping

**Status:** Completed

## Objectives

Define the initial scope of the project before implementation.

## Decisions

### Dataset

* Dataset: MindBigData 2023 — MNIST-2B
* Source: Hugging Face
* Initial experiment size: 10–20% stratified subset
* Future plan: Scale to larger/full dataset

### Success Criteria

The first milestone is not based on achieving a fixed accuracy target.

Success means:

* Complete data pipeline
* Reproducible experiments
* Honest baseline performance
* Working evaluation pipeline

The goal is to establish a strong foundation before optimization.

---

# Phase 1 — Project & Environment Setup

**Status:** Completed

## Objectives

Create a clean and reproducible development environment.

## Tasks

* Create Git repository
* Define project structure
* Configure `.gitignore`
* Create Python environment
* Create `requirements.txt`
* Create documentation structure
* Decide development workflow:

  * Local development
  * Colab/GPU training

## Deliverables

* Working repository
* Reproducible environment
* Initial documentation

---

# Phase 2 — Data Acquisition & Storage

**Status:** In Progress

## Objectives

Understand, acquire, validate, and organize the EEG dataset.

## Tasks

### Dataset Exploration

* Inspect Hugging Face dataset structure
* Understand:

  * Number of trials
  * EEG channels
  * Sampling frequency
  * Epoch length
  * Labels
  * Metadata

### Data Download

* Download selected subset
* Store original files in:

```
data/raw/
```

### Data Processing

Convert raw dataset into efficient machine learning format:

```
data/processed/
```

Preferred format:

* HDF5

Alternative:

* NPZ

### Validation Checks

Verify:

* Class distribution
* Missing values
* Corrupted trials
* Sample length consistency
* Sampling frequency consistency

### Decisions

Document:

* Padding/truncation strategy
* Data normalization strategy
* Storage format

## Deliverables

* Clean processed dataset
* Data validation report
* Dataset loading script

---

# Phase 3 — Leakage-Safe Dataset Splitting

## Objectives

Create reliable train, validation, and test datasets.

## Tasks

* Define split strategy
* Perform trial-level splitting
* Prevent window-level leakage
* Stratify by digit labels
* Create reproducible random seeds

## Initial Split

Potential starting point:

* Training: 70%
* Validation: 15%
* Testing: 15%

Final split may change after dataset inspection.

## Additional Validation

Implement:

* Label permutation test
* Random baseline comparison

## Deliverables

* Train/validation/test datasets
* Split generation script
* Leakage checks

---

# Phase 4 — EEG Preprocessing Pipeline

## Objectives

Create reusable EEG preprocessing pipelines.

## Signal Processing

Initial preprocessing:

* Bandpass filtering
* Notch filtering
* Basic artifact handling

Advanced methods for later:

* ICA
* Automated artifact rejection

## Two Processing Paths

### Classical ML Path

Extract:

* Band power features
* Statistical features
* Frequency-domain features

### Deep Learning Path

Use:

* Filtered raw EEG signals

## Configuration

Make preprocessing parameters configurable:

Example:

```
configs/preprocessing.yaml
```

Parameters:

* Frequency bands
* Filter settings
* Artifact thresholds

## Deliverables

* Preprocessing pipeline
* Feature extraction pipeline
* Visualization notebooks

---

# Phase 5 — Classical Machine Learning Baselines

## Objectives

Establish baseline performance before deep learning.

## Models

Implement:

* Linear Discriminant Analysis
* Support Vector Machine
* Random Forest

## Evaluation

Metrics:

* Accuracy
* Macro F1-score
* Confusion matrix
* Per-digit performance

Compare against:

* Random classifier
* Permutation baseline

## Deliverables

* Baseline results
* Evaluation framework

---

# Phase 6 — Deep Learning Pipeline

## Objectives

Train EEG-specific deep learning models.

## Primary Model

EEGNet-style CNN

Reasons:

* Designed for EEG
* Lightweight
* Strong baseline
* Suitable for limited datasets

## Additional Experiments

Possible comparisons:

* 1D CNN
* LSTM
* Transformer-based models (future)

## Training Pipeline

Implement:

* Training loop
* Validation loop
* Checkpoint saving
* Learning rate scheduling
* Early stopping

## Deliverables

* Deep learning baseline
* Saved model checkpoints
* Training curves

---

# Phase 7 — Evaluation & 2-Week Milestone

## Objectives

Analyze results and document progress.

## Evaluation

Report:

* Test accuracy
* Macro F1-score
* Confusion matrix
* Digit-level performance

## Error Analysis

Investigate:

* Commonly confused digits
* Subject/session effects
* Signal quality issues

## Documentation

Update:

* README
* Results notebook
* Experiment notes

## Deliverable

A complete working EEG classification framework.

---

# Future Improvements

After the initial milestone:

## Dataset

* Increase dataset size
* Explore full MindBigData dataset
* Evaluate generalization

## Signal Processing

* Advanced artifact removal
* ICA
* Better normalization strategies

## Modeling

* Hyperparameter optimization
* Larger architectures
* Transfer learning
* Pretrained EEG models

## Engineering

* Experiment tracking
* Automated pipelines
* Model deployment experiments

---

# Current Progress

| Phase                    | Status         |
| ------------------------ | -------------- |
| Phase 0 — Scoping        | ✅ Complete     |
| Phase 1 — Setup          | 🚧 In Progress |
| Phase 2 — Data Pipeline  | ⬜ Not Started  |
| Phase 3 — Data Splitting | ⬜ Not Started  |
| Phase 4 — Preprocessing  | ⬜ Not Started  |
| Phase 5 — ML Baselines   | ⬜ Not Started  |
| Phase 6 — Deep Learning  | ⬜ Not Started  |
| Phase 7 — Evaluation     | ⬜ Not Started  |

