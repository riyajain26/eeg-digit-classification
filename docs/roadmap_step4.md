## Stage 4 Roadmap — Advanced Architectures & Full-Scale Data

**Status**: Not Started
**Depends on**: Stage 3 complete (modular pipeline, alternatives compared,
strongest candidate(s) identified and swept).

### Why this stage comes last

These are the most expensive, most architecturally-involved directions —
each is effectively its own mini-project. They're only worth investing in
once Stage 3 has established a solid, well-understood baseline pipeline to
build them on top of (and, per Stage 3's diagnostics, confirmed that what's
being learned is genuine digit-related signal).

### Variants to Build & Compare

| Variant | Status | Description |
|---|---|---|
| **One-vs-Rest** | Not Started | 10 independent binary classifiers (one per digit), combined via argmax confidence |
| **Multi-stream fusion** | Not Started | Separate sub-networks per channel region (e.g. frontal vs. occipital), merged at the representation level rather than the input level |
| **Self-supervised pretraining** | Not Started | Autoencoder reconstruction pretraining; encoder reused as a feature extractor |
| **Transformer architecture** | Not Started | Likely more promising once data scale increases — see Dataset Scaling below |
| **Session/block-level meta-learning** | Not Started | Adapting to signal drift within the single subject over time (distinct from cross-subject meta-learning) |

Each follows the same registry pattern as Stage 2/3: standalone function,
one registry entry, no changes needed elsewhere in the pipeline.

### Dataset Scaling

Once the above are implemented and compared at the current data scale, retrain
the strongest candidate(s) — plus Stage 2 and Stage 3's winning combination —
on the full MNIST-8B release (100%), pending confirmation of its Hugging Face
repo id (currently a placeholder in `config.py`'s `DATASET_VARIANTS["8B"]`).

### Evaluation

Same standard as every prior stage: permutation test (shuffled-label floor)
for every variant, val + held-out test evaluation, no variant trusted on raw
accuracy alone.

### Definition of Done

- [ ] Each variant above built, trained, evaluated (val + test + permutation)
- [ ] Comparison against Stage 3's best pipeline written up
- [ ] Strongest overall approach retrained and evaluated on the full 8B dataset
