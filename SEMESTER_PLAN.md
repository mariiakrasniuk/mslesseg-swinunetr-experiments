# Semester Plan — Learnable Wavelet Filters & Explainable MS Lesion Segmentation

**Author:** Mariia Krasniuk · **Supervisor:** Prof. Dalia Čalnerytė · KTU, Master's Final Degree Project
**Builds on:** Research Project 2 (RP2) — *Application of AI methods to the analysis of multiple sclerosis*

---

## 1. One-sentence summary

Improve MS lesion segmentation, especially for **small lesions**, by extending the wavelet-based SwinUNETR with **learnable filters**. Then evaluate it with **lesion-level metrics**, and use **explainability methods** to find which decomposition levels, sub-bands and learned filters actually drive each lesion prediction.

---

## 2. Where we are (RP2 recap)

**Pipeline:** FLAIR MRI → 3D DWT patch embedding (replaces SwinUNETR's `Conv3d(k=2, s=2)`) → SwinUNETR encoder/decoder → lesion mask.

**Data split** (patient-level, [splits.py](splits.py)): 40 train patients (P1–P40), 13 val (P41–P53), 22 test (P54–P75). FLAIR only.

| Model | Wavelet | Levels | Val Dice | Test Dice |
|---|---|---|---|---|
| Baseline SwinUNETR | Conv3d | – | 0.6423 | 0.595 |
| Single-level (Variant A) | Haar | 1 | 0.673 | 0.6299 |
| Multi-level + SE | Haar | 1 / 2 / 3 | 0.5717 / 0.6613 / 0.6164 | 0.5786 / 0.6293 / 0.5877 |
| Multi-level + SE | db2 | 1 / **2** / 3 | 0.6297 / **0.6783** / 0.6395 | 0.5988 / **0.6372** / 0.5999 |
| Multi-level + SE | sym4 | 1 / 2 / 3 | 0.6629 / 0.6302 / 0.6642 | 0.6301 / 0.6056 / 0.6291 |

**What RP2 established**
- Frequency-aware patch embedding with the *same parameter budget* beat the conv baseline (+0.035 test Dice for single-level Haar, +0.042 for db2 L2).
- The effect of decomposition depth is non-monotonic, and it depends on the wavelet family.

**What RP2 did *not* establish**
- **Why** wavelets help. Is it the wavelet structure, or just an extra filter bank?
- Which levels and sub-bands matter, and for which lesions.
- Whether gains come from **detecting more lesions**, or only from better overlap on lesions that are already found.
- Whether differences of ~0.01 Dice are real. Each configuration was trained **once** (one seed). Haar L1 inside the multi-level module (0.5786) vs. standalone single-level Haar (0.6299) shows that variance across runs or architectures can be larger than many of the gaps in the table.

**Available checkpoints** (`Desktop/KTU/Research Project/trained_models_masters/`):
`best_baseline.pth`, `best_wavelet_a_single_level.pth`, and `{haar,db2,sym4}/best_wavelet_ml_<family>_l{1,2,3}.pth`. That makes 11 models, all of which can be re-evaluated **without retraining**. The folder also contains a duplicate nested copy.

---

## 3. Main goal and research questions

**Main goal:** improve MS lesion segmentation in a way that shows up at the *lesion* level:
- more real lesions detected
- fewer missed lesions
- fewer false-positive lesions

Better Dice alone is not enough.

**Research questions**
1. **RQ1 — Reliability.** Do the RP2 gains over the baseline survive multiple seeds, correct thresholding and a validated DWT?
2. **RQ2 — Learnable filters.** Does letting the wavelet filters adapt during training improve segmentation, and small-lesion detection in particular?
3. **RQ3 — Does the wavelet structure matter?** Does a wavelet *initialisation* beat a random trainable filter bank of the same shape?
4. **RQ4 — How much freedom?** Fixed vs. slightly adaptable vs. fully trainable filters: should the mathematical wavelet structure be preserved?
5. **RQ5 — Scale and frequency.** Which decomposition levels and sub-bands contribute most? Do small lesions rely more on fine-scale / high-frequency bands, and large lesions more on coarse / low-frequency context? *(Hypothesis to be measured, not assumed.)*
6. **RQ6 — Explainability.** For a given detected, missed or false-positive lesion, which input region, level, sub-band and filter drove the prediction?
7. **RQ7 — Learned filters.** How did the trained filters change relative to their initialisation, and what are they now sensitive to?

---

## 4. Work plan (phased)

The phases are ordered so that **each one produces something usable even if later phases slip**.

### Phase 0 — Make the RP2 results trustworthy *(do this first)*

Nothing below should be concluded from single-seed, soft-Dice numbers.

- [ ] **Fix and verify thresholding.** [train.py](train.py) and [final_test_evaluation.py](final_test_evaluation.py) pass `sigmoid(logits)` straight into `DiceMetric`, on every branch including `master`. With a 1-channel input, MONAI 1.5's `DiceMetric` does **not** threshold. It counts only voxels whose probability is exactly 1.0 in float32 (logit > ~16.6). RP2 Dice, and the checkpoint selection based on it, therefore used an extreme cut-off that depends on each model's logit scale. On test patient P54 (db2 L2), this metric gives 0.851, while Dice at a 0.5 threshold is 0.649. [evaluate_lesions.py](evaluate_lesions.py) reports both. Add `AsDiscrete(threshold=0.5)` (or an equivalent step) for evaluation. Optionally, tune the threshold on validation only and never on test.
- [ ] **Validate the DWT implementation** against PyWavelets for Haar, db2 and sym4 on real FLAIR crops:
  - Filter orientation: `F.conv3d` is cross-correlation, so the filters must be the correctly flipped `dec_lo` / `dec_hi`.
  - Sub-band ordering.
  - Boundary handling: zero padding vs. PyWavelets' symmetric/periodic modes.
  - Perfect reconstruction / energy preservation (Parseval) where applicable.
- [ ] **Check the multi-level recursion.** In `WaveletPatchEmbedML`, the next level decomposes the **SE-rescaled** LLL (`approx = sub[:, :1]` after `se_blocks[i]`), so levels ≥2 are not a pure DWT of the input. Decide whether this is intended. A cleaner design applies SE *after* the full decomposition.
- [ ] **Reconcile the training config** with the report. When the RP2 runs were made (commits up to April 2026), the code had `EPOCHS=70, PATIENCE=10`. RP2 states "up to 100 epochs". Every run stopped early (≤59 epochs), so the numbers are unaffected; only the report text needs correcting. Commit `13fde49` (June 2026) later raised this to `EPOCHS=150, PATIENCE=20`, so new runs are not directly comparable to RP2. Also note that checkpoints are selected on val Dice, while early stopping monitors val loss.
- [ ] **Re-run the key configurations with ≥3 seeds:** baseline, single-level Haar, db2 L2, sym4 L1/L3. If compute allows, use patient-level k-fold cross-validation over P1–P53 instead of a single val split. Report mean ± std and a paired test across test patients (e.g. Wilcoxon).

**Deliverable:** a corrected RP2 table with error bars. It is the reference point for everything else.

### Phase 1 — Lesion-level evaluation framework

Build this once, then use it on every model, including the 11 existing checkpoints.

- [ ] Connected-component lesion extraction on the GT and on the prediction. Fix and document:
  - connectivity (26-conn)
  - minimum lesion size (e.g. ≥3 voxels)
  - the matching rule for a "detected" lesion (≥1 voxel overlap, or IoU > threshold)
- [ ] Metrics per patient and aggregated:
  - **Voxel-level:** Dice (binarised), precision, recall, AVD
  - **Lesion-level:** lesion recall (LTPR), lesion precision, **lesion F1**, **false-positive lesions per patient**, missed lesions per patient
  - **Boundary:** HD95, ASSD
- [ ] **Stratify by lesion size:** small / medium / large. Choose the bin edges from the MSLesSeg lesion-volume distribution (e.g. tertiles) or from literature thresholds, fix them once, and report per-bin lesion recall.
- [ ] Save per-lesion records (patient, lesion id, volume, location, detected / missed / FP, model). These feed the explainability phase directly.

**Deliverable:** `evaluate_lesions.py` plus a lesion-level results table for the baseline and all RP2 wavelet models. This already answers "do wavelets find more small lesions?" for the fixed filters.

### Phase 2 — Learnable filters *(core contribution)*

Make the 1D analysis filters `lo` / `hi` `nn.Parameter`s. Keep building the 3D bank as separable outer products, so the model has only a few dozen parameters per filter pair, not a free 3D kernel.

**Experiment matrix** (same data, same config, ≥3 seeds each):

| # | Variant | Initialisation | Trainable? | Purpose |
|---|---|---|---|---|
| E0 | Baseline SwinUNETR | – | Conv3d | Reference |
| E1 | Fixed db2 (L2) | db2 | No | RP2 best |
| E2 | Trainable db2 | db2 | Fully | Does adaptation help? |
| E3 | Constrained trainable db2 | db2 | Yes, with wavelet constraints | Keep the structure but adapt |
| E4 | Random trainable filters | Random, same length | Fully | **Control:** is the gain wavelet-specific? |
| E5 | Trainable Haar / sym4 | Haar / sym4 | Fully | Does the initial family still matter after training? |

**How to read the results**
- If E1 > E0, E2 > E1 and E4 < E2: the wavelet initialisation matters, and adapting it helps.
- If E4 ≈ E2: the gain probably comes from having an extra trainable filter bank, not from wavelets specifically.

**Degree of freedom (RQ4).** Pick one of these ways to implement "slightly adaptable":
- Parameterise `lo = lo_init + δ` and add an L2 penalty on δ, or clamp ‖δ‖.
- Use a separate small learning rate for the filter parameters.
- Constrained version: derive `hi` from `lo` via the quadrature-mirror relation, and add soft orthonormality / `Σlo = √2`, `Σhi = 0` penalties. This keeps it a valid wavelet.

Sweep the penalty strength or learning-rate ratio (e.g. 3 values) to get a fixed → slightly adaptable → free curve.

**Level ablation with learnable filters.** Repeat L1 / L2 / L3 for the best learnable variant only, not the full grid.

**Deliverable:** results table (voxel-level and lesion-level, mean ± std) answering RQ2–RQ4.

### Phase 3 — Sub-band and level importance

- [ ] **Test-time band ablation.** For a trained model, remove one sub-band (or one whole level) before the projection, then measure the change in Dice, lesion recall and per-size-bin recall.
  - Removing a band can mean zeroing it, or replacing it with its training-set mean. Zeroing LLL is strongly out-of-distribution, so report both variants.
- [ ] Optional, more expensive: **retrain without a band or level** on the best configuration, to separate "the model relies on it" from "the model needs it".
- [ ] Read the learned **SE weights** and the **1×1×1 projection weights** per band as a cheap, complementary importance signal.

**Deliverable:** importance heatmap (bands × levels) overall and per lesion-size bin. This gives a first answer to RQ5.

### Phase 4 — Explainability, per lesion

- [ ] **Attribution target:** the sum of logits inside one lesion component, which gives an explanation per lesion rather than per volume.
  - Detected lesions and FPs: use the predicted component.
  - Missed lesions: use the GT component.
- [ ] **Methods:**
  - Gradient × Input and **Integrated Gradients** computed on the *sub-band tensor*, i.e. the input to the projection. This directly gives attribution per level and per band.
  - Optionally, the same on the raw FLAIR for spatial maps.
  - Consider LRP only if time allows; it is harder to apply correctly through Swin attention.
- [ ] **Analyse by outcome and size:** detected vs. missed vs. FP lesions, and small vs. medium vs. large. Test RQ5's hypothesis statistically. For example, compare the fraction of attribution in level-1 high-frequency bands between small and large lesions.
- [ ] Sanity checks for the attributions: model-randomisation test, and agreement with the Phase 3 ablation.

**Deliverable:** per-lesion attribution dataset plus figures: example lesions and aggregate level/band attribution by size and outcome.

### Phase 5 — How the learned filters changed

- [ ] Plot the coefficients before and after training (e.g. db2 init vs. trained).
- [ ] Plot the **frequency response** (magnitude of the DFT of `lo` / `hi`): did the pass-bands shift or sharpen?
- [ ] Check the wavelet properties after training: orthogonality, vanishing moments (sum of moments), symmetry.
- [ ] Compare **feature maps around lesions**, fixed vs. learned filters. Is the high-pass response more concentrated on lesion boundaries relative to normal tissue edges?
- [ ] Compare learned filters across seeds: do they converge to the same thing?

**Deliverable:** answer to RQ7 with figures.

---

## 5. Possible later extensions *(only if the core is done)*

- **Fixed + learned dual branch.** A fixed wavelet branch and a learnable filter branch, fused before SwinUNETR. Use attribution to see which branch drives which lesions.
- **Robustness.** Test-time noise, blur, intensity / bias-field shifts and resolution changes. Compare the stability of the baseline, fixed wavelets and learned filters. This relates to MRI scanner variability, and the patient scanner info CSVs are in the dataset.
- **Multimodal (FLAIR + T1 + T2).** The code already supports `--multimodal`, and the current branch also has `--use_v2` and `--aug`. Keep this **secondary**: combining multimodality with learnable filters at the same time would make it impossible to attribute any improvement.

---

## 6. Evaluation protocol (applies to all experiments)

- Same patient-level split (or the same CV folds) and the same preprocessing and training config for every run.
- ≥3 seeds per configuration. Report mean ± std. Use paired statistical tests across test patients.
- The **binarisation threshold** is fixed or tuned on validation only.
- The **test set is used only for final numbers**. It is never used for choosing thresholds, levels or checkpoints.
- Every table reports **both** voxel-level (Dice, HD95/ASSD) and lesion-level (lesion recall / precision / F1, FP lesions per patient, per-size recall) metrics.

---

## 7. Expected contributions

1. A corrected, statistically grounded comparison of fixed wavelet patch embeddings vs. the conv baseline for MS lesion segmentation.
2. A learnable wavelet patch embedding, with controls that separate "wavelet prior" from "extra filter bank".
3. Evidence on how much the filters should be allowed to deviate from a true wavelet.
4. Lesion-level and size-stratified evaluation showing whether frequency-aware embeddings help **small-lesion detection**.
5. Per-lesion explanations linking predictions (hits, misses, FPs) to decomposition levels, sub-bands and learned filters.

---

## 8. Decisions to agree with the supervisor

- Single fixed split + seeds, or full patient-level k-fold CV? This depends on the compute budget.
- Lesion size bin edges and the lesion-matching rule.
- The "slightly adaptable" mechanism: penalty to init, low LR, or hard constraints.
- Whether to fix the SE-before-recursion design now. This changes the RP2 architecture, so the RP2 comparison would need re-running.
- Scope: which of the later extensions (if any) belong in the thesis.

---

## 9. Suggested timeline *(adjust to the semester calendar)*

| Weeks | Work |
|---|---|
| 1–2 | Phase 0: thresholding fix, DWT validation, config reconciliation; start seed re-runs |
| 2–4 | Phase 1: lesion-level evaluation; evaluate all 11 existing checkpoints |
| 4–8 | Phase 2: learnable filters E1–E5, multi-seed |
| 8–10 | Phase 3: band / level ablations |
| 10–13 | Phase 4: per-lesion explainability |
| 13–14 | Phase 5: learned-filter analysis |
| 14+ | Optional extensions; writing |
