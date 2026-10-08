# Multi-seed results (results/exp5c_test)

## thr0.5

| config | dice | lesion_f1 | lesion_recall | lesion_precision | n_fp_lesions | hd95 | recall_small |
|---|---|---|---|---|---|---|---|
| baseline_r2 | 0.692 ± 0.005 | 0.679 ± 0.007 | 0.737 ± 0.015 | 0.688 ± 0.021 | 8.485 ± 1.189 | 23.855 ± 1.499 | 0.355 ± 0.024 |
| waveup_haar_r2 | 0.690 ± 0.006 | 0.679 ± 0.012 | 0.729 ± 0.013 | 0.697 ± 0.026 | 8.053 ± 1.546 | 24.065 ± 2.239 | 0.353 ± 0.019 |
| waveup_rand_learn_r2 | 0.689 ± 0.007 | 0.684 ± 0.015 | 0.729 ± 0.018 | 0.700 ± 0.029 | 7.773 ± 1.489 | 24.040 ± 1.620 | 0.368 ± 0.026 |

## val_thr

| config | dice | lesion_f1 | fp_at_t_f1 | recall_small_at_t_f1 |
|---|---|---|---|---|
| baseline_r2 | 0.692 ± 0.005 | 0.719 ± 0.007 | 6.629 ± 1.276 | 0.305 ± 0.033 |
| waveup_haar_r2 | 0.688 ± 0.008 | 0.715 ± 0.004 | 7.152 ± 1.816 | 0.327 ± 0.036 |
| waveup_rand_learn_r2 | 0.689 ± 0.007 | 0.722 ± 0.009 | 6.720 ± 1.301 | 0.331 ± 0.016 |

## matched_fp

| config | lesion_recall@5fp | lesion_recall@6fp | lesion_recall@7fp | lesion_recall@8fp | recall_small@5fp | recall_small@6fp | recall_small@7fp | recall_small@8fp |
|---|---|---|---|---|---|---|---|---|
| baseline_r2 | 0.619 ± 0.021 | 0.639 ± 0.013 | 0.660 ± 0.008 | 0.678 ± 0.011 | 0.269 ± 0.027 | 0.292 ± 0.018 | 0.316 ± 0.016 | 0.346 ± 0.027 |
| waveup_haar_r2 | 0.619 ± 0.005 | 0.639 ± 0.007 | 0.658 ± 0.011 | 0.680 ± 0.017 | 0.282 ± 0.013 | 0.308 ± 0.017 | 0.334 ± 0.017 | 0.365 ± 0.025 |
| waveup_rand_learn_r2 | 0.633 ± 0.012 | 0.652 ± 0.013 | 0.667 ± 0.016 | 0.682 ± 0.012 | 0.301 ± 0.016 | 0.322 ± 0.025 | 0.349 ± 0.027 | 0.369 ± 0.026 |

## Paired vs baseline_r2 (Δ mean, Wilcoxon p)

| config | dice@0.5 | lesion_f1@0.5 | dice@val_thr |
|---|---|---|---|
| waveup_haar_r2 | -0.0020 (p=0.321) | +0.0001 (p=0.899) | -0.0035 (p=0.235) |
| waveup_rand_learn_r2 | -0.0031 (p=0.588) | +0.0050 (p=0.443) | -0.0033 (p=0.726) |
