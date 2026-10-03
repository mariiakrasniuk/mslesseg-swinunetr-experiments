# Multi-seed results (results/exp3_test)

## thr0.5

| config | dice | lesion_f1 | lesion_recall | lesion_precision | n_fp_lesions | hd95 | recall_small |
|---|---|---|---|---|---|---|---|
| baseline | 0.680 ± 0.010 | 0.685 ± 0.003 | 0.708 ± 0.015 | 0.721 ± 0.019 | 6.606 ± 0.936 | 21.931 ± 0.816 | 0.319 ± 0.020 |
| baseline_dicebce | 0.683 ± 0.011 | 0.679 ± 0.022 | 0.726 ± 0.035 | 0.698 ± 0.044 | 8.212 ± 2.286 | 22.995 ± 1.994 | 0.330 ± 0.057 |
| baseline_dicebce_hf1 | 0.678 ± 0.009 | 0.682 ± 0.008 | 0.706 ± 0.039 | 0.719 ± 0.044 | 6.606 ± 2.036 | 23.313 ± 1.206 | 0.317 ± 0.044 |

## val_thr

| config | dice | lesion_f1 | fp_at_t_f1 | recall_small_at_t_f1 |
|---|---|---|---|---|
| baseline | 0.680 ± 0.009 | 0.710 ± 0.008 | 5.742 ± 0.329 | 0.294 ± 0.013 |
| baseline_dicebce | 0.682 ± 0.010 | 0.706 ± 0.013 | 6.470 ± 1.962 | 0.286 ± 0.016 |
| baseline_dicebce_hf1 | 0.680 ± 0.006 | 0.705 ± 0.005 | 5.106 ± 0.956 | 0.268 ± 0.009 |

## matched_fp

| config | lesion_recall@5fp | lesion_recall@6fp | lesion_recall@7fp | lesion_recall@8fp | recall_small@5fp | recall_small@6fp | recall_small@7fp | recall_small@8fp |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.609 ± 0.009 | 0.632 ± 0.016 | 0.660 ± 0.018 | 0.674 ± 0.020 | 0.272 ± 0.004 | 0.301 ± 0.017 | 0.340 ± 0.026 | 0.356 ± 0.011 |
| baseline_dicebce | 0.600 ± 0.035 | 0.622 ± 0.030 | 0.639 ± 0.034 | 0.661 ± 0.037 | 0.249 ± 0.050 | 0.279 ± 0.046 | 0.299 ± 0.053 | 0.327 ± 0.051 |
| baseline_dicebce_hf1 | 0.615 ± 0.011 | 0.632 ± 0.015 | 0.659 ± 0.015 | 0.680 ± 0.009 | 0.276 ± 0.029 | 0.293 ± 0.027 | 0.330 ± 0.034 | 0.356 ± 0.023 |

## Paired vs baseline (Δ mean, Wilcoxon p)

| config | dice@0.5 | lesion_f1@0.5 | dice@val_thr |
|---|---|---|---|
| baseline_dicebce | +0.0032 (p=0.899) | -0.0052 (p=0.337) | +0.0018 (p=0.524) |
| baseline_dicebce_hf1 | -0.0014 (p=0.899) | -0.0025 (p=0.799) | -0.0004 (p=0.899) |
