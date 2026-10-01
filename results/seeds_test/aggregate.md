# Multi-seed results (results/seeds_test)

## thr0.5

| config | dice | lesion_f1 | lesion_recall | lesion_precision | n_fp_lesions | hd95 | recall_small |
|---|---|---|---|---|---|---|---|
| baseline | 0.680 ± 0.010 | 0.685 ± 0.003 | 0.708 ± 0.015 | 0.721 ± 0.019 | 6.606 ± 0.936 | 21.931 ± 0.816 | 0.319 ± 0.020 |
| wavelet_ml_db2_l2 | 0.676 ± 0.009 | 0.671 ± 0.003 | 0.713 ± 0.033 | 0.697 ± 0.025 | 8.227 ± 1.547 | 22.579 ± 0.750 | 0.348 ± 0.064 |
| wavelet_ml_haar_l2 | 0.674 ± 0.004 | 0.670 ± 0.038 | 0.721 ± 0.009 | 0.688 ± 0.048 | 8.318 ± 2.741 | 22.429 ± 2.489 | 0.325 ± 0.034 |
| wavelet_ml_haar_l3 | 0.679 ± 0.007 | 0.674 ± 0.036 | 0.753 ± 0.008 | 0.666 ± 0.054 | 9.970 ± 2.760 | 21.797 ± 2.058 | 0.392 ± 0.022 |
| wavelet_ml_sym4_l1 | 0.660 ± 0.019 | 0.667 ± 0.037 | 0.689 ± 0.052 | 0.717 ± 0.047 | 7.561 ± 2.066 | 23.984 ± 3.085 | 0.292 ± 0.065 |

## val_thr

| config | dice | lesion_f1 | fp_at_t_f1 | recall_small_at_t_f1 |
|---|---|---|---|---|
| baseline | 0.680 ± 0.009 | 0.710 ± 0.008 | 5.742 ± 0.329 | 0.294 ± 0.013 |
| wavelet_ml_db2_l2 | 0.677 ± 0.010 | 0.702 ± 0.011 | 8.061 ± 1.509 | 0.332 ± 0.054 |
| wavelet_ml_haar_l2 | 0.675 ± 0.007 | 0.700 ± 0.034 | 7.212 ± 1.409 | 0.287 ± 0.060 |
| wavelet_ml_haar_l3 | 0.679 ± 0.007 | 0.725 ± 0.023 | 7.667 ± 0.860 | 0.357 ± 0.038 |
| wavelet_ml_sym4_l1 | 0.664 ± 0.021 | 0.684 ± 0.027 | 6.379 ± 1.721 | 0.254 ± 0.025 |

## matched_fp

| config | lesion_recall@5fp | lesion_recall@10fp | lesion_recall@20fp | recall_small@5fp | recall_small@10fp | recall_small@20fp |
|---|---|---|---|---|---|---|
| baseline | 0.609 ± 0.009 | nan | nan | 0.272 ± 0.004 | nan | nan |
| wavelet_ml_db2_l2 | 0.586 | 0.684 ± 0.009 | nan | 0.263 | 0.390 ± 0.048 | nan |
| wavelet_ml_haar_l2 | 0.612 ± 0.022 | 0.613 | nan | 0.275 ± 0.025 | 0.251 | nan |
| wavelet_ml_haar_l3 | 0.637 | 0.692 ± 0.015 | nan | 0.301 | 0.382 ± 0.026 | nan |
| wavelet_ml_sym4_l1 | 0.568 ± 0.070 | 0.667 | nan | 0.229 ± 0.082 | 0.354 | nan |

## Paired vs baseline (Δ mean, Wilcoxon p)

| config | dice@0.5 | lesion_f1@0.5 | dice@val_thr |
|---|---|---|---|
| wavelet_ml_db2_l2 | -0.0032 (p=0.113) | -0.0140 (p=0.121) | -0.0035 (p=0.079) |
| wavelet_ml_haar_l2 | -0.0058 (p=0.085) | -0.0145 (p=0.074) | -0.0050 (p=0.025) |
| wavelet_ml_haar_l3 | -0.0005 (p=0.388) | -0.0103 (p=0.545) | -0.0008 (p=0.198) |
| wavelet_ml_sym4_l1 | -0.0191 (p=0.000) | -0.0180 (p=0.166) | -0.0161 (p=0.001) |
