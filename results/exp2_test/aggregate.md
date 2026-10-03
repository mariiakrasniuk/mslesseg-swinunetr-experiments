# Multi-seed results (results/exp2_test)

## thr0.5

| config | dice | lesion_f1 | lesion_recall | lesion_precision | n_fp_lesions | hd95 | recall_small |
|---|---|---|---|---|---|---|---|
| baseline | 0.680 ± 0.010 | 0.685 ± 0.003 | 0.708 ± 0.015 | 0.721 ± 0.019 | 6.606 ± 0.936 | 21.931 ± 0.816 | 0.319 ± 0.020 |
| detail_skip_haar | 0.678 ± 0.001 | 0.670 ± 0.014 | 0.735 ± 0.025 | 0.676 ± 0.032 | 9.182 ± 1.614 | 23.743 ± 0.574 | 0.367 ± 0.027 |
| detail_skip_plain | 0.689 ± 0.005 | 0.683 ± 0.014 | 0.735 ± 0.023 | 0.692 ± 0.035 | 8.227 ± 1.772 | 22.261 ± 1.201 | 0.355 ± 0.024 |

## val_thr

| config | dice | lesion_f1 | fp_at_t_f1 | recall_small_at_t_f1 |
|---|---|---|---|---|
| baseline | 0.680 ± 0.009 | 0.710 ± 0.008 | 5.742 ± 0.329 | 0.294 ± 0.013 |
| detail_skip_haar | 0.678 ± 0.001 | 0.710 ± 0.002 | 7.500 ± 0.842 | 0.327 ± 0.022 |
| detail_skip_plain | 0.689 ± 0.005 | 0.719 ± 0.008 | 7.061 ± 1.591 | 0.326 ± 0.032 |

## matched_fp

| config | lesion_recall@5fp | lesion_recall@6fp | lesion_recall@7fp | lesion_recall@8fp | recall_small@5fp | recall_small@6fp | recall_small@7fp | recall_small@8fp |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.609 ± 0.009 | 0.632 ± 0.016 | 0.660 ± 0.018 | 0.674 ± 0.020 | 0.272 ± 0.004 | 0.301 ± 0.017 | 0.340 ± 0.026 | 0.356 ± 0.011 |
| detail_skip_haar | 0.591 | 0.622 ± 0.007 | 0.640 ± 0.001 | 0.661 ± 0.008 | 0.246 | 0.286 ± 0.003 | 0.314 ± 0.003 | 0.340 ± 0.010 |
| detail_skip_plain | 0.628 | 0.648 | 0.656 ± 0.015 | 0.672 ± 0.019 | 0.321 | 0.357 | 0.336 ± 0.057 | 0.354 ± 0.067 |

## Paired vs baseline (Δ mean, Wilcoxon p)

| config | dice@0.5 | lesion_f1@0.5 | dice@val_thr |
|---|---|---|---|
| detail_skip_haar | -0.0012 (p=0.503) | -0.0150 (p=0.371) | -0.0018 (p=0.406) |
| detail_skip_plain | +0.0095 (p=0.074) | -0.0014 (p=0.849) | +0.0090 (p=0.074) |
