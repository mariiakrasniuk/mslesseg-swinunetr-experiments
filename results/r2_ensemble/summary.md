# Seed ensembles + small-component removal (test set; k chosen on validation)

| config | setting | Dice | lesion recall | lesion precision | lesion F1 | FP/pt | small recall | Δ vs baseline (p) |
|---|---|---|---|---|---|---|---|---|
| baseline | single (mean of seeds) | 0.6796 | 0.646 | 0.801 | 0.715 | 6.61 | 0.319 |  |
| baseline | ensemble | 0.6841 | 0.642 | 0.821 | 0.721 | 5.73 | 0.314 |  |
| baseline | ensemble + remove <5 vox | 0.6841 | 0.628 | 0.837 | 0.718 | 5.00 | 0.296 |  |
| baseline_r2 | single (mean of seeds) | 0.6893 | 0.676 | 0.771 | 0.721 | 7.97 | 0.342 |  |
| baseline_r2 | ensemble | 0.7000 | 0.672 | 0.807 | 0.734 | 6.41 | 0.325 |  |
| baseline_r2 | ensemble + remove <5 vox | 0.7000 | 0.664 | 0.818 | 0.733 | 5.86 | 0.318 | +0.0159 (p=0.025) |
| baseline_r2_hf1 | single (mean of seeds) | 0.6908 | 0.677 | 0.766 | 0.719 | 8.18 | 0.344 |  |
| baseline_r2_hf1 | ensemble | 0.7023 | 0.674 | 0.794 | 0.729 | 6.91 | 0.332 |  |
| baseline_r2_hf1 | ensemble + remove <20 vox | 0.7020 | 0.612 | 0.845 | 0.710 | 4.36 | 0.239 | +0.0179 (p=0.003) |
