# Seed ensembles + small-component removal (test set; k chosen on validation)

| config | setting | Dice | lesion recall | lesion precision | lesion F1 | FP/pt | small recall | Δ vs baseline (p) |
|---|---|---|---|---|---|---|---|---|
| baseline_r2 | single (mean of seeds) | 0.6893 | 0.676 | 0.771 | 0.721 | 7.97 | 0.342 |  |
| baseline_r2 | ensemble | 0.7000 | 0.672 | 0.807 | 0.734 | 6.41 | 0.325 |  |
| baseline_r2 | ensemble + remove <5 vox | 0.7000 | 0.664 | 0.818 | 0.733 | 5.86 | 0.318 |  |
| baseline_r2_hf1 | single (mean of seeds) | 0.6908 | 0.677 | 0.766 | 0.719 | 8.18 | 0.344 |  |
| baseline_r2_hf1 | ensemble | 0.7023 | 0.674 | 0.794 | 0.729 | 6.91 | 0.332 |  |
| baseline_r2_hf1 | ensemble + remove <20 vox | 0.7020 | 0.612 | 0.845 | 0.710 | 4.36 | 0.239 | +0.0020 (p=0.924) |
| baseline_r2_hf1db2 | single (mean of seeds) | 0.6828 | 0.643 | 0.802 | 0.714 | 6.23 | 0.324 |  |
| baseline_r2_hf1db2 | ensemble | 0.6985 | 0.640 | 0.834 | 0.724 | 5.00 | 0.318 |  |
| baseline_r2_hf1db2 | ensemble + remove <10 vox | 0.6982 | 0.624 | 0.851 | 0.720 | 4.27 | 0.293 | -0.0018 (p=0.799) |
| baseline_r2_hf1sym4 | single (mean of seeds) | 0.6915 | 0.646 | 0.807 | 0.717 | 5.97 | 0.308 |  |
| baseline_r2_hf1sym4 | ensemble | 0.7016 | 0.648 | 0.833 | 0.729 | 5.05 | 0.307 |  |
| baseline_r2_hf1sym4 | ensemble + remove <5 vox | 0.7015 | 0.640 | 0.835 | 0.725 | 4.91 | 0.304 | +0.0015 (p=0.566) |
