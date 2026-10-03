# Seed ensembles + small-component removal (test set; k chosen on validation)

| config | setting | Dice | lesion recall | lesion precision | lesion F1 | FP/pt | small recall | Δ vs baseline (p) |
|---|---|---|---|---|---|---|---|---|
| baseline | single (mean of seeds) | 0.6796 | 0.646 | 0.801 | 0.715 | 6.61 | 0.319 |  |
| baseline | ensemble | 0.6841 | 0.642 | 0.821 | 0.721 | 5.73 | 0.314 |  |
| baseline | ensemble + remove <5 vox | 0.6841 | 0.628 | 0.837 | 0.718 | 5.00 | 0.296 |  |
| baseline_dicebce | single (mean of seeds) | 0.6828 | 0.660 | 0.770 | 0.711 | 8.21 | 0.330 |  |
| baseline_dicebce | ensemble | 0.6930 | 0.659 | 0.802 | 0.723 | 6.73 | 0.325 |  |
| baseline_dicebce | ensemble + remove <0 vox | 0.6930 | 0.659 | 0.802 | 0.723 | 6.73 | 0.325 | +0.0089 (p=0.424) |
| baseline_dicebce_hf1 | single (mean of seeds) | 0.6782 | 0.650 | 0.795 | 0.715 | 6.61 | 0.317 |  |
| baseline_dicebce_hf1 | ensemble | 0.6923 | 0.641 | 0.817 | 0.718 | 5.64 | 0.296 |  |
| baseline_dicebce_hf1 | ensemble + remove <5 vox | 0.6923 | 0.636 | 0.830 | 0.721 | 5.09 | 0.289 | +0.0082 (p=1.000) |
| detail_skip_haar | single (mean of seeds) | 0.6783 | 0.676 | 0.751 | 0.711 | 9.18 | 0.367 |  |
| detail_skip_haar | ensemble | 0.6885 | 0.676 | 0.780 | 0.724 | 7.82 | 0.361 |  |
| detail_skip_haar | ensemble + remove <5 vox | 0.6884 | 0.654 | 0.795 | 0.717 | 6.91 | 0.329 | +0.0043 (p=0.924) |
| detail_skip_plain | single (mean of seeds) | 0.6891 | 0.675 | 0.773 | 0.720 | 8.23 | 0.355 |  |
| detail_skip_plain | ensemble | 0.6947 | 0.674 | 0.796 | 0.730 | 7.18 | 0.339 |  |
| detail_skip_plain | ensemble + remove <5 vox | 0.6948 | 0.663 | 0.805 | 0.727 | 6.68 | 0.329 | +0.0108 (p=0.222) |
| wavelet_ml_db2_l2 | single (mean of seeds) | 0.6764 | 0.652 | 0.772 | 0.707 | 8.23 | 0.348 |  |
| wavelet_ml_db2_l2 | ensemble | 0.6860 | 0.646 | 0.789 | 0.710 | 7.41 | 0.329 |  |
| wavelet_ml_db2_l2 | ensemble + remove <5 vox | 0.6861 | 0.638 | 0.803 | 0.711 | 6.68 | 0.311 | +0.0021 (p=0.371) |
| wavelet_ml_haar_l2 | single (mean of seeds) | 0.6738 | 0.657 | 0.765 | 0.707 | 8.32 | 0.325 |  |
| wavelet_ml_haar_l2 | ensemble | 0.6840 | 0.655 | 0.812 | 0.725 | 6.27 | 0.314 |  |
| wavelet_ml_haar_l2 | ensemble + remove <10 vox | 0.6835 | 0.613 | 0.839 | 0.708 | 4.82 | 0.250 | -0.0006 (p=0.176) |
| wavelet_ml_haar_l3 | single (mean of seeds) | 0.6791 | 0.702 | 0.745 | 0.723 | 9.97 | 0.392 |  |
| wavelet_ml_haar_l3 | ensemble | 0.6886 | 0.699 | 0.770 | 0.733 | 8.77 | 0.386 |  |
| wavelet_ml_haar_l3 | ensemble + remove <20 vox | 0.6889 | 0.626 | 0.836 | 0.716 | 5.05 | 0.264 | +0.0048 (p=0.750) |
| wavelet_ml_sym4_l1 | single (mean of seeds) | 0.6604 | 0.623 | 0.777 | 0.692 | 7.56 | 0.292 |  |
| wavelet_ml_sym4_l1 | ensemble | 0.6771 | 0.617 | 0.802 | 0.697 | 6.59 | 0.271 |  |
| wavelet_ml_sym4_l1 | ensemble + remove <10 vox | 0.6772 | 0.584 | 0.838 | 0.688 | 4.82 | 0.239 | -0.0069 (p=0.025) |
