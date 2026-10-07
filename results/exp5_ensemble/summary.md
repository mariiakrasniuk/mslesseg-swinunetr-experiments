# Seed ensembles + small-component removal (test set; k chosen on validation)

| config | setting | Dice | lesion recall | lesion precision | lesion F1 | FP/pt | small recall | Δ vs baseline (p) |
|---|---|---|---|---|---|---|---|---|
| baseline_r2 | single (mean of seeds) | 0.6893 | 0.676 | 0.771 | 0.721 | 7.97 | 0.342 |  |
| baseline_r2 | ensemble | 0.7000 | 0.672 | 0.807 | 0.734 | 6.41 | 0.325 |  |
| baseline_r2 | ensemble + remove <5 vox | 0.7000 | 0.664 | 0.818 | 0.733 | 5.86 | 0.318 |  |
| waveup_haar_learn_r2 | single (mean of seeds) | 0.6874 | 0.686 | 0.753 | 0.718 | 8.86 | 0.371 |  |
| waveup_haar_learn_r2 | ensemble | 0.6963 | 0.689 | 0.791 | 0.736 | 7.14 | 0.371 |  |
| waveup_haar_learn_r2 | ensemble + remove <5 vox | 0.6964 | 0.672 | 0.798 | 0.730 | 6.68 | 0.343 | -0.0036 (p=0.702) |
| waveup_haar_r2 | single (mean of seeds) | 0.6892 | 0.682 | 0.764 | 0.721 | 8.41 | 0.355 |  |
| waveup_haar_r2 | ensemble | 0.7011 | 0.683 | 0.782 | 0.729 | 7.55 | 0.350 |  |
| waveup_haar_r2 | ensemble + remove <10 vox | 0.7016 | 0.645 | 0.819 | 0.721 | 5.64 | 0.296 | +0.0015 (p=0.975) |
| waveup_rand_learn_r2 | single (mean of seeds) | 0.6832 | 0.681 | 0.763 | 0.719 | 8.42 | 0.357 |  |
| waveup_rand_learn_r2 | ensemble | 0.6905 | 0.681 | 0.790 | 0.731 | 7.23 | 0.357 |  |
| waveup_rand_learn_r2 | ensemble + remove <20 vox | 0.6895 | 0.596 | 0.845 | 0.699 | 4.32 | 0.243 | -0.0105 (p=0.187) |
