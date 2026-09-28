"""
threshold_sweep.py

Threshold sweep / FROC analysis on probability maps saved by
`evaluate_lesions.py --save_probs`. CPU only — no model is loaded.

For every run and every threshold it computes voxel Dice and lesion-level
detection counts (overall and per lesion-size bin), using the same lesion
definition as lesion_metrics.py (26-connected, components < min size ignored).

Typical use:
    # 1. choose thresholds on validation
    python threshold_sweep.py --split val
    # 2. apply them to test + FROC comparison at matched FP rates
    python threshold_sweep.py --split test \
        --select_from results/eval_val/sweep/selected_thresholds.csv

Outputs (results/eval_<split>/sweep/):
    sweep_per_case.csv        run × case × threshold
    sweep_summary.csv         run × threshold (means / pooled lesion counts)
    selected_thresholds.csv   per run: threshold maximising mean Dice and lesion F1
    recall_at_fp.csv          per run: lesion recall at fixed FP-lesions-per-case
    froc.png                  FROC curves, one panel per wavelet family
"""

import argparse
import csv
import glob
import math
import os
from multiprocessing import Pool

import numpy as np

from lesion_metrics import label_lesions

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
parser.add_argument("--probs_dir", type=str, default=None,
                    help="Default: results/eval_<split>/probs")
parser.add_argument("--thresholds", type=str, default=None,
                    help="Comma-separated; default 0.05..0.95 step 0.05")
parser.add_argument("--min_lesion_size", type=int, default=3)
parser.add_argument("--size_bins", type=str, default="40,115",
                    help="Lesion volume bin edges in mm^3: small < a <= medium < b <= large")
parser.add_argument("--fp_levels", type=str, default="5,10,20",
                    help="FP lesions per case at which lesion recall is interpolated")
parser.add_argument("--select_from", type=str, default=None,
                    help="selected_thresholds.csv from a val sweep, applied to this split")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

PROBS_DIR = args.probs_dir or os.path.join("results", f"eval_{args.split}", "probs")
OUT_DIR   = os.path.join(os.path.dirname(PROBS_DIR.rstrip("/\\")), "sweep")
THRESHOLDS = ([float(t) for t in args.thresholds.split(",")] if args.thresholds
              else [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)])
EDGES     = [float(v) for v in args.size_bins.split(",")]
BIN_NAMES = ["small", "medium", "large"]
FP_LEVELS = [float(v) for v in args.fp_levels.split(",")]
VOXEL_VOLUME = 1.0   # 1 mm isotropic after transforms.py resampling


def bin_index(volumes):
    return np.digitize(volumes, EDGES)          # 0 small, 1 medium, 2 large


# ------------------
# Per-case sweep
# ------------------
def sweep_case(job):
    run, path = job
    case = os.path.splitext(os.path.basename(path))[0]
    d = np.load(path)
    prob, gt = d["prob"].astype(np.float32), d["gt"].astype(bool)

    # Crop to everything that can matter at the lowest threshold (exact)
    roi = (prob > min(THRESHOLDS)) | gt
    if roi.any():
        idx = np.argwhere(roi)
        lo, hi = np.maximum(idx.min(0) - 1, 0), idx.max(0) + 2
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        prob, gt = prob[sl], gt[sl]

    gt_lab, n_gt = label_lesions(gt, args.min_lesion_size)
    gt_bins = bin_index(np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:] * VOXEL_VOLUME)
    n_gt_vox = int(gt.sum())

    rows = []
    for t in THRESHOLDS:
        pred = prob > t
        tp_vox, n_pred_vox = int((pred & gt).sum()), int(pred.sum())
        denom = n_pred_vox + n_gt_vox
        dice = 2 * tp_vox / denom if denom else 1.0

        pr_lab, n_pr = label_lesions(pred, args.min_lesion_size)
        gt_det = np.bincount(gt_lab[pr_lab > 0], minlength=n_gt + 1)[1:] > 0
        pr_tp  = np.bincount(pr_lab[gt_lab > 0], minlength=n_pr + 1)[1:] > 0
        pr_bins = bin_index(np.bincount(pr_lab.ravel(), minlength=n_pr + 1)[1:] * VOXEL_VOLUME)

        row = {"run": run, "case": case, "threshold": t, "dice": dice,
               "n_gt": n_gt, "n_det": int(gt_det.sum()),
               "n_pred": n_pr, "n_tp": int(pr_tp.sum()), "n_fp": int((~pr_tp).sum())}
        for b, name in enumerate(BIN_NAMES):
            row[f"gt_{name}"]  = int((gt_bins == b).sum())
            row[f"det_{name}"] = int(gt_det[gt_bins == b].sum())
            row[f"fp_{name}"]  = int((~pr_tp[pr_bins == b]).sum())
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows):
    """Aggregate per-case rows of one (run, threshold)."""
    n_cases = len(rows)
    tot = lambda k: sum(r[k] for r in rows)
    recall    = tot("n_det") / tot("n_gt") if tot("n_gt") else float("nan")
    precision = tot("n_tp") / tot("n_pred") if tot("n_pred") else float("nan")
    f1 = (2 * recall * precision / (recall + precision)
          if recall + precision > 0 else 0.0)
    s = {"mean_dice": float(np.mean([r["dice"] for r in rows])),
         "lesion_recall": recall, "lesion_precision": precision, "lesion_f1": f1,
         "fp_per_case": tot("n_fp") / n_cases}
    for name in BIN_NAMES:
        g = tot(f"gt_{name}")
        s[f"recall_{name}"] = tot(f"det_{name}") / g if g else float("nan")
        s[f"fp_{name}_per_case"] = tot(f"fp_{name}") / n_cases
    return s


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = sorted(d for d in os.listdir(PROBS_DIR) if os.path.isdir(os.path.join(PROBS_DIR, d)))
    jobs = [(run, p) for run in runs
            for p in sorted(glob.glob(os.path.join(PROBS_DIR, run, "*.npz")))]
    print(f"{len(runs)} runs, {len(jobs)} cases, {len(THRESHOLDS)} thresholds, "
          f"size bins {EDGES} mm^3  ->  {OUT_DIR}/")

    per_case = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(sweep_case, jobs), 1):
            per_case += rows
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} cases done")
    per_case.sort(key=lambda r: (r["run"], r["case"], r["threshold"]))
    write_csv(os.path.join(OUT_DIR, "sweep_per_case.csv"), per_case)

    # ---- run × threshold summary ----
    summary = []
    for run in runs:
        for t in THRESHOLDS:
            rows = [r for r in per_case if r["run"] == run and r["threshold"] == t]
            summary.append({"run": run, "threshold": t, "n_cases": len(rows), **summarise(rows)})
    write_csv(os.path.join(OUT_DIR, "sweep_summary.csv"), summary)

    # ---- best thresholds on this split ----
    selected = []
    for run in runs:
        rs = [s for s in summary if s["run"] == run]
        best_dice = max(rs, key=lambda s: s["mean_dice"])
        best_f1   = max(rs, key=lambda s: s["lesion_f1"])
        selected.append({"run": run,
                         "t_dice": best_dice["threshold"], "dice_at_t_dice": best_dice["mean_dice"],
                         "t_f1": best_f1["threshold"], "f1_at_t_f1": best_f1["lesion_f1"]})
    write_csv(os.path.join(OUT_DIR, "selected_thresholds.csv"), selected)

    # ---- lesion recall at matched FP rates (FROC interpolation) ----
    at_fp = []
    for run in runs:
        rs = sorted((s for s in summary if s["run"] == run), key=lambda s: s["fp_per_case"])
        fps = np.array([s["fp_per_case"] for s in rs])
        row = {"run": run}
        for key in ("lesion_recall", "recall_small"):
            ys = np.array([s[key] for s in rs])
            for lvl in FP_LEVELS:
                inside = fps.min() <= lvl <= fps.max()
                row[f"{key}@{lvl:g}fp"] = float(np.interp(lvl, fps, ys)) if inside else float("nan")
        at_fp.append(row)
    write_csv(os.path.join(OUT_DIR, "recall_at_fp.csv"), at_fp)

    # ---- print ----
    def at(run, t):
        return next(s for s in summary if s["run"] == run and abs(s["threshold"] - t) < 1e-9)

    print(f"\nAt threshold 0.5 (should match evaluate_lesions.py):")
    print(f"{'run':<22}{'dice':>7}{'L-rec':>7}{'L-prec':>7}{'L-F1':>7}{'FP/pt':>7}"
          f"{'rec_S':>7}{'rec_M':>7}{'rec_L':>7}")
    for run in runs:
        if 0.5 in THRESHOLDS:
            s = at(run, 0.5)
            print(f"{run:<22}{s['mean_dice']:>7.4f}{s['lesion_recall']:>7.3f}"
                  f"{s['lesion_precision']:>7.3f}{s['lesion_f1']:>7.3f}{s['fp_per_case']:>7.2f}"
                  f"{s['recall_small']:>7.3f}{s['recall_medium']:>7.3f}{s['recall_large']:>7.3f}")

    cols = [f"{k}@{l:g}fp" for k in ("lesion_recall", "recall_small") for l in FP_LEVELS]
    print("\nLesion recall at matched FP lesions per case (all lesions | small lesions):")
    print(f"{'run':<22}" + "".join(f"{c.replace('lesion_recall', 'all').replace('recall_small', 'S'):>12}"
                                    for c in cols))
    for r in at_fp:
        print(f"{r['run']:<22}" + "".join(f"{r[c]:>12.3f}" for c in cols))

    if args.select_from:
        chosen = {r["run"]: r for r in csv.DictReader(open(args.select_from))}
        print(f"\nThresholds selected on validation ({args.select_from}) applied to {args.split}:")
        print(f"{'run':<22}{'t_dice':>7}{'dice':>7}{'L-F1':>7}{'FP/pt':>7}"
              f"{'t_f1':>7}{'dice':>7}{'L-F1':>7}{'FP/pt':>7}{'rec_S':>7}")
        applied = []
        for run in runs:
            if run not in chosen:
                continue
            td, tf = float(chosen[run]["t_dice"]), float(chosen[run]["t_f1"])
            sd, sf = at(run, td), at(run, tf)
            applied.append({"run": run, "t_dice": td, "dice": sd["mean_dice"],
                            "lesion_f1_at_t_dice": sd["lesion_f1"], "fp_at_t_dice": sd["fp_per_case"],
                            "t_f1": tf, "dice_at_t_f1": sf["mean_dice"], "lesion_f1": sf["lesion_f1"],
                            "fp_at_t_f1": sf["fp_per_case"], "recall_small_at_t_f1": sf["recall_small"]})
            print(f"{run:<22}{td:>7.2f}{sd['mean_dice']:>7.4f}{sd['lesion_f1']:>7.3f}"
                  f"{sd['fp_per_case']:>7.2f}{tf:>7.2f}{sf['mean_dice']:>7.4f}{sf['lesion_f1']:>7.3f}"
                  f"{sf['fp_per_case']:>7.2f}{sf['recall_small']:>7.3f}")
        write_csv(os.path.join(OUT_DIR, "val_selected_applied.csv"), applied)

    # ---- FROC figure: one panel per family, baseline as grey reference ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SERIES   = ["#2a78d6", "#eb6834", "#1baf7a"]    # validated categorical slots 1-3
    BASE_CLR = "#52514e"
    families = [("Single-level Haar", ["wavelet_a"])] + [
        (f"Multi-level {w}", [f"wavelet_ml_{w}_l{l}" for l in (1, 2, 3)]) for w in ("haar", "db2", "sym4")]
    families = [(title, [r for r in rs if r in runs]) for title, rs in families]
    families = [(title, rs) for title, rs in families if rs]

    if families:
        fig, axes = plt.subplots(2, len(families), figsize=(4.2 * len(families), 7.2),
                                 sharex=True, sharey="row", squeeze=False)
        for row, (key, ylabel) in enumerate([("lesion_recall", "Lesion recall (all)"),
                                             ("recall_small", "Lesion recall (small)")]):
            for col, (title, fam_runs) in enumerate(families):
                ax = axes[row][col]
                plotted = (["baseline"] if "baseline" in runs else []) + fam_runs
                for i, run in enumerate(plotted):
                    rs = sorted((s for s in summary if s["run"] == run), key=lambda s: s["threshold"])
                    is_base = run == "baseline"
                    color = BASE_CLR if is_base else SERIES[(i - ("baseline" in runs)) % 3]
                    label = "baseline" if is_base else run.replace("wavelet_ml_", "").replace("_", " ")
                    ax.plot([s["fp_per_case"] for s in rs], [s[key] for s in rs],
                            color=color, lw=2, ls="--" if is_base else "-", label=label)
                    if 0.5 in THRESHOLDS:
                        s = at(run, 0.5)
                        ax.plot(s["fp_per_case"], s[key], "o", ms=8, color=color,
                                markeredgecolor="white", markeredgewidth=2, zorder=3)
                ax.set_xlim(0, 60)
                ax.set_ylim(0, 1)
                ax.grid(True, color="#e5e4e0", lw=0.8)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=11)
                    ax.legend(frameon=False, fontsize=9, loc="lower right")
                if row == 1:
                    ax.set_xlabel("False-positive lesions per case")
                if col == 0:
                    ax.set_ylabel(ylabel)
        fig.suptitle(f"FROC on {args.split} set (dots: threshold 0.5)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "froc.png"), dpi=150)
        print(f"\nFigure: {os.path.join(OUT_DIR, 'froc.png')}")
