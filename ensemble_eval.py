"""
ensemble_eval.py

Seed ensembling + small-component removal, from the probability maps saved by
`evaluate_lesions.py --save_probs`. CPU only — no model is loaded.

For every configuration (run name without _s<seed>):
  single    : each seed evaluated alone (mean over seeds) — the current numbers
  ensemble  : probability maps of all seeds averaged, then thresholded
  ens+pp    : ensemble, then predicted components smaller than k voxels removed;
              k is chosen per configuration on the VALIDATION split (max mean
              Dice) and applied unchanged to the test split

Usage:
    python ensemble_eval.py \
        --val_dirs  results/seeds_val/probs  results/exp2_val/probs  results/exp3_val/probs \
        --test_dirs results/seeds_test/probs results/exp2_test/probs results/exp3_test/probs

Outputs (results/ensemble/): per_case.csv, summary.csv, summary.md
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from scipy.stats import wilcoxon

from lesion_metrics import STRUCT_26, label_lesions

parser = argparse.ArgumentParser()
parser.add_argument("--val_dirs", nargs="+", required=True)
parser.add_argument("--test_dirs", nargs="+", required=True)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--min_sizes", type=str, default="0,5,10,20,30,50,80",
                    help="Candidate minimum component sizes (voxels) for post-processing")
parser.add_argument("--min_lesion_size", type=int, default=3,
                    help="Lesion definition for lesion metrics (same as evaluate_lesions.py)")
parser.add_argument("--small_max", type=float, default=40.0,
                    help="Upper volume (mm^3) of the 'small' lesion bin")
parser.add_argument("--baseline", type=str, default="baseline")
parser.add_argument("--out_dir", type=str, default="results/ensemble")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

MIN_SIZES = [int(k) for k in args.min_sizes.split(",")]


def config_of(run):
    return re.sub(r"_s\d+$", "", run)


def collect(dirs):
    """{config: {run: probs_dir}} — first occurrence of a run name wins."""
    runs = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[missing] {d}")
            continue
        for run in sorted(os.listdir(d)):
            if os.path.isdir(os.path.join(d, run)) and run not in runs:
                runs[run] = os.path.join(d, run)
    out = defaultdict(dict)
    for run, path in runs.items():
        out[config_of(run)][run] = path
    return out


def remove_small(mask, k):
    if k <= 1 or not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=STRUCT_26)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= k
    keep[0] = False
    return keep[lab]


def metrics(pred, gt, gt_lab, n_gt, gt_small):
    tp, n_p, n_g = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
    dice = 2 * tp / (n_p + n_g) if n_p + n_g else 1.0
    pr_lab, n_pr = label_lesions(pred, args.min_lesion_size)
    gt_det = np.bincount(gt_lab[pr_lab > 0], minlength=n_gt + 1)[1:] > 0
    pr_tp = np.bincount(pr_lab[gt_lab > 0], minlength=n_pr + 1)[1:] > 0
    return {"dice": dice, "n_gt": n_gt, "n_det": int(gt_det.sum()), "n_pred": n_pr,
            "n_fp": int((~pr_tp).sum()), "gt_small": int(gt_small.sum()),
            "det_small": int(gt_det[gt_small].sum())}


def eval_case(job):
    split, config, case, paths = job
    probs, gt = [], None
    for p in paths:
        d = np.load(p)
        probs.append(d["prob"].astype(np.float32))
        gt = d["gt"].astype(bool)
    # crop to everything that can be foreground (exact for any threshold >= 0.01)
    roi = gt.copy()
    for pr in probs:
        roi |= pr > 0.01
    if roi.any():
        idx = np.argwhere(roi)
        sl = tuple(slice(max(a - 1, 0), b + 2) for a, b in zip(idx.min(0), idx.max(0)))
        probs, gt = [pr[sl] for pr in probs], gt[sl]
    gt_lab, n_gt = label_lesions(gt, args.min_lesion_size)
    gt_small = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)[1:] < args.small_max

    rows = []
    single = [metrics(pr > args.threshold, gt, gt_lab, n_gt, gt_small) for pr in probs]
    rows.append({"split": split, "config": config, "case": case, "mode": "single", "k": 0,
                 **{key: float(np.mean([m[key] for m in single])) for key in single[0]}})
    ens = np.mean(probs, axis=0) > args.threshold
    for k in MIN_SIZES:
        rows.append({"split": split, "config": config, "case": case, "mode": "ensemble", "k": k,
                     **metrics(remove_small(ens, k), gt, gt_lab, n_gt, gt_small)})
    return rows


def aggregate(rows):
    n = len(rows)
    tot = lambda key: sum(r[key] for r in rows)
    rec = tot("n_det") / tot("n_gt") if tot("n_gt") else float("nan")
    prec = (tot("n_pred") - tot("n_fp")) / tot("n_pred") if tot("n_pred") else float("nan")
    return {"dice": float(np.mean([r["dice"] for r in rows])),
            "lesion_recall": rec, "lesion_precision": prec,
            "lesion_f1": 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0,
            "fp_per_case": tot("n_fp") / n,
            "recall_small": tot("det_small") / tot("gt_small") if tot("gt_small") else float("nan")}


if __name__ == "__main__":
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = []
    for split, dirs in (("val", args.val_dirs), ("test", args.test_dirs)):
        for config, runs in collect(dirs).items():
            cases = set.intersection(*[{f[:-4] for f in os.listdir(p) if f.endswith(".npz")}
                                       for p in runs.values()])
            for case in sorted(cases):
                jobs.append((split, config, case,
                             [os.path.join(p, f"{case}.npz") for p in runs.values()]))
    print(f"{len(jobs)} (split, config, case) jobs")

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_case, jobs), 1):
            rows += r
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")

    with open(os.path.join(args.out_dir, "per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def sel(split, config, mode, k):
        return [r for r in rows if r["split"] == split and r["config"] == config
                and r["mode"] == mode and r["k"] == k]

    configs = sorted({r["config"] for r in rows if r["split"] == "test"},
                     key=lambda c: (c != args.baseline, c))
    # k chosen on validation, per configuration
    best_k = {}
    for c in configs:
        cand = [(aggregate(sel("val", c, "ensemble", k))["dice"], k) for k in MIN_SIZES
                if sel("val", c, "ensemble", k)]
        best_k[c] = max(cand)[1] if cand else 0

    summary = []
    for c in configs:
        for label, mode, k in (("single (mean of seeds)", "single", 0),
                               ("ensemble", "ensemble", 0),
                               (f"ensemble + remove <{best_k[c]} vox", "ensemble", best_k[c])):
            rs = sel("test", c, mode, k)
            if rs:
                summary.append({"config": c, "setting": label, "k": k, "n_cases": len(rs), **aggregate(rs)})

    # paired test: each config's ens+pp vs the baseline's ens+pp (per patient)
    if args.baseline in best_k:
        base = {r["case"]: r["dice"] for r in sel("test", args.baseline, "ensemble", best_k[args.baseline])}
        for s in summary:
            if s["config"] != args.baseline and s["setting"].startswith("ensemble +"):
                cur = {r["case"]: r["dice"] for r in sel("test", s["config"], "ensemble", s["k"])}
                d = np.array([cur[k] - base[k] for k in sorted(set(cur) & set(base))])
                s["delta_vs_baseline"] = float(d.mean())
                s["p_wilcoxon"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")

    fields = ["config", "setting", "k", "n_cases", "dice", "lesion_recall", "lesion_precision",
              "lesion_f1", "fp_per_case", "recall_small", "delta_vs_baseline", "p_wilcoxon"]
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    hdr = (f"{'config':<24}{'setting':<28}{'dice':>7}{'L-rec':>7}{'L-prec':>7}{'L-F1':>7}"
           f"{'FP/pt':>7}{'rec_S':>7}{'d_base':>9}{'p':>7}")
    print("\nTEST set (k chosen on validation)\n" + hdr + "\n" + "-" * len(hdr))
    md = ["# Seed ensembles + small-component removal (test set; k chosen on validation)", "",
          "| config | setting | Dice | lesion recall | lesion precision | lesion F1 | FP/pt | small recall | Δ vs baseline (p) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        dp = (f"{s['delta_vs_baseline']:+.4f}", f"{s['p_wilcoxon']:.3f}") if "p_wilcoxon" in s else ("", "")
        print(f"{s['config']:<24}{s['setting']:<28}{s['dice']:>7.4f}{s['lesion_recall']:>7.3f}"
              f"{s['lesion_precision']:>7.3f}{s['lesion_f1']:>7.3f}{s['fp_per_case']:>7.2f}"
              f"{s['recall_small']:>7.3f}{dp[0]:>9}{dp[1]:>7}")
        md.append(f"| {s['config']} | {s['setting']} | {s['dice']:.4f} | {s['lesion_recall']:.3f} | "
                  f"{s['lesion_precision']:.3f} | {s['lesion_f1']:.3f} | {s['fp_per_case']:.2f} | "
                  f"{s['recall_small']:.3f} | " + (f"{dp[0]} (p={dp[1]})" if dp[0] else "") + " |")
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWritten: {args.out_dir}/summary.csv, summary.md, per_case.csv")
