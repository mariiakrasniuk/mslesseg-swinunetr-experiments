"""
aggregate_seeds.py

Averages multi-seed results per model configuration and tests each
configuration against the baseline. CPU only — reads the CSVs written by
evaluate_lesions.py and threshold_sweep.py.

Usage:
    python aggregate_seeds.py --results results/seeds_test --val_results results/seeds_val

Reports, per configuration (run name without the _s<seed> suffix):
  1. Threshold 0.5            : mean ± sd over seeds (evaluate_lesions summary.csv)
  2. Val-selected threshold   : mean ± sd over seeds (sweep/val_selected_applied.csv)
  3. Matched FP rate          : lesion recall at fixed FP lesions/case (sweep/recall_at_fp.csv)
  4. Paired test vs baseline  : per-patient metric averaged over seeds, Wilcoxon
                                signed-rank across patients

Outputs: <results>/aggregate.csv and <results>/aggregate.md
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, default="results/seeds_test")
parser.add_argument("--val_results", type=str, default="results/seeds_val",
                    help="Used for the val-selected thresholds in the paired test")
parser.add_argument("--baseline", type=str, default="baseline")
args = parser.parse_args()


def read(path):
    if not os.path.exists(path):
        print(f"[missing] {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def config_of(run):
    return re.sub(r"_s\d+$", "", run)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean_sd(values):
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")), len(v)


def by_config(rows, keys):
    """{config: {key: [value per seed]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in keys:
            if k in r:
                out[config_of(r["run"])][k].append(num(r[k]))
    return out


summary  = read(os.path.join(args.results, "summary.csv"))
per_pat  = read(os.path.join(args.results, "per_patient.csv"))
applied  = read(os.path.join(args.results, "sweep", "val_selected_applied.csv"))
at_fp    = read(os.path.join(args.results, "sweep", "recall_at_fp.csv"))
sweep_pc = read(os.path.join(args.results, "sweep", "sweep_per_case.csv"))
val_sel  = read(os.path.join(args.val_results, "sweep", "selected_thresholds.csv"))

configs = sorted({config_of(r["run"]) for r in summary},
                 key=lambda c: (c != args.baseline, c))
rows_out = []

# ------------------
# 1-3. Mean ± sd over seeds
# ------------------
sections = [
    ("thr0.5", summary, ["dice", "lesion_f1", "lesion_recall", "lesion_precision",
                         "n_fp_lesions", "hd95", "recall_small"]),
    ("val_thr", applied, ["dice", "lesion_f1", "fp_at_t_f1", "recall_small_at_t_f1"]),
    ("matched_fp", at_fp, [k for k in (at_fp[0] if at_fp else {}) if k != "run"]),
]
for tag, rows, keys in sections:
    if not rows:
        continue
    agg = by_config(rows, keys)
    print(f"\n=== {tag}: mean ± sd over seeds ===")
    print(f"{'config':<22}{'n':>3}" + "".join(f"{k[:16]:>20}" for k in keys))
    for c in configs:
        cells, n_seeds = [], 0
        for k in keys:
            m, s, n = mean_sd(agg[c][k])
            n_seeds = max(n_seeds, n)
            cells.append(f"{m:.3f} +/- {s:.3f}" if not math.isnan(s) else f"{m:.3f}")
            rows_out.append({"section": tag, "config": c, "metric": k,
                             "mean": m, "sd": s, "n_seeds": n})
        print(f"{c:<22}{n_seeds:>3}" + "".join(f"{x:>20}" for x in cells))

# ------------------
# 4. Paired test vs baseline (per patient, averaged over seeds)
# ------------------
def per_patient_means(rows, key, select=None):
    """{config: {case: mean over seeds}}; `select(row)` filters rows."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if select is None or select(r):
            acc[config_of(r["run"])][r["case"]].append(num(r[key]))
    return {c: {case: float(np.nanmean(v)) for case, v in d.items()} for c, d in acc.items()}


tests = [("dice@0.5", per_patient_means(per_pat, "dice")),
         ("lesion_f1@0.5", per_patient_means(per_pat, "lesion_f1"))]
if sweep_pc and val_sel:
    t_dice = {r["run"]: float(r["t_dice"]) for r in val_sel}
    tests.append(("dice@val_thr", per_patient_means(
        sweep_pc, "dice",
        select=lambda r: r["run"] in t_dice and abs(float(r["threshold"]) - t_dice[r["run"]]) < 1e-9)))

print(f"\n=== Paired Wilcoxon signed-rank vs {args.baseline} (patients; seeds averaged) ===")
print(f"{'config':<22}" + "".join(f"{name:>26}" for name, _ in tests))
for c in configs:
    if c == args.baseline:
        continue
    cells = []
    for name, d in tests:
        if args.baseline not in d or c not in d:
            cells.append("n/a")
            continue
        cases = sorted(set(d[c]) & set(d[args.baseline]))
        diff = np.array([d[c][k] - d[args.baseline][k] for k in cases])
        diff = diff[~np.isnan(diff)]
        p = wilcoxon(diff).pvalue if len(diff) and np.any(diff != 0) else float("nan")
        cells.append(f"{diff.mean():+.4f} (p={p:.3f})")
        rows_out.append({"section": "paired_vs_baseline", "config": c, "metric": name,
                         "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                         "n_seeds": len(diff), "p_wilcoxon": p})
    print(f"{c:<22}" + "".join(f"{x:>26}" for x in cells))
print("(difference = config - baseline; n_seeds column holds the number of patients here)")

# ------------------
# Write
# ------------------
out_csv = os.path.join(args.results, "aggregate.csv")
fields = ["section", "config", "metric", "mean", "sd", "n_seeds", "p_wilcoxon"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

md = [f"# Multi-seed results ({args.results})", ""]
for tag in ("thr0.5", "val_thr", "matched_fp"):
    rs = [r for r in rows_out if r["section"] == tag]
    if not rs:
        continue
    metrics = list(dict.fromkeys(r["metric"] for r in rs))
    md += [f"## {tag}", "", "| config | " + " | ".join(metrics) + " |",
           "|---|" + "---|" * len(metrics)]
    for c in configs:
        cells = []
        for m in metrics:
            r = next((r for r in rs if r["config"] == c and r["metric"] == m), None)
            cells.append("" if r is None else
                         (f"{r['mean']:.3f} ± {r['sd']:.3f}" if not math.isnan(r["sd"])
                          else f"{r['mean']:.3f}"))
        md.append(f"| {c} | " + " | ".join(cells) + " |")
    md.append("")
rs = [r for r in rows_out if r["section"] == "paired_vs_baseline"]
if rs:
    metrics = list(dict.fromkeys(r["metric"] for r in rs))
    md += [f"## Paired vs {args.baseline} (Δ mean, Wilcoxon p)", "",
           "| config | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for c in configs:
        if c == args.baseline:
            continue
        cells = []
        for m in metrics:
            r = next((r for r in rs if r["config"] == c and r["metric"] == m), None)
            cells.append("" if r is None else f"{r['mean']:+.4f} (p={r['p_wilcoxon']:.3f})")
        md.append(f"| {c} | " + " | ".join(cells) + " |")
with open(os.path.join(args.results, "aggregate.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print(f"\nWritten: {out_csv}, {os.path.join(args.results, 'aggregate.md')}")
