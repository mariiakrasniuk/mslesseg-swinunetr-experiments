"""
evaluate_lesions.py

Re-evaluates trained checkpoints with thresholded voxel-level, boundary and
lesion-level metrics (see lesion_metrics.py).

It also recomputes the Dice exactly as train.py / final_test_evaluation.py do
(`rp2_dice`: MONAI DiceMetric on sigmoid probabilities). With a 1-channel
input and MONAI 1.5, DiceMetric does not threshold: it counts only voxels
whose probability is exactly 1.0 in float32 (logit > ~16.6). Reporting it next
to the corrected `dice` (threshold 0.5) shows how much this affected RP2.

Usage (GPU server):
    CUDA_VISIBLE_DEVICES=2 python evaluate_lesions.py --ckpt_dir rp_checkpoints/rp2
    CUDA_VISIBLE_DEVICES=2 python evaluate_lesions.py --ckpt_dir rp_checkpoints/rp2 \
        --runs baseline wavelet_ml_db2_l2 --limit 2

Outputs (in --out_dir, default results/eval_<split>):
    per_patient.csv   one row per (run, case)
    per_lesion.csv    one row per GT lesion and per predicted lesion
    summary.csv       one row per run: means over cases + size-binned lesion recall
"""

import argparse
import csv
import glob
import math
import os
import re

import numpy as np
import torch
from tqdm import tqdm
from monai.data import CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

from build_datalist import build_test_list, build_train_list
from lesion_metrics import lesion_metrics, soft_dice, surface_metrics, voxel_metrics
from model import build_model
from splits import VAL_PATIENTS
from transforms import get_val_transforms

# ------------------
# Run registry
# ------------------
# RP2 checkpoints (no seed suffix) were trained with the legacy multi-level
# recursion; runs from the current train.py are named <run>_s<seed>.
RUNS = {
    "baseline":  {"variant": "baseline",  "ckpt": ["best_baseline.pth"]},
    "wavelet_a": {"variant": "wavelet_a", "ckpt": ["best_wavelet_a_single_level.pth", "best_wavelet_a.pth"]},
}
for _w in ("haar", "db2", "sym4"):
    for _l in (1, 2, 3):
        RUNS[f"wavelet_ml_{_w}_l{_l}"] = {
            "variant": "wavelet_ml", "wavelet": _w, "levels": _l,
            "ckpt": [f"best_wavelet_ml_{_w}_l{_l}.pth"],
            "legacy_se_recursion": True,
        }
RP2_RUNS = list(RUNS)


def spec_from_name(run):
    """Spec for a seeded run name, e.g. 'wavelet_ml_sym4_l1_s2' or 'baseline_s1'."""
    # Optional recipe/loss suffixes (_r2, _dicebce, _hf<w>) don't change the architecture.
    m = re.fullmatch(r"(baseline|wavelet_a|detail_skip_plain|detail_skip_haar|wavelet_ml_(haar|db2|sym4)_l(\d))"
                     r"(?:_r2)?(?:_dicebce)?(?:_hf[\d.]+)?_s(\d+)", run)
    if not m:
        return None
    if m.group(2):
        return {"variant": "wavelet_ml", "wavelet": m.group(2), "levels": int(m.group(3)),
                "ckpt": [f"best_{run}.pth"]}
    return {"variant": m.group(1), "ckpt": [f"best_{run}.pth"]}

# ------------------
# Args
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True,
                    help="Folder with checkpoints (searched recursively)")
parser.add_argument("--runs", nargs="+", default=None,
                    help="Runs to evaluate: RP2 names (default: all RP2 runs) "
                         "or seeded names like wavelet_ml_sym4_l1_s1")
parser.add_argument("--discover", action="store_true",
                    help="Evaluate every best_<run>_s<seed>.pth found in --ckpt_dir")
parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--min_lesion_size", type=int, default=3,
                    help="Components smaller than this (voxels) are ignored in lesion metrics")
parser.add_argument("--size_bins", type=str, default="40,115",
                    help="Lesion volume bin edges in mm^3: small < a <= medium < b <= large")
parser.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N cases (smoke test)")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--tta", type=str, default="none", choices=["none", "flip3", "flip7"],
                    help="Test-time augmentation: average over the identity plus mirror flips "
                         "(flip3: each axis once = 4 passes; flip7: all axis combinations = 8 passes)")
parser.add_argument("--overlap", type=float, default=0.25,
                    help="Sliding-window overlap (MONAI default 0.25)")
parser.add_argument("--save_probs", action="store_true",
                    help="Save float16 probability maps (npz) for threshold_sweep.py")
args = parser.parse_args()

ROOT          = "MSLesSeg_Dataset"
ROI_SIZE      = (96, 96, 96)
SW_BATCH_SIZE = 2
SPACING       = (1.0, 1.0, 1.0)    # transforms.py resamples to 1 mm isotropic
VOXEL_VOLUME  = float(np.prod(SPACING))
OUT_DIR       = args.out_dir or os.path.join("results", f"eval_{args.split}")
BIN_EDGES     = [float(v) for v in args.size_bins.split(",")]
BINS = [("small", 0.0, BIN_EDGES[0]), ("medium", BIN_EDGES[0], BIN_EDGES[1]),
        ("large", BIN_EDGES[1], math.inf)]

os.makedirs(OUT_DIR, exist_ok=True)

if args.discover:
    found = glob.glob(os.path.join(args.ckpt_dir, "**", "best_*_s*.pth"), recursive=True)
    names = sorted({os.path.basename(p)[len("best_"):-len(".pth")] for p in found})
    args.runs = [n for n in names if spec_from_name(n)]
elif args.runs is None:
    args.runs = RP2_RUNS
for run in args.runs:
    if run not in RUNS:
        spec = spec_from_name(run)
        if spec is None:
            parser.error(f"unknown run name '{run}'")
        RUNS[run] = spec
print(f"Runs: {args.runs}")


def find_checkpoint(names):
    for name in names:
        hits = sorted(glob.glob(os.path.join(args.ckpt_dir, "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def size_bin(volume_mm3):
    for name, lo, hi in BINS:
        if lo <= volume_mm3 < hi:
            return name
    return BINS[-1][0]


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------
# Data — preprocessed once, cached in RAM, reused for every run
# ------------------
if args.split == "test":
    data = build_test_list(ROOT)
    for d in data:
        d["case"] = d["patient"]
else:
    data = build_train_list(ROOT, VAL_PATIENTS)
    for d in data:
        d["case"] = f"{d['patient']}_{d['timepoint']}"
if args.limit:
    data = data[:args.limit]

print(f"Split: {args.split}  |  cases: {len(data)}  |  threshold: {args.threshold}  "
      f"|  min lesion size: {args.min_lesion_size} vox  |  size bins (mm^3): {BIN_EDGES}")
ds = CacheDataset(data, transform=get_val_transforms(), cache_rate=1.0, num_workers=4)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

# ------------------
# Inference (optionally with flip TTA)
# ------------------
FLIPS = {"none": [()],
         "flip3": [(), (2,), (3,), (4,)],
         "flip7": [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]}[args.tta]


def predict(model, x):
    """Sigmoid probabilities, averaged over the TTA flips (flipped back first)."""
    acc = 0.0
    for dims in FLIPS:
        xi = torch.flip(x, dims) if dims else x
        p = torch.sigmoid(sliding_window_inference(xi, ROI_SIZE, SW_BATCH_SIZE, model,
                                                   overlap=args.overlap))
        acc = acc + (torch.flip(p, dims) if dims else p)
    return acc / len(FLIPS)


# ------------------
# Evaluate
# ------------------
all_patient_rows, all_lesion_rows, summary_rows = [], [], []

for run in args.runs:
    spec = RUNS[run]
    ckpt = find_checkpoint(spec["ckpt"])
    if ckpt is None:
        print(f"\n[{run}] checkpoint not found ({spec['ckpt']}) — skipped")
        continue
    print(f"\n[{run}] loading {ckpt}")

    model = build_model(spec["variant"], in_channels=1,
                        wavelet=spec.get("wavelet", "haar"),
                        levels=spec.get("levels", 1),
                        legacy_se_recursion=spec.get("legacy_se_recursion", False)).to(args.device)
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[{run}] state_dict mismatch — skipped\n{e}")
        continue
    model.eval()

    rp2_metric = DiceMetric(include_background=False, reduction="mean")
    patient_rows, lesion_rows = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=run):
            case = batch["case"][0]
            x = batch["image"].to(args.device)
            y = batch["label"].to(args.device)

            probs = predict(model, x)

            # Dice exactly as computed in RP2 (train.py / final_test_evaluation.py)
            rp2_dice = float(rp2_metric(probs, y).mean())

            prob = probs[0, 0].float().cpu().numpy()
            gt   = y[0, 0].cpu().numpy() > 0.5
            pred = prob > args.threshold

            row = {"run": run, "case": case, "rp2_dice": rp2_dice,
                   "soft_dice": soft_dice(prob, gt),
                   "frac_pred_prob_eq_1": float((prob == 1.0).sum() / max(pred.sum(), 1))}
            row.update(voxel_metrics(pred, gt))
            row.update(surface_metrics(pred, gt, SPACING))
            les_summary, les_records = lesion_metrics(
                pred, gt, prob, min_size=args.min_lesion_size, voxel_volume=VOXEL_VOLUME)
            row.update(les_summary)
            patient_rows.append(row)

            for rec in les_records:
                rec.update({"run": run, "case": case, "size_bin": size_bin(rec["volume_mm3"])})
                lesion_rows.append(rec)

            if args.save_probs:
                p_dir = os.path.join(OUT_DIR, "probs", run)
                os.makedirs(p_dir, exist_ok=True)
                # Probabilities < 0.01 are zeroed so the file compresses well;
                # threshold_sweep.py never uses thresholds below that.
                prob16 = prob.astype(np.float16)
                prob16[prob < 0.01] = 0
                np.savez_compressed(os.path.join(p_dir, f"{case}.npz"), prob=prob16, gt=gt)

    # ---- per-run summary ----
    def mean(key):
        vals = [r[key] for r in patient_rows if not math.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def std(key):
        vals = [r[key] for r in patient_rows if not math.isnan(r[key])]
        return float(np.std(vals)) if vals else float("nan")

    s = {"run": run, "n_cases": len(patient_rows)}
    for key in ("rp2_dice", "soft_dice", "dice", "precision", "recall", "avd",
                "hd95", "assd", "lesion_recall", "lesion_precision", "lesion_f1",
                "n_fp_lesions", "n_missed"):
        s[key] = mean(key)
    s["dice_std"] = std("dice")
    s["hd95_nan_cases"] = sum(math.isnan(r["hd95"]) for r in patient_rows)

    # Lesion recall per size bin, pooled over all GT lesions of all cases
    for name, _, _ in BINS:
        gts = [r for r in lesion_rows if r["kind"] == "gt" and r["size_bin"] == name]
        s[f"n_gt_{name}"] = len(gts)
        s[f"recall_{name}"] = (sum(r["hit"] for r in gts) / len(gts)) if gts else float("nan")
        prs = [r for r in lesion_rows if r["kind"] == "pred" and r["size_bin"] == name]
        s[f"n_fp_{name}"] = sum(not r["hit"] for r in prs)
    summary_rows.append(s)

    print(f"[{run}] rp2_dice={s['rp2_dice']:.4f}  dice@{args.threshold}={s['dice']:.4f}  "
          f"soft={s['soft_dice']:.4f}  L-recall={s['lesion_recall']:.3f}  "
          f"L-prec={s['lesion_precision']:.3f}  FP/case={s['n_fp_lesions']:.2f}  "
          f"HD95={s['hd95']:.2f}")

    all_patient_rows += patient_rows
    all_lesion_rows += lesion_rows

    # Write after every run so partial results survive an interruption
    write_csv(os.path.join(OUT_DIR, "per_patient.csv"), all_patient_rows)
    write_csv(os.path.join(OUT_DIR, "per_lesion.csv"), all_lesion_rows)
    write_csv(os.path.join(OUT_DIR, "summary.csv"), summary_rows)

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

# ------------------
# Final table
# ------------------
if summary_rows:
    gt_vols = sorted(r["volume_mm3"] for r in all_lesion_rows
                     if r["kind"] == "gt" and r["run"] == summary_rows[0]["run"])
    if gt_vols:
        q = np.percentile(gt_vols, [10, 25, 33, 50, 67, 75, 90])
        print(f"\nGT lesion volumes (mm^3), n={len(gt_vols)}: "
              f"p10={q[0]:.0f} p25={q[1]:.0f} p33={q[2]:.0f} p50={q[3]:.0f} "
              f"p67={q[4]:.0f} p75={q[5]:.0f} p90={q[6]:.0f}")

    hdr = (f"{'run':<22}{'rp2':>7}{'dice':>7}{'soft':>7}{'L-rec':>7}{'L-prec':>7}"
           f"{'L-F1':>7}{'FP/pt':>7}{'HD95':>7}{'rec_S':>7}{'rec_M':>7}{'rec_L':>7}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for s in summary_rows:
        print(f"{s['run']:<22}{s['rp2_dice']:>7.4f}{s['dice']:>7.4f}{s['soft_dice']:>7.4f}"
              f"{s['lesion_recall']:>7.3f}{s['lesion_precision']:>7.3f}{s['lesion_f1']:>7.3f}"
              f"{s['n_fp_lesions']:>7.2f}{s['hd95']:>7.2f}{s['recall_small']:>7.3f}"
              f"{s['recall_medium']:>7.3f}{s['recall_large']:>7.3f}")
    print(f"\nResults written to {OUT_DIR}/")
