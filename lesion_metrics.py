"""
lesion_metrics.py

Voxel-, boundary- and lesion-level metrics for binary MS lesion masks.

All functions take numpy arrays of shape (D, H, W). Masks are boolean.

Lesions are 26-connected components. Components smaller than `min_size`
voxels are removed from BOTH the ground truth and the prediction before
lesion matching. A GT lesion counts as detected if it overlaps any predicted
lesion voxel; a predicted lesion counts as a true positive if it overlaps any
GT lesion voxel, otherwise it is a false-positive lesion.
"""

import numpy as np
from scipy import ndimage

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
STRUCT_6  = ndimage.generate_binary_structure(3, 1)


# ------------------
# Connected components
# ------------------
def label_lesions(mask: np.ndarray, min_size: int = 3):
    """26-connected components, dropping those smaller than min_size voxels.

    Returns (labels, n) with labels renumbered 1..n.
    """
    labels, n = ndimage.label(mask, structure=STRUCT_26)
    if n == 0:
        return labels, 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[labels], int(keep.sum())


# ------------------
# Voxel-level
# ------------------
def voxel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(np.logical_and(pred, gt).sum())
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    fp, fn = n_pred - tp, n_gt - tp

    if n_pred + n_gt == 0:
        dice = 1.0
    else:
        dice = 2 * tp / (n_pred + n_gt)

    return {
        "dice":      dice,
        "precision": tp / n_pred if n_pred else float("nan"),
        "recall":    tp / n_gt if n_gt else float("nan"),
        "avd":       abs(n_pred - n_gt) / n_gt if n_gt else float("nan"),
        "tp_vox": tp, "fp_vox": fp, "fn_vox": fn,
        "pred_vox": n_pred, "gt_vox": n_gt,
    }


def soft_dice(prob: np.ndarray, gt: np.ndarray) -> float:
    denom = float(prob.sum() + gt.sum())
    return 2 * float((prob * gt).sum()) / denom if denom > 0 else 1.0


# ------------------
# Boundary
# ------------------
def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=STRUCT_6)


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """HD95 and ASSD in mm. NaN if either mask is empty."""
    if not pred.any() or not gt.any():
        return {"hd95": float("nan"), "assd": float("nan")}

    # Crop to the joint bounding box (+1 voxel) — exact, and much faster
    # than a full-volume distance transform.
    idx = np.argwhere(pred | gt)
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = idx.max(0) + 2
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    pred, gt = pred[sl], gt[sl]

    s_pred, s_gt = _surface(pred), _surface(gt)
    dt_gt   = ndimage.distance_transform_edt(~s_gt,   sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~s_pred, sampling=spacing)
    d_p2g = dt_gt[s_pred]
    d_g2p = dt_pred[s_gt]

    hd95 = max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))
    assd = (d_p2g.sum() + d_g2p.sum()) / (len(d_p2g) + len(d_g2p))
    return {"hd95": float(hd95), "assd": float(assd)}


# ------------------
# Lesion-level
# ------------------
def lesion_metrics(pred: np.ndarray, gt: np.ndarray, prob: np.ndarray | None = None,
                   min_size: int = 3, voxel_volume: float = 1.0):
    """Lesion detection metrics plus one record per GT and per predicted lesion.

    Returns (summary: dict, records: list[dict]).
    """
    gt_lab, n_gt = label_lesions(gt, min_size)
    pr_lab, n_pr = label_lesions(pred, min_size)

    gt_size = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pr_size = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)
    gt_overlap = np.bincount(gt_lab[pr_lab > 0], minlength=n_gt + 1)  # voxels of each GT lesion that are predicted
    pr_overlap = np.bincount(pr_lab[gt_lab > 0], minlength=n_pr + 1)  # voxels of each pred lesion that are GT

    gt_detected = gt_overlap[1:] > 0
    pr_tp       = pr_overlap[1:] > 0
    n_det, n_tp = int(gt_detected.sum()), int(pr_tp.sum())

    l_recall    = n_det / n_gt if n_gt else float("nan")
    l_precision = n_tp / n_pr if n_pr else float("nan")
    if n_gt and n_pr and (l_recall + l_precision) > 0:
        l_f1 = 2 * l_recall * l_precision / (l_recall + l_precision)
    elif n_gt == 0 and n_pr == 0:
        l_f1 = 1.0
    else:
        l_f1 = 0.0

    summary = {
        "n_gt_lesions":     n_gt,
        "n_pred_lesions":   n_pr,
        "n_detected":       n_det,
        "n_missed":         n_gt - n_det,
        "n_tp_pred":        n_tp,
        "n_fp_lesions":     n_pr - n_tp,
        "lesion_recall":    l_recall,
        "lesion_precision": l_precision,
        "lesion_f1":        l_f1,
    }

    records = []
    for kind, lab, n, size, overlap, hit in (
        ("gt",   gt_lab, n_gt, gt_size, gt_overlap, gt_detected),
        ("pred", pr_lab, n_pr, pr_size, pr_overlap, pr_tp),
    ):
        if n == 0:
            continue
        ids = np.arange(1, n + 1)
        centroids = ndimage.center_of_mass(np.ones_like(lab, dtype=np.uint8), lab, ids)
        if prob is not None:
            p_max  = ndimage.maximum(prob, lab, ids)
            p_mean = ndimage.mean(prob, lab, ids)
        for k, i in enumerate(ids):
            rec = {
                "kind":        kind,
                "lesion_id":   int(i),
                "volume_vox":  int(size[i]),
                "volume_mm3":  float(size[i] * voxel_volume),
                "centroid":    "{:.1f};{:.1f};{:.1f}".format(*centroids[k]),
                "overlap_vox": int(overlap[i]),
                # GT: detected; pred: true positive
                "hit":         bool(hit[k]),
            }
            if prob is not None:
                rec["prob_max"]  = float(p_max[k])
                rec["prob_mean"] = float(p_mean[k])
            records.append(rec)

    return summary, records
