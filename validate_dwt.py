"""
validate_dwt.py

Checks the fixed DWT used by the wavelet patch embeddings (wavelet.py: DWT3d)
against PyWavelets. CPU only, runs in seconds.

For each wavelet (haar, db2, sym4) it checks:
  1. Coefficients   — are the stored 1-D filters PyWavelets' dec_* filters,
                      and in which orientation?
  2. Orthogonality  — energy preservation (Parseval) and perfect
                      reconstruction with the transposed filter bank.
  3. Equivalence    — does DWT3d produce the same sub-bands as pywt.dwtn,
                      either with the true filters or with time-reversed ones
                      (i.e. the mirrored wavelet), and in the same sub-band
                      order LLL, LLH, ..., HHH?

Checks 2 and 3 use a random volume with a zero border, so zero padding at the
edges plays no role.

Usage:
    pip install --user PyWavelets     # once
    python validate_dwt.py
"""

import itertools

import numpy as np
import pywt
import torch
import torch.nn.functional as F

from wavelet import DWT3d, _WAVELET_FILTERS

N, BORDER = 32, 8
TOL = 1e-5
rng = np.random.default_rng(0)
x = np.zeros((N, N, N))
x[BORDER:-BORDER, BORDER:-BORDER, BORDER:-BORDER] = rng.standard_normal((N - 2 * BORDER,) * 3)
x_t = torch.tensor(x, dtype=torch.float64)[None, None]

# Sub-band order in DWT3d: loops (D, H, W) over (L, H) -> index = 4*d + 2*h + w
BAND_NAMES = ["".join(b) for b in itertools.product("LH", repeat=3)]
PYWT_KEYS  = [n.replace("L", "a").replace("H", "d") for n in BAND_NAMES]


def best_shift_error(ours, ref):
    """Smallest relative error between `ours` and any integer 3-D shift of `ref`,
    also allowing a global sign flip. Returns (rel_err, shift, sign)."""
    n = ours.shape[0]
    pad = n
    ref_p = np.pad(ref, pad)
    best = (np.inf, None, None)
    norm = np.linalg.norm(ours) + 1e-12
    for s in range(0, ref_p.shape[0] - n + 1):
        win = ref_p[s:s + n, s:s + n, s:s + n]
        for sign in (1, -1):
            err = np.linalg.norm(ours - sign * win) / norm
            if err < best[0]:
                best = (err, s - pad, sign)
    return best


def compare_to_pywt(bands, wavelet, reverse, phase=0):
    """Max relative error over the 8 sub-bands vs pywt.dwtn (optionally with
    time-reversed filters, implemented as flip -> dwtn -> flip). `phase`
    shifts the input by that many voxels along every axis first, which
    changes which samples the stride-2 decimation keeps."""
    xin = np.roll(x, phase, axis=(0, 1, 2))
    xin = xin[::-1, ::-1, ::-1].copy() if reverse else xin
    ref = pywt.dwtn(xin, wavelet, mode="zero")
    worst, details = 0.0, []
    for i, key in enumerate(PYWT_KEYS):
        r = ref[key][::-1, ::-1, ::-1] if reverse else ref[key]
        err, shift, sign = best_shift_error(bands[i], r)
        worst = max(worst, err)
        details.append((BAND_NAMES[i], err, shift, sign))
    return worst, details


all_ok = True
for name in ("haar", "db2", "sym4"):
    print(f"\n=== {name} ===")
    w = pywt.Wavelet(name)
    lo, hi = (np.array(f) for f in _WAVELET_FILTERS[name])

    # 1. coefficients
    matches = {
        "dec (as stored by pywt)": (np.array(w.dec_lo), np.array(w.dec_hi)),
        "dec reversed (= rec)":    (np.array(w.dec_lo)[::-1], np.array(w.dec_hi)[::-1]),
    }
    for label, (plo, phi) in matches.items():
        lo_ok, hi_ok = np.allclose(lo, plo, atol=1e-10), np.allclose(hi, phi, atol=1e-10)
        hi_neg = np.allclose(hi, -phi, atol=1e-10)
        print(f"  filters vs pywt {label:<24}: lo {'MATCH' if lo_ok else '-'}   "
              f"hi {'MATCH' if hi_ok else ('MATCH (sign flipped)' if hi_neg else '-')}")

    # 2. orthogonality
    dwt = DWT3d(name).double()
    with torch.no_grad():
        y = dwt(x_t)
        x_rec = F.conv_transpose3d(y, dwt.weight, stride=2, padding=dwt.padding)
    energy_ratio = float((y ** 2).sum() / (x_t ** 2).sum())
    rec_err = float((x_rec - x_t).abs().max()) if x_rec.shape == x_t.shape else float("inf")
    orth_ok = abs(energy_ratio - 1) < TOL and rec_err < TOL
    print(f"  energy ratio sum|DWT|^2 / sum|x|^2 : {energy_ratio:.8f}")
    print(f"  perfect reconstruction max error   : {rec_err:.2e}   "
          f"{'OK' if orth_ok else 'FAIL'}")

    # 3. equivalence with pywt.dwtn
    bands = y[0].numpy()
    results = []
    for reverse in (False, True):
        for phase in (0, 1, -1):
            err, det = compare_to_pywt(bands, name, reverse, phase)
            results.append((err, det, reverse, phase))
            print(f"  vs pywt.dwtn, {'reversed' if reverse else 'true'} filters, "
                  f"input phase {phase:+d} : max rel err {err:.2e}")
    best_err, best_det, reverse, phase = min(results, key=lambda r: r[0])
    if best_err < TOL:
        shifts = sorted({d[2] for d in best_det})
        flips = [d[0] for d in best_det if d[3] == -1]
        kind = f"time-reversed (mirrored) {name}" if reverse else f"standard {name}"
        print(f"  => DWT3d computes the {kind} DWT"
              + ("" if phase == 0 else f", sampled at the other decimation phase "
                                        f"(= pywt on the input shifted by {phase:+d} voxel)")
              + f"; sub-band order matches pywt keys {PYWT_KEYS[0]}..{PYWT_KEYS[-1]}; "
              f"coefficient shift(s) {shifts}"
              + (f"; sign flipped in {flips}" if flips else ""))
    else:
        all_ok = False
        print("  => DWT3d does NOT match pywt in any orientation / phase. Per band (true filters, phase 0):")
        for band, err, shift, sign in results[0][1]:
            print(f"       {band}: rel err {err:.2e} (shift {shift}, sign {sign})")
    all_ok &= orth_ok

print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED — see above")
