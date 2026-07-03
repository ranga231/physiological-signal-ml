"""
diagnose_acc_threshold.py
=========================
Investigate why ACC artifact detection flags nearly all windows.

Run from the project root:
    python diagnose_acc_threshold.py

Prints:
  - Raw ACC min/max/mean to identify units
  - HP-filtered RMS distribution (percentiles)
  - Flagged window count at 10 different thresholds for all subjects
"""

import sys
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, ".")
from src.ingestion.wesad_loader import WESADSubject, FS

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WINDOW_S = 60.0
STEP_S   = 30.0
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]

def hp_rms(acc, fs):
    nyq = fs / 2.0
    b, a = butter(2, 0.5 / nyq, btype="high")
    acc_hp = filtfilt(b, a, acc, axis=0)
    return np.sqrt((acc_hp ** 2).sum(axis=1))

def sweep_thresholds(rms, fs, window_s, step_s, thresholds):
    n_win  = int(window_s * fs)
    n_step = int(step_s   * fs)
    n_windows = (len(rms) - n_win) // n_step + 1
    results = {}
    for t in thresholds:
        flagged = sum(
            1 for i in range(n_windows)
            if (rms[i*n_step : i*n_step + n_win] > t).mean() > 0.20
        )
        results[t] = (flagged, n_windows)
    return results

print("=" * 70)
print("ACC UNIT CHECK (raw values before any filtering)")
print("=" * 70)
s2 = WESADSubject(2, "data/raw")
acc = s2.acc
raw_mag = np.sqrt((acc**2).sum(axis=1))
print(f"S2 ACC shape       : {acc.shape}")
print(f"S2 ACC raw range   : [{acc.min():.4f}, {acc.max():.4f}]")
print(f"S2 ACC mean/axis   : {acc.mean(axis=0)}")
print(f"S2 raw magnitude   : mean={raw_mag.mean():.4f}  (should be ~1.0 if in g, ~64 if in 1/64g)")
print()

rms2 = hp_rms(acc, FS["ACC"])
print(f"S2 HP-filtered RMS stats:")
for p in [50, 75, 90, 95, 99, 100]:
    print(f"  p{p:3d}: {np.percentile(rms2, p):.6f}")
print()

print("=" * 70)
print("THRESHOLD SWEEP — all subjects")
print("=" * 70)
header = f"{'Subj':>5} | " + " | ".join(f"{t:.2f}" for t in THRESHOLDS)
print(header)
print("-" * len(header))

totals = {t: [0, 0] for t in THRESHOLDS}
for sid in SUBJECTS:
    try:
        s = WESADSubject(sid, "data/raw")
        rms = hp_rms(s.acc, FS["ACC"])
        res = sweep_thresholds(rms, FS["ACC"], WINDOW_S, STEP_S, THRESHOLDS)
        row = f"  S{sid:2d} | " + " | ".join(
            f"{res[t][0]:>3}/{res[t][1]}" for t in THRESHOLDS
        )
        print(row)
        for t in THRESHOLDS:
            totals[t][0] += res[t][0]
            totals[t][1] += res[t][1]
    except Exception as e:
        print(f"  S{sid:2d} | ERROR: {e}")

print("-" * len(header))
pct_row = "  TOT | " + " | ".join(
    f"{100*totals[t][0]/max(totals[t][1],1):>5.0f}%" for t in THRESHOLDS
)
print(pct_row + "  (% windows flagged)")
print()
print("Look for the threshold where flagging drops below ~20% — that's a good calibrated value.")
