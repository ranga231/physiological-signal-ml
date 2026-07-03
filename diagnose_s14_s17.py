"""
diagnose_s14_s17.py
====================
Investigates why S14 and S17 are hard subjects for the RF classifier.

Checks:
1. PPG peak detection quality (how many RR intervals per window)
2. Frequency-domain feature availability (are LF/HF features all NaN?)
3. Feature distributions vs. other subjects (are their HRV features outliers?)
4. Label distribution (is the stress/non-stress split unusual?)

Run from project root:
    python diagnose_s14_s17.py
"""

import sys
import numpy as np
import warnings
sys.path.insert(0, ".")

from src.ingestion.wesad_loader import WESADSubject, FS
from src.preprocessing.signal_processing import segment_subject, WindowConfig
from src.features.hrv_features import (
    detect_ppg_peaks, _rr_intervals_ms, hrv_time_domain,
    hrv_frequency_domain, FEATURE_NAMES, extract_subject_features
)

HARD_SUBJECTS  = [14, 17]
OTHER_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16]
ALL_SUBJECTS   = HARD_SUBJECTS + OTHER_SUBJECTS

cfg = WindowConfig(window_s=60, step_s=30)

# ── Load all subjects ─────────────────────────────────────────────────────────
print("Loading subjects...")
windows = {}
features = {}
for sid in ALL_SUBJECTS:
    s = WESADSubject(sid, "data/raw")
    sw = segment_subject(s, cfg)
    windows[sid] = sw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[sid] = extract_subject_features(sw)

# ── 1. Label distribution ─────────────────────────────────────────────────────
print("\n── 1. Label Distribution ────────────────────────────────────────")
print(f"{'Subj':>5} | {'Total':>5} | {'Stress':>6} | {'NonStr':>6} | {'Stress%':>7} | {'Artifact':>8}")
print("-" * 55)
for sid in ALL_SUBJECTS:
    sw = windows[sid]
    n = len(sw.labels)
    ns = (sw.labels == 1).sum()
    nn = (sw.labels == 0).sum()
    na = sw.artifact_mask.sum()
    marker = " <-- HARD" if sid in HARD_SUBJECTS else ""
    print(f"  S{sid:2d} | {n:>5} | {ns:>6} | {nn:>6} | {100*ns/max(n,1):>6.1f}% | {na:>8}{marker}")

# ── 2. PPG peak detection quality ────────────────────────────────────────────
print("\n── 2. PPG Peak Detection Quality (RR intervals per 60s window) ──")
print(f"{'Subj':>5} | {'Mean RR/win':>10} | {'Min RR/win':>10} | {'% wins < 8 RR':>13} | {'% wins 0 peaks':>14}")
print("-" * 65)

for sid in ALL_SUBJECTS:
    sw = windows[sid]
    rr_counts = []
    zero_peak_wins = 0
    for i in range(sw.windows_bvp.shape[0]):
        bvp = sw.windows_bvp[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks = detect_ppg_peaks(bvp, fs=FS["BVP"])
        n_rr = max(0, len(peaks) - 1)
        rr_counts.append(n_rr)
        if len(peaks) == 0:
            zero_peak_wins += 1

    rr_counts = np.array(rr_counts)
    pct_low = 100 * (rr_counts < 8).mean()
    pct_zero = 100 * zero_peak_wins / max(len(rr_counts), 1)
    marker = " <--" if sid in HARD_SUBJECTS else ""
    print(f"  S{sid:2d} | {rr_counts.mean():>10.1f} | {rr_counts.min():>10d} | "
          f"{pct_low:>12.1f}% | {pct_zero:>13.1f}%{marker}")

# ── 3. Frequency-domain feature availability ──────────────────────────────────
print("\n── 3. Frequency-Domain HRV Feature Availability ────────────────")
# Features 8-11 are lf_power, hf_power, lf_hf, total_power
fd_idx = [8, 9, 10, 11]
fd_names = [FEATURE_NAMES[i] for i in fd_idx]
print(f"{'Subj':>5} | {'lf_power % valid':>16} | {'hf_power % valid':>16} | {'lf_hf % valid':>13}")
print("-" * 60)
for sid in ALL_SUBJECTS:
    X = features[sid]
    pct_lf   = 100 * (~np.isnan(X[:, 8])).mean()
    pct_hf   = 100 * (~np.isnan(X[:, 9])).mean()
    pct_lfhf = 100 * (~np.isnan(X[:, 10])).mean()
    marker = " <--" if sid in HARD_SUBJECTS else ""
    print(f"  S{sid:2d} | {pct_lf:>15.1f}% | {pct_hf:>15.1f}% | {pct_lfhf:>12.1f}%{marker}")

# ── 4. Key HRV feature distributions ─────────────────────────────────────────
print("\n── 4. Key Feature Means: Hard vs. Other Subjects ───────────────")
key_features = {
    "hr_mean": 0,
    "rmssd":   3,
    "sdnn":    2,
    "eda_mean": FEATURE_NAMES.index("eda_mean"),
    "temp_mean": FEATURE_NAMES.index("temp_mean"),
}

# Separate stress and non-stress windows
print(f"\n  STRESS windows:")
print(f"  {'Feature':>12} | {'S14':>8} | {'S17':>8} | {'Others mean':>11} | {'Others std':>10}")
print("  " + "-" * 55)
for fname, fidx in key_features.items():
    vals_14 = features[14][windows[14].labels == 1, fidx]
    vals_17 = features[17][windows[17].labels == 1, fidx]
    vals_other = np.concatenate([
        features[sid][windows[sid].labels == 1, fidx]
        for sid in OTHER_SUBJECTS
    ])
    m14 = np.nanmean(vals_14)
    m17 = np.nanmean(vals_17)
    mo  = np.nanmean(vals_other)
    so  = np.nanstd(vals_other)
    print(f"  {fname:>12} | {m14:>8.3f} | {m17:>8.3f} | {mo:>11.3f} | {so:>10.3f}")

print(f"\n  NON-STRESS windows:")
print(f"  {'Feature':>12} | {'S14':>8} | {'S17':>8} | {'Others mean':>11} | {'Others std':>10}")
print("  " + "-" * 55)
for fname, fidx in key_features.items():
    vals_14 = features[14][windows[14].labels == 0, fidx]
    vals_17 = features[17][windows[17].labels == 0, fidx]
    vals_other = np.concatenate([
        features[sid][windows[sid].labels == 0, fidx]
        for sid in OTHER_SUBJECTS
    ])
    m14 = np.nanmean(vals_14)
    m17 = np.nanmean(vals_17)
    mo  = np.nanmean(vals_other)
    so  = np.nanstd(vals_other)
    print(f"  {fname:>12} | {m14:>8.3f} | {m17:>8.3f} | {mo:>11.3f} | {so:>10.3f}")

# ── 5. Stress vs non-stress separation ───────────────────────────────────────
print("\n── 5. Stress vs Non-Stress Separation (per subject) ────────────")
print("   (rmssd stress mean vs non-stress mean — should be lower under stress)")
print(f"  {'Subj':>5} | {'rmssd stress':>12} | {'rmssd non-str':>13} | {'direction':>10}")
print("  " + "-" * 50)
rmssd_idx = FEATURE_NAMES.index("rmssd")
for sid in ALL_SUBJECTS:
    X = features[sid]
    sw = windows[sid]
    s_mean  = np.nanmean(X[sw.labels == 1, rmssd_idx])
    ns_mean = np.nanmean(X[sw.labels == 0, rmssd_idx])
    direction = "correct (stress < rest)" if s_mean < ns_mean else "INVERTED <--"
    marker = " ***" if sid in HARD_SUBJECTS else ""
    print(f"  S{sid:2d}  | {s_mean:>12.3f} | {ns_mean:>13.3f} | {direction}{marker}")

print("\nDone.")
