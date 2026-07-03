"""
verify_artifact_fix.py
======================
After applying the /64 unit fix in wesad_loader.py, verify artifact
detection is now working at a reasonable rate.

Run from project root:
    python verify_artifact_fix.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from src.ingestion.wesad_loader import WESADSubject, FS
from src.preprocessing.signal_processing import segment_subject, WindowConfig

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
cfg = WindowConfig(window_s=60, step_s=30)

print(f"{'Subj':>5} | {'Windows':>7} | {'Flagged':>7} | {'Pct':>6} | Stress | NonStress")
print("-" * 60)

total_w = total_f = total_s = total_ns = 0
for sid in SUBJECTS:
    try:
        s = WESADSubject(sid, "data/raw")
        sw = segment_subject(s, cfg)
        n_w = len(sw.labels)
        n_f = sw.artifact_mask.sum()
        n_s = (sw.labels[~sw.artifact_mask] == 1).sum()
        n_ns = (sw.labels[~sw.artifact_mask] == 0).sum()
        print(f"  S{sid:2d} | {n_w:>7} | {n_f:>7} | {100*n_f/max(n_w,1):>5.1f}% | {n_s:>6} | {n_ns:>9}")
        total_w += n_w; total_f += n_f; total_s += n_s; total_ns += n_ns
    except Exception as e:
        print(f"  S{sid:2d} | ERROR: {e}")

print("-" * 60)
print(f"  TOT | {total_w:>7} | {total_f:>7} | {100*total_f/max(total_w,1):>5.1f}% | {total_s:>6} | {total_ns:>9}")
print(f"\nClean windows (no artifact): {total_w - total_f}")
print(f"  Stress     : {total_s}")
print(f"  Non-stress : {total_ns}")
