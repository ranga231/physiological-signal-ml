"""
diagnose_lombscargle.py
========================
Debug why frequency-domain HRV features are all NaN.

Run from project root:
    python diagnose_lombscargle.py
"""
import sys, warnings
import numpy as np
sys.path.insert(0, ".")

from src.ingestion.wesad_loader import WESADSubject, FS
from src.preprocessing.signal_processing import segment_subject, WindowConfig
from src.features.hrv_features import detect_ppg_peaks, _rr_intervals_ms

# Load one subject
s = WESADSubject(2, "data/raw")
cfg = WindowConfig(window_s=60, step_s=30)
sw = segment_subject(s, cfg)

# Take one window with good peak count
bvp = sw.windows_bvp[5]
peaks = detect_ppg_keys = detect_ppg_peaks(bvp, fs=FS["BVP"])
rr_ms = _rr_intervals_ms(peaks, fs=FS["BVP"])

print(f"Window 5 — peaks: {len(peaks)}, RR intervals: {len(rr_ms)}")
print(f"RR mean: {rr_ms.mean():.1f} ms, std: {rr_ms.std():.1f} ms")
print(f"RR min: {rr_ms.min():.1f}, max: {rr_ms.max():.1f}")

# Now step through hrv_frequency_domain manually
print("\n── Lomb-Scargle step-by-step ────────────────────────────────────")

print(f"Step 1: len(rr_ms) = {len(rr_ms)} (need >= 8)")

try:
    from scipy.signal import lombscargle
    print("Step 2: lombscargle import OK")
except Exception as e:
    print(f"Step 2: lombscargle import FAILED: {e}")
    sys.exit(1)

try:
    t = np.cumsum(rr_ms) / 1000.0
    t -= t[0]
    print(f"Step 3: time axis OK — range 0 to {t[-1]:.1f}s, {len(t)} points")
except Exception as e:
    print(f"Step 3: time axis FAILED: {e}")

try:
    freqs = np.linspace(0.01, 0.5, 500)
    ang_freqs = 2 * np.pi * freqs
    print(f"Step 4: freq axis OK — {len(freqs)} freqs from {freqs[0]} to {freqs[-1]} Hz")
except Exception as e:
    print(f"Step 4: freq axis FAILED: {e}")

try:
    rr_centered = rr_ms - rr_ms.mean()
    print(f"Step 5: centering OK — mean={rr_ms.mean():.2f}, centered mean={rr_centered.mean():.6f}")
except Exception as e:
    print(f"Step 5: centering FAILED: {e}")

try:
    pgram = lombscargle(t, rr_centered, ang_freqs, normalize=True)
    print(f"Step 6: lombscargle OK — pgram shape {pgram.shape}, "
          f"min={pgram.min():.4f}, max={pgram.max():.4f}, NaNs={np.isnan(pgram).sum()}")
except Exception as e:
    print(f"Step 6: lombscargle FAILED: {e}")
    import traceback
    traceback.print_exc()

try:
    LF_BAND = (0.04, 0.15)
    HF_BAND = (0.15, 0.40)
    mask_lf = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    mask_hf = (freqs >= HF_BAND[0]) & (freqs < HF_BAND[1])
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    lf = float(_trapz(pgram[mask_lf], freqs[mask_lf]))
    hf = float(_trapz(pgram[mask_hf], freqs[mask_hf]))
    print(f"Step 7: band power OK — LF={lf:.6f}, HF={hf:.6f}, LF/HF={lf/hf if hf>0 else 'inf':.3f}")
except Exception as e:
    print(f"Step 7: band power FAILED: {e}")

# Also call the actual function and see what happens
print("\n── Calling hrv_frequency_domain directly ────────────────────────")
from src.features.hrv_features import hrv_frequency_domain
result = hrv_frequency_domain(rr_ms)
print(f"Result: {result}")

# Check if neurokit2 is available
print("\n── Environment check ────────────────────────────────────────────")
try:
    import neurokit2 as nk
    print(f"neurokit2: INSTALLED (version {nk.__version__})")
except ImportError:
    print("neurokit2: NOT installed (using scipy fallback for peak detection)")

print("\nDone.")
