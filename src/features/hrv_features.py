"""
HRV & Multimodal Feature Extraction
=====================================
Extracts clinically validated features from a single windowed segment.

Features extracted
------------------
BVP / PPG (HRV time-domain and frequency-domain):
  - mean HR, SDNN, RMSSD, pNN50
  - Poincaré: SD1, SD2, SD1/SD2 ratio
  - Frequency domain: LF power, HF power, LF/HF ratio, total power
  - Morphological: mean amplitude, pulse width, rise time

Accelerometer (motion context):
  - Mean and std of vector magnitude
  - Per-axis mean, std
  - Signal magnitude area (SMA)

EDA (sympathetic arousal proxy):
  - Mean, std, min, max, slope (linear trend)

TEMP:
  - Mean, std, slope

All features are returned as a flat 1D numpy array with named columns
accessible via FEATURE_NAMES.

References
----------
- Shaffer & Ginsberg (2017) "An Overview of Heart Rate Variability Metrics"
- Plews et al. (2013) Poincaré analysis
- Schmidt et al. (2018) WESAD feature baseline
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

# Optional: use neurokit2 for more robust peak detection if available
try:
    import neurokit2 as nk
    _NK_AVAILABLE = True
except ImportError:
    _NK_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

BVP_FS   = 64    # Hz
ACC_FS   = 32    # Hz
TEMP_FS  =  4    # Hz
EDA_FS   =  4    # Hz

LF_BAND = (0.04, 0.15)   # Hz  — low-frequency HRV band
HF_BAND = (0.15, 0.40)   # Hz  — high-frequency HRV band


# ── Peak detection ────────────────────────────────────────────────────────────

def detect_ppg_peaks(
    bvp: np.ndarray,
    fs: float = BVP_FS,
) -> np.ndarray:
    """
    Detect systolic peaks in a PPG (BVP) signal.

    Tries neurokit2 first (better artifact handling); falls back to
    scipy find_peaks with physiologically-constrained distance.

    Returns
    -------
    peaks : 1D int array of peak indices (empty if <2 peaks detected)
    """
    if _NK_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, info = nk.ppg_peaks(bvp, sampling_rate=int(fs), method="elgendi")
            peaks = info["PPG_Peaks"]
            if len(peaks) >= 2:
                return np.asarray(peaks)
        except Exception:
            pass

    # Fallback: scipy — min distance corresponds to 30 bpm (2 s)
    min_dist = int(fs * 0.33)   # ~180 bpm max
    peaks, _ = find_peaks(bvp, distance=min_dist, prominence=0.3)
    return peaks


# ── HRV feature computation ───────────────────────────────────────────────────

def _rr_intervals_ms(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak indices to RR intervals in milliseconds."""
    return np.diff(peaks) / fs * 1000.0


def hrv_time_domain(rr_ms: np.ndarray) -> dict[str, float]:
    """
    Standard HRV time-domain features.

    Parameters
    ----------
    rr_ms : RR intervals in milliseconds (need ≥ 2 values)
    """
    if len(rr_ms) < 2:
        return {k: np.nan for k in ["hr_mean", "sdnn", "rmssd", "pnn50"]}

    hr = 60_000.0 / rr_ms          # instantaneous HR in bpm
    successive_diff = np.diff(rr_ms)

    return {
        "hr_mean":  float(hr.mean()),
        "hr_std":   float(hr.std()),
        "sdnn":     float(rr_ms.std()),                           # ms
        "rmssd":    float(np.sqrt(np.mean(successive_diff ** 2))),# ms
        "pnn50":    float((np.abs(successive_diff) > 50).mean()), # fraction
    }


def hrv_poincare(rr_ms: np.ndarray) -> dict[str, float]:
    """
    Poincaré plot features: SD1, SD2, SD1/SD2 ratio.

    SD1 ≈ parasympathetic modulation (short-term variability)
    SD2 ≈ overall long-term variability
    """
    if len(rr_ms) < 3:
        return {"sd1": np.nan, "sd2": np.nan, "sd1_sd2": np.nan}

    rr_n  = rr_ms[:-1]
    rr_n1 = rr_ms[1:]

    sd1 = float(np.std((rr_n1 - rr_n) / np.sqrt(2)))
    sd2 = float(np.std((rr_n1 + rr_n) / np.sqrt(2)))
    sd1_sd2 = sd1 / sd2 if sd2 > 0 else np.nan

    return {"sd1": sd1, "sd2": sd2, "sd1_sd2": sd1_sd2}


def hrv_frequency_domain(
    rr_ms: np.ndarray,
    fs_interp: float = 4.0,
) -> dict[str, float]:
    """
    Frequency-domain HRV via Lomb-Scargle (handles unevenly spaced RR).

    Returns LF power, HF power, LF/HF ratio, total power (all in ms²).

    Uses interpolation approach if >= 8 RR intervals available.
    """
    blank = {k: np.nan for k in ["lf_power", "hf_power", "lf_hf", "total_power"]}
    if len(rr_ms) < 8:
        return blank

    try:
        from scipy.signal import lombscargle

        # Build time axis for RR series
        t = np.cumsum(rr_ms) / 1000.0   # seconds
        t -= t[0]

        # Frequency axis
        freqs = np.linspace(0.01, 0.5, 500)
        ang_freqs = 2 * np.pi * freqs

        pgram = lombscargle(t, rr_ms - rr_ms.mean(), ang_freqs, normalize=True)

        try:                        # NumPy 2.0 renamed trapz → trapezoid
            _trapz = np.trapezoid
        except AttributeError:
            _trapz = np.trapz

        def band_power(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                return np.nan
            return float(_trapz(pgram[mask], freqs[mask]))

        lf  = band_power(*LF_BAND)
        hf  = band_power(*HF_BAND)
        tot = band_power(0.04, 0.50)
        lf_hf = lf / hf if (hf and hf > 0) else np.nan

        return {"lf_power": lf, "hf_power": hf, "lf_hf": lf_hf, "total_power": tot}

    except Exception:
        return blank


def ppg_morphology(bvp: np.ndarray, peaks: np.ndarray, fs: float) -> dict[str, float]:
    """
    PPG waveform morphology features.

    Extracts mean pulse amplitude, rise time, and pulse width
    from individual beats.
    """
    blank = {"ppg_amplitude": np.nan, "ppg_rise_time_ms": np.nan, "ppg_pulse_width_ms": np.nan}
    if len(peaks) < 3:
        return blank

    amplitudes, rise_times, pulse_widths = [], [], []

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end   = peaks[i + 1]
        beat  = bvp[start:end]
        if len(beat) < 4:
            continue

        # Amplitude: peak - trough (trough is min in first half of beat)
        mid = len(beat) // 2
        trough_idx = beat[:mid].argmin()
        amp = float(beat[peaks[i] - start] - beat[trough_idx])
        amplitudes.append(amp)

        # Rise time: trough → peak
        rise_times.append((start - trough_idx) / fs * 1000)

        # Pulse width at 50% amplitude (FWHM)
        half = amp / 2
        above = (beat > (beat.min() + half))
        if above.any():
            pulse_widths.append(above.sum() / fs * 1000)

    return {
        "ppg_amplitude":     float(np.nanmean(amplitudes))   if amplitudes  else np.nan,
        "ppg_rise_time_ms":  float(np.nanmean(rise_times))   if rise_times  else np.nan,
        "ppg_pulse_width_ms":float(np.nanmean(pulse_widths)) if pulse_widths else np.nan,
    }


# ── Accelerometer features ────────────────────────────────────────────────────

def acc_features(acc: np.ndarray) -> dict[str, float]:
    """
    Motion features from 3-axis accelerometer.
    acc: (N, 3)
    """
    mag = np.sqrt((acc ** 2).sum(axis=1))    # vector magnitude

    feats: dict[str, float] = {
        "acc_mag_mean": float(mag.mean()),
        "acc_mag_std":  float(mag.std()),
        "acc_sma":      float(np.sum(np.abs(acc)) / len(acc)),  # signal magnitude area
    }
    for ax, name in enumerate(["x", "y", "z"]):
        feats[f"acc_{name}_mean"] = float(acc[:, ax].mean())
        feats[f"acc_{name}_std"]  = float(acc[:, ax].std())

    return feats


# ── EDA and TEMP features ─────────────────────────────────────────────────────

def scalar_signal_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean, std, min, max, and linear slope for a 1D signal."""
    slope = float(linregress(np.arange(len(signal)), signal).slope)
    return {
        f"{prefix}_mean":  float(signal.mean()),
        f"{prefix}_std":   float(signal.std()),
        f"{prefix}_min":   float(signal.min()),
        f"{prefix}_max":   float(signal.max()),
        f"{prefix}_slope": slope,
    }


# ── Combined feature vector ───────────────────────────────────────────────────

def extract_features(
    bvp:  np.ndarray,
    acc:  np.ndarray,
    temp: np.ndarray,
    eda:  np.ndarray,
    fs_bvp:  float = BVP_FS,
) -> np.ndarray:
    """
    Extract all features from one window. Returns a 1D float32 array.

    Parameters
    ----------
    bvp  : (N_bvp,) preprocessed BVP/PPG signal
    acc  : (N_acc, 3) preprocessed accelerometer
    temp : (N_temp,) preprocessed temperature
    eda  : (N_eda,) preprocessed EDA

    Returns
    -------
    features : (D,) float32 — use FEATURE_NAMES for column names
    """
    feats: dict[str, float] = {}

    # HRV features
    peaks = detect_ppg_peaks(bvp, fs=fs_bvp)
    if len(peaks) >= 2:
        rr = _rr_intervals_ms(peaks, fs_bvp)
        feats.update(hrv_time_domain(rr))
        feats.update(hrv_poincare(rr))
        feats.update(hrv_frequency_domain(rr))
        feats.update(ppg_morphology(bvp, peaks, fs_bvp))
    else:
        feats.update({k: np.nan for k in [
            "hr_mean", "hr_std", "sdnn", "rmssd", "pnn50",
            "sd1", "sd2", "sd1_sd2",
            "lf_power", "hf_power", "lf_hf", "total_power",
            "ppg_amplitude", "ppg_rise_time_ms", "ppg_pulse_width_ms",
        ]})

    feats.update(acc_features(acc))
    feats.update(scalar_signal_features(temp, "temp"))
    feats.update(scalar_signal_features(eda, "eda"))

    return np.array(list(feats.values()), dtype=np.float32)


# ── Feature names ─────────────────────────────────────────────────────────────

# Build canonical ordered list by running on dummy data once
def _build_feature_names() -> list[str]:
    dummy = {
        "hr_mean": 0, "hr_std": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0,
        "sd1": 0, "sd2": 0, "sd1_sd2": 0,
        "lf_power": 0, "hf_power": 0, "lf_hf": 0, "total_power": 0,
        "ppg_amplitude": 0, "ppg_rise_time_ms": 0, "ppg_pulse_width_ms": 0,
    }
    dummy.update({f"acc_{k}": 0 for k in ["mag_mean", "mag_std", "sma", "x_mean", "x_std", "y_mean", "y_std", "z_mean", "z_std"]})
    dummy.update({f"temp_{k}": 0 for k in ["mean", "std", "min", "max", "slope"]})
    dummy.update({f"eda_{k}":  0 for k in ["mean", "std", "min", "max", "slope"]})
    return list(dummy.keys())

FEATURE_NAMES: list[str] = _build_feature_names()


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_subject_features(subject_windows) -> np.ndarray:
    """
    Extract features for all windows of one subject.

    Parameters
    ----------
    subject_windows : SubjectWindows from signal_processing.segment_subject()

    Returns
    -------
    X : (W, D) float32 feature matrix
    """
    from tqdm import tqdm

    W = subject_windows.windows_bvp.shape[0]
    feats = np.zeros((W, len(FEATURE_NAMES)), dtype=np.float32)

    for i in tqdm(range(W), desc=f"S{subject_windows.subject_id} features", leave=False):
        feats[i] = extract_features(
            bvp  = subject_windows.windows_bvp_filt[i],  # bandpass-only for accurate peak detection
            acc  = subject_windows.windows_acc[i],
            temp = subject_windows.windows_temp[i],
            eda  = subject_windows.windows_eda[i],
        )
    return feats


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic 60-second signals at correct sample rates
    bvp_dummy  = np.sin(2 * np.pi * 1.2 * np.arange(60 * BVP_FS) / BVP_FS).astype(np.float32)
    acc_dummy  = np.random.randn(60 * ACC_FS, 3).astype(np.float32) * 0.05
    temp_dummy = (np.random.randn(60 * TEMP_FS) * 0.1 + 33.5).astype(np.float32)
    eda_dummy  = (np.random.randn(60 * EDA_FS)  * 0.5 + 2.0).astype(np.float32)

    features = extract_features(bvp_dummy, acc_dummy, temp_dummy, eda_dummy)
    print(f"Feature vector length: {len(features)}")
    print(f"\nSample features:")
    for name, val in zip(FEATURE_NAMES[:8], features[:8]):
        print(f"  {name:<25} {val:.4f}")
    print(f"  ...")
