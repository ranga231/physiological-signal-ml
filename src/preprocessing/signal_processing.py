"""
Signal Preprocessing Pipeline
==============================
Covers the key steps between raw sensor data and features/model input:

1. Bandpass filter (PPG / BVP)
2. Motion artifact detection via accelerometer magnitude
3. Per-subject z-score normalization
4. Sliding-window segmentation
5. Label assignment per window (majority vote)

Design notes
------------
- All filtering is done with zero-phase (filtfilt) Butterworth filters to
  avoid phase distortion — critical when downstream HRV features depend on
  accurate peak timing.
- Motion artifact removal gates windows where ACC RMS exceeds a threshold,
  matching the approach used in actigraphy-class wearable algorithms.
- Windows with > 50% undefined labels (label 0) are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ── Filter design ─────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Default passband 0.5–4.0 Hz passes PPG fundamental + harmonics for HR
    range 30–240 bpm while rejecting baseline wander and high-frequency noise.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """60/50 Hz powerline notch filter (optional, mainly useful for ECG/EDA)."""
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, quality)
    return filtfilt(b, a, signal).astype(np.float32)


def lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 1.0,
    order: int = 4,
) -> np.ndarray:
    """Low-pass filter — used for TEMP / EDA smoothing."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


# ── Motion artifact detection ─────────────────────────────────────────────────

def acc_rms(acc: np.ndarray) -> np.ndarray:
    """
    Compute per-sample ACC vector magnitude (RMS across 3 axes).
    acc: (N, 3)  →  returns (N,)
    """
    return np.sqrt((acc ** 2).sum(axis=1)).astype(np.float32)


def motion_artifact_mask(
    acc: np.ndarray,
    fs_acc: float,
    window_s: float,
    step_s: float,
    threshold: float = 0.15,
) -> np.ndarray:
    """
    Produce a boolean array (one entry per window) indicating whether the
    window is likely corrupted by motion artifact.

    A window is flagged if the ACC RMS (after removing gravity / DC component)
    exceeds `threshold` g for more than 20% of the window duration.

    Parameters
    ----------
    acc        : (N, 3) raw accelerometer array in g g (E4 raw counts must be pre-divided by 64)
    fs_acc     : ACC sampling rate (Hz)
    window_s   : window length in seconds (must match BVP window)
    step_s     : step size in seconds
    threshold  : RMS threshold in g above which motion is declared

    Returns
    -------
    artifact_mask : bool array, True = motion artifact present
    """
    # Remove gravity (DC) by high-pass at 0.5 Hz
    nyq = fs_acc / 2.0
    b, a = butter(2, 0.5 / nyq, btype="high")
    acc_hp = filtfilt(b, a, acc, axis=0)
    rms = acc_rms(acc_hp)  # (N,)

    n_win = int(window_s * fs_acc)
    n_step = int(step_s * fs_acc)
    n_windows = (len(rms) - n_win) // n_step + 1

    artifact = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * n_step
        chunk = rms[start : start + n_win]
        artifact[i] = (chunk > threshold).mean() > 0.20   # >20% samples flagged
    return artifact


# ── Normalization ─────────────────────────────────────────────────────────────

def zscore_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-signal (per-subject) z-score normalization."""
    mu = signal.mean()
    sigma = signal.std()
    return ((signal - mu) / (sigma + eps)).astype(np.float32)


# ── Windowing ─────────────────────────────────────────────────────────────────

@dataclass
class WindowConfig:
    window_s: float = 60.0    # window length in seconds
    step_s:   float = 30.0    # step / hop size (50% overlap by default)
    min_valid_label_frac: float = 0.80  # discard window if < 80% samples are labelled


@dataclass
class SubjectWindows:
    """Output of segment_subject() — one window set per subject."""
    subject_id: int
    windows_bvp:      np.ndarray   # (W, N_bvp)   z-scored BVP windows (for CNN)
    windows_bvp_filt: np.ndarray   # (W, N_bvp)   bandpass-only BVP (for HRV peak detection)
    windows_acc:      np.ndarray   # (W, N_acc, 3) raw ACC windows
    windows_temp:     np.ndarray   # (W, N_temp)   raw TEMP windows
    windows_eda:      np.ndarray   # (W, N_eda)    raw EDA windows
    labels:           np.ndarray   # (W,)          binary label per window
    artifact_mask:    np.ndarray   # (W,)          True = motion artifact


def segment_subject(
    subject,                   # WESADSubject instance
    cfg: WindowConfig = WindowConfig(),
) -> SubjectWindows:
    """
    Preprocess and window one subject's wrist data.

    Steps per signal
    ----------------
    BVP  : bandpass 0.5–4 Hz → z-score → window
    ACC  : z-score (per axis) → window; also compute artifact mask
    TEMP : lowpass 1 Hz → z-score → window
    EDA  : lowpass 1 Hz → z-score → window

    Labels are derived from the BVP-rate label stream by majority vote per
    window; windows with insufficient labelled samples are dropped.
    """
    from ..ingestion.wesad_loader import FS, BINARY_MAP

    # ── Preprocess signals ────────────────────────────────────────────────────
    bvp_filt = bandpass_filter(subject.bvp, fs=FS["BVP"])
    bvp_norm = zscore_normalize(bvp_filt)

    acc_norm = np.stack(
        [zscore_normalize(subject.acc[:, ax]) for ax in range(3)], axis=1
    )

    temp_filt = lowpass_filter(subject.temp, fs=FS["TEMP"])
    temp_norm = zscore_normalize(temp_filt)

    eda_filt  = lowpass_filter(subject.eda, fs=FS["EDA"])
    eda_norm  = zscore_normalize(eda_filt)

    # ── Window sizes in samples ───────────────────────────────────────────────
    def n_samp(fs):    return int(cfg.window_s * fs)
    def n_step(fs):    return int(cfg.step_s   * fs)

    n_bvp  = n_samp(FS["BVP"]);  s_bvp  = n_step(FS["BVP"])
    n_acc  = n_samp(FS["ACC"]);  s_acc  = n_step(FS["ACC"])
    n_temp = n_samp(FS["TEMP"]); s_temp = n_step(FS["TEMP"])
    n_eda  = n_samp(FS["EDA"]);  s_eda  = n_step(FS["EDA"])

    # Number of windows determined by shortest signal
    n_windows = min(
        (len(bvp_norm)  - n_bvp)  // s_bvp  + 1,
        (len(acc_norm)  - n_acc)  // s_acc  + 1,
        (len(temp_norm) - n_temp) // s_temp + 1,
        (len(eda_norm)  - n_eda)  // s_eda  + 1,
    )

    # ── Slice windows ─────────────────────────────────────────────────────────
    def slice_windows(arr, n, step, n_win, one_d=True):
        wins = []
        for i in range(n_win):
            chunk = arr[i * step : i * step + n]
            wins.append(chunk)
        return np.array(wins)  # (W, N) or (W, N, C)

    win_bvp      = slice_windows(bvp_norm,  n_bvp,  s_bvp,  n_windows)   # z-scored → CNN
    win_bvp_filt = slice_windows(bvp_filt,  n_bvp,  s_bvp,  n_windows)   # filtered only → HRV
    win_acc  = slice_windows(acc_norm,  n_acc,  s_acc,  n_windows, one_d=False)
    win_temp = slice_windows(temp_norm, n_temp, s_temp, n_windows)
    win_eda  = slice_windows(eda_norm,  n_eda,  s_eda,  n_windows)

    # ── Labels: majority vote per BVP-rate window ─────────────────────────────
    lbl_stream = subject.label  # at BVP rate
    win_labels = np.full(n_windows, -1, dtype=np.int8)
    valid_mask = np.ones(n_windows, dtype=bool)

    for i in range(n_windows):
        chunk = lbl_stream[i * s_bvp : i * s_bvp + n_bvp]
        labelled_frac = (chunk > 0).mean()
        if labelled_frac < cfg.min_valid_label_frac:
            valid_mask[i] = False
            continue
        labelled = chunk[chunk > 0]
        majority = np.bincount(labelled, minlength=5).argmax()
        if majority == 0:
            valid_mask[i] = False
            continue
        binary = BINARY_MAP.get(majority, -1)
        if binary < 0:
            valid_mask[i] = False
            continue
        win_labels[i] = binary

    # ── Motion artifact mask ──────────────────────────────────────────────────
    art_mask = motion_artifact_mask(
        subject.acc, FS["ACC"], cfg.window_s, cfg.step_s
    )
    # Align length (may differ by 1 due to rounding)
    art_mask = art_mask[:n_windows]
    if len(art_mask) < n_windows:
        art_mask = np.pad(art_mask, (0, n_windows - len(art_mask)), constant_values=False)

    # Apply valid_mask (keep structure; caller decides whether to remove artifacts)
    return SubjectWindows(
        subject_id       = subject.sid,
        windows_bvp      = win_bvp[valid_mask],
        windows_bvp_filt = win_bvp_filt[valid_mask],
        windows_acc      = win_acc[valid_mask],
        windows_temp     = win_temp[valid_mask],
        windows_eda      = win_eda[valid_mask],
        labels           = win_labels[valid_mask],
        artifact_mask    = art_mask[valid_mask],
    )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
    from src.ingestion.wesad_loader import WESADSubject

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    s = WESADSubject(2, data_root)
    cfg = WindowConfig(window_s=60, step_s=30)
    sw = segment_subject(s, cfg)

    print(f"S{s.sid} → {sw.windows_bvp.shape[0]} windows")
    print(f"  BVP  shape : {sw.windows_bvp.shape}")
    print(f"  ACC  shape : {sw.windows_acc.shape}")
    print(f"  Labels     : stress={( sw.labels == 1).sum()}  non-stress={(sw.labels == 0).sum()}")
    print(f"  Artifacts  : {sw.artifact_mask.sum()} windows flagged")
