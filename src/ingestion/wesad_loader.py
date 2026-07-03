"""
WESAD Dataset Loader
====================
Loads the WESAD wrist-worn physiological signal dataset.

Dataset: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
Reference: Schmidt et al. (2018), ICMI

Wrist sensors:
  - BVP (Blood Volume Pulse / PPG): 64 Hz
  - ACC (Accelerometer, 3-axis): 32 Hz
  - TEMP (Skin temperature): 4 Hz
  - EDA (Electrodermal activity): 4 Hz

Labels (sampled at 700 Hz, aligned to signal windows):
  0 = not defined / transient
  1 = baseline
  2 = stress (TSST protocol)
  3 = amusement (funny video clips)
  4 = meditation

Binary task: stress (2) vs. non-stress (1, 3, 4)
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Sampling rates ────────────────────────────────────────────────────────────
FS = {
    "BVP":  64,
    "ACC":  32,
    "TEMP":  4,
    "EDA":   4,
}
LABEL_FS = 700  # label stream native rate

# All 14 available subjects (S12 is missing from the dataset)
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Label mapping
LABEL_NAMES = {0: "undefined", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
BINARY_MAP  = {1: 0, 2: 1, 3: 0, 4: 0}   # 0 = non-stress, 1 = stress


class WESADSubject:
    """Container for one subject's wrist signals and labels."""

    def __init__(self, subject_id: int, data_root: str | Path):
        self.sid = subject_id
        self._load(Path(data_root))

    def _load(self, root: Path) -> None:
        pkl_path = root / f"S{self.sid}" / f"S{self.sid}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Expected WESAD file at {pkl_path}.\n"
                "Download from: https://uni-siegen.de/labs/sigproc/redmine/projects/wesad/wiki"
            )

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        wrist = raw["signal"]["wrist"]

        # Raw signals — shape (N, channels) for ACC, (N,) for scalars
        self.bvp  = wrist["BVP"].flatten().astype(np.float32)   # (N,)
        self.acc  = wrist["ACC"].astype(np.float32) / 64.0      # (N, 3)
        self.temp = wrist["TEMP"].flatten().astype(np.float32)   # (N,)
        self.eda  = wrist["EDA"].flatten().astype(np.float32)    # (N,)

        # Labels: downsample from 700 Hz → each signal's native rate
        raw_labels = raw["label"].flatten().astype(np.int8)

        self.labels: dict[str, np.ndarray] = {
            sig: self._align_labels(raw_labels, fs)
            for sig, fs in FS.items()
        }

        # Convenience: label at BVP rate (most commonly used downstream)
        self.label = self.labels["BVP"]

    def _align_labels(self, raw_labels: np.ndarray, target_fs: int) -> np.ndarray:
        """Downsample label stream from LABEL_FS to target_fs by majority vote in each block."""
        ratio = LABEL_FS // target_fs
        n_blocks = len(raw_labels) // ratio
        trimmed = raw_labels[: n_blocks * ratio].reshape(n_blocks, ratio)
        # Majority vote per block; ties → 0 (undefined)
        aligned = np.apply_along_axis(
            lambda row: np.bincount(row.clip(0), minlength=5).argmax(), axis=1, arr=trimmed
        ).astype(np.int8)
        return aligned[: len(self.bvp)]   # clip to signal length

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        label_counts = pd.Series(self.label).value_counts().sort_index()
        rows = []
        for lbl, cnt in label_counts.items():
            rows.append({
                "label_id":   lbl,
                "label_name": LABEL_NAMES.get(lbl, "unknown"),
                "n_samples":  cnt,
                "duration_s": cnt / FS["BVP"],
            })
        df = pd.DataFrame(rows)
        df.insert(0, "subject", self.sid)
        return df

    def binary_mask(self) -> np.ndarray:
        """Boolean mask selecting only labelled (non-zero) samples."""
        return self.label > 0

    def binary_labels(self) -> np.ndarray:
        """0 = non-stress, 1 = stress; undefined samples mapped to -1."""
        out = np.full(len(self.label), -1, dtype=np.int8)
        for orig, mapped in BINARY_MAP.items():
            out[self.label == orig] = mapped
        return out


class WESADDataset:
    """
    Load and iterate over multiple WESAD subjects.

    Usage
    -----
    >>> ds = WESADDataset("data/raw")
    >>> for subject in ds:
    ...     print(subject.sid, subject.bvp.shape)
    """

    def __init__(
        self,
        data_root: str | Path,
        subject_ids: Optional[list[int]] = None,
    ):
        self.root = Path(data_root)
        self.subject_ids = subject_ids or ALL_SUBJECTS
        self._subjects: dict[int, WESADSubject] = {}

    def load_subject(self, sid: int) -> WESADSubject:
        if sid not in self._subjects:
            self._subjects[sid] = WESADSubject(sid, self.root)
        return self._subjects[sid]

    def load_all(self, verbose: bool = True) -> list[WESADSubject]:
        subjects = []
        for sid in self.subject_ids:
            if verbose:
                print(f"  Loading S{sid}...", end=" ", flush=True)
            try:
                s = self.load_subject(sid)
                subjects.append(s)
                if verbose:
                    n_stress = (s.label == 2).sum()
                    print(f"OK  ({n_stress / FS['BVP']:.0f}s stress)")
            except FileNotFoundError as e:
                if verbose:
                    print(f"SKIPPED ({e})")
        return subjects

    def __iter__(self):
        return iter(self.load_all(verbose=False))

    def __len__(self):
        return len(self.subject_ids)

    def dataset_summary(self) -> pd.DataFrame:
        frames = [s.summary() for s in self]
        return pd.concat(frames, ignore_index=True)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    print(f"\nLoading WESAD from: {data_root}\n")
    ds = WESADDataset(data_root)
    subjects = ds.load_all(verbose=True)

    if subjects:
        df = ds.dataset_summary()
        print("\n── Dataset Summary ──────────────────────────────────────")
        print(df.pivot_table(index="label_name", values=["n_samples", "duration_s"],
                             aggfunc="sum").to_string())
        print(f"\nTotal subjects loaded: {len(subjects)}")
