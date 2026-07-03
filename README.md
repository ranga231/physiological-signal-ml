# Physiological Stress Detection from Wrist-Worn Sensors

End-to-end ML pipeline comparing Random Forests, 1D CNNs, and Transformers for stress detection using the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) (15 subjects, Empatica E4 wrist device).

**Paper:** *Physiological Stress Detection from Wrist-Worn Sensors: Comparing Random Forests, 1D CNNs, and Transformers with Systematic Channel Ablation on WESAD* — [arXiv preprint](https://arxiv.org) (link to be added)

---

## Results (LOSO-CV, 15 subjects)

All models use the same 6-channel input (BVP, ACC xyz, TEMP, EDA) and identical Leave-One-Subject-Out Cross-Validation splits.

| Model | Mean F1 | Std | AUC-ROC |
|---|---|---|---|
| Random Forest (34 handcrafted features) | 0.831 | 0.292 | 0.953 |
| 1D CNN (raw signals, 6ch) | 0.882 | 0.240 | 0.956 |
| Patch-based Transformer (raw signals, 6ch) | 0.893 | 0.211 | — |

### CNN Channel Ablation

| Configuration | Channels | Mean F1 | Std |
|---|---|---|---|
| BVP only | 1 | 0.737 | 0.094 |
| BVP + ACC | 4 | 0.589 | 0.175 |
| BVP + ACC + TEMP | 5 | 0.599 | 0.212 |
| BVP + EDA | 2 | 0.857 | 0.179 |
| BVP + ACC + EDA | 5 | 0.884 | 0.154 |
| BVP + ACC + TEMP + EDA | 6 | 0.878 | 0.232 |

**Key finding:** EDA is the dominant modality (+0.295 F1 over BVP+ACC baseline). A 2-channel BVP+EDA configuration achieves 0.857 F1 — within 2 points of the full 6-channel model. ACC *decreases* performance in this sedentary protocol (TSST is conducted seated).

### RF Feature Group Ablation

| Feature Group | Features | Mean F1 |
|---|---|---|
| EDA only | 5 | 0.774 |
| HRV only | 15 | 0.622 |
| HRV + EDA | 20 | 0.829 |
| All features | 34 | 0.831 |

EDA dominance is architecture-independent: 5 simple EDA statistics outperform 15 HRV features in the RF as well.

---

## Dataset

**WESAD** — 15 subjects (S2–S17, S12 missing) wearing an Empatica E4 during:
- Baseline (sitting quietly)
- TSST stress protocol (public speaking + mental arithmetic in front of evaluators)
- Amusement (Stroop task)

**Binary task:** stress (TSST) vs. non-stress (baseline). 60 s windows, 30 s stride.

**Wrist signals:**

| Signal | Rate | Description |
|---|---|---|
| BVP (PPG) | 64 Hz | Blood volume pulse — source for HRV |
| ACC | 32 Hz | 3-axis accelerometer |
| EDA | 4 Hz | Electrodermal activity |
| TEMP | 4 Hz | Skin temperature |

Data access: request from the [WESAD project page](https://uni-siegen.de/labs/sigproc/redmine/projects/wesad). Data is not included in this repository.

---

## Project Structure

```
physiological-signal-ml/
├── run_pipeline.py              # RF / CNN / Transformer — full LOSO-CV
├── run_ablation.py              # CNN channel ablation (6 configurations)
├── run_ablation_bvp_eda.py      # BVP + EDA only (2-channel CNN)
├── run_ablation_rf.py           # RF feature group ablation (10 configurations)
├── requirements.txt
├── .gitignore
├── data/                        # Not included — download from WESAD project
│   └── raw/S2/S2.pkl ...
└── src/
    ├── ingestion/
    │   └── wesad_loader.py          # WESAD .pkl → WESADSubject objects
    ├── preprocessing/
    │   └── signal_processing.py     # Filtering, artifact removal, windowing
    ├── features/
    │   └── hrv_features.py          # 34 handcrafted features (HRV, ACC, EDA, TEMP)
    ├── models/
    │   ├── baseline_rf.py           # RandomForest pipeline (sklearn)
    │   ├── cnn_classifier.py        # 1D CNN (PyTorch)
    │   └── transformer_classifier.py # Patch-based Transformer (PyTorch)
    └── evaluation/
        └── loso_cv.py               # LOSO-CV engine for all model types
```

---

## Setup

```bash
git clone https://github.com/ranga231/physiological-signal-ml
cd physiological-signal-ml
pip install -r requirements.txt

# Download WESAD data, extract so structure is:
# data/raw/S2/S2.pkl, data/raw/S3/S3.pkl, ...

# Run RF
python run_pipeline.py --model rf

# Run CNN (6 channels)
python run_pipeline.py --model cnn --epochs 30

# Run Transformer
python run_pipeline.py --model transformer --epochs 40

# Run all three
python run_pipeline.py --model all

# Channel ablation (CNN)
python run_ablation.py

# RF feature group ablation
python run_ablation_rf.py

# Quick test on subset of subjects
python run_pipeline.py --model rf --subjects 2 3 4 5 6
```

---

## Models

### Random Forest
- 34 handcrafted features: HRV (time-domain, Poincaré, Lomb-Scargle frequency-domain, PPG morphology), ACC (magnitude, per-axis), EDA (5 statistics), TEMP (5 statistics)
- 300 trees, balanced class weights, NaN imputation with training-set means

### 1D CNN
- 3 ConvBlocks (Conv1d → BatchNorm → ReLU → MaxPool), 32→64→128 filters
- Classifier: AdaptiveAvgPool1d → Dropout(0.5) → Linear(128, 2)
- ~140K parameters

### Patch-based Transformer
- Patches of 64 samples (1 s at 64 Hz ≈ one cardiac cycle)
- CLS token, learnable positional encoding, 3 Pre-LayerNorm encoder layers
- d_model=64, 4 heads, d_ff=128, dropout=0.3
- ~180K parameters

All models trained with class-weighted cross-entropy, AdamW, cosine LR annealing, early stopping on validation F1.

---

## Key Design Decisions

**LOSO-CV** — one subject entirely held out per fold, with no data leakage. This is the correct protocol for generalizable wearable AI and standard for wearable dataset benchmarks.

**EDA dominance** — confirmed across both CNN and RF architectures, ruling out a modeling artifact. Five simple EDA statistics (mean, std, min, max, slope) carry more stress-discriminative information than 15 HRV features.

**ACC as context, not signal** — ACC degrades performance in sedentary stress (TSST is conducted seated). In free-living deployment, ACC serves as an activity gate: suppress stress classification during exercise when thermoregulatory EDA confounds the stress signal.

---

## Citation

If you use this code or findings, please cite:

```
@article{narayanaswami2025wesad,
  title={Physiological Stress Detection from Wrist-Worn Sensors:
         Comparing Random Forests, 1D CNNs, and Transformers
         with Systematic Channel Ablation on WESAD},
  author={Narayanaswami, Ranga},
  journal={arXiv preprint},
  year={2025}
}
```

---

## References

- Schmidt et al. (2018). *Introducing WESAD, a multimodal dataset for wearable stress and affect detection.* ICMI 2018.
- Dosovitskiy et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR 2021.
- Nie et al. (2023). *A time series is worth 64 words.* ICLR 2023.
