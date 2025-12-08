# P300 ERP Classification — From-Scratch PyTorch SVM

## Overview

Minimal, reproducible pipeline to classify **P300 ERP** from **MOABB BI2013a** using a **from-scratch Linear SVM in PyTorch** (no `torch.nn`, no `torch.optim`).
We fetch data, preprocess (ERP-friendly), extract compact features, train an SVM with manual SGD, tune a threshold on validation, and report test metrics.

## Pipeline

1. **Load** BI2013a (3 subjects recommended).
2. **Preprocess**: Butterworth **band-pass 0.1–15 Hz** (SOS, zero-phase) → **baseline** (−0.2–0 s) → **post-stimulus window** (0.30–0.60 s) → (optional) **decimate** to **128 Hz**.
3. **Features**: split window into `n_bins` (e.g., 6); per **channel × bin** compute **mean / max / min / slope** → vector size `C * n_bins * 4`.
4. **Split** train/val/test (stratified or subject-wise).
5. **Standardize** (fit on train only).
6. **Train SVM** (hinge + L2) with manual SGD (autograd only).
7. **Tune threshold** on val (maximize **F1**).
8. **Evaluate** on test: **Acc**, **Balanced Acc**, **Precision/Recall/F1**, **ROC-AUC**, **Confusion Matrix**.

## Model (from scratch)

* Parameters: `W, b` as tensors with `requires_grad=True`.
* Loss: `C * mean(max(0, 1 − y*score)) + 0.5*reg*||W||²`.
* Update: `with torch.no_grad(): W -= lr*W.grad; b -= lr*b.grad`.

## How to Run (Colab)

1. Install deps (NumPy, SciPy, MNE, MOABB, scikit-learn, **PyTorch**).
2. Mount Google Drive; set `MNE_DATA`, `MOABB_DATA_PATH`; symlink `/root/mne_data` to Drive.
3. Load data via MOABB; run preprocessing → features.
4. Split, scale, train SVM, tune threshold, evaluate.

## Notes

* Keep **target_fs ≥ 128 Hz** for ERP 0.1–15 Hz.
* Use **zero-phase** filtering to preserve latency.
* Consider **class weighting** if imbalance hurts F1.

## License

Educational use; respect dataset/MOABB/MNE licenses.
