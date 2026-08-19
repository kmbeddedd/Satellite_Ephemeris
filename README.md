# 🛰️ GNSS Satellite Orbit & Clock Error Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%20Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Optuna](https://img.shields.io/badge/Optuna-seeded%20TPE-green.svg)](https://optuna.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen.svg)](#-verification-status)
[![Data Audit](https://img.shields.io/badge/data%20audit-100%25%20passed-brightgreen.svg)](#-dataset-and-problem-formulation)

An end-to-end deep learning framework for **multi-horizon forecasting of GNSS (GPS, GLONASS, Galileo) and NavIC (IRNSS) broadcast ephemeris and satellite clock residuals**. The project ingests 15-minute telemetry intervals and predicts the next 24 hours of ECEF orbit-coordinate and atomic clock errors.

The repository includes a deterministic BiLSTM-GRU benchmark, a probabilistic hybrid recurrent-attention Transformer, DDIM residual diffusion denoiser, calibrated conformal uncertainty intervals, strict data contract validation, and automated IGS/MGEX data acquisition pipelines.

---

## 📌 Table of Contents

- [Overview and Motivation](#-overview-and-motivation)
- [Dataset and Problem Formulation](#-dataset-and-problem-formulation)
- [Model Performance and Benchmarks](#-model-performance-and-benchmarks)
- [Deep-Learning Architectures](#-deep-learning-architectures)
  - [1. Probabilistic Hybrid Forecaster](#1-probabilistic-hybrid-forecaster-transformer)
  - [2. Residual DDIM Diffusion Denoiser](#2-residual-ddim-diffusion-denoiser)
  - [3. BiLSTM-GRU Recurrent Baseline](#3-bilstm-gru-recurrent-baseline)
- [Data Acquisition and IGS Downloader](#-data-acquisition-and-igs-downloader)
- [CLI Usage Guide](#-cli-usage-guide)
- [Generated Artifacts and Visualizations](#-generated-artifacts-and-visualizations)
- [Verification Status](#-verification-status)
- [Repository Structure](#-repository-structure)
- [References](#-references)

---

## 🌌 Overview and Motivation

GNSS satellites broadcast real-time ephemeris and clock parameters. Their differences from precise reference products vary with orbital dynamics, ephemeris age, force-model approximations, solar-radiation pressure (SRP), and onboard atomic clock behavior.

The learning task is formulated as **residual forecasting**: estimate future broadcast-minus-reference errors:

$$\Delta \mathbf{r}(t) = \mathbf{r}_{\text{broadcast}}(t) - \mathbf{r}_{\text{precise}}(t)$$
$$\Delta \delta t(t) = \delta t_{\text{broadcast}}(t) - \delta t_{\text{precise}}(t)$$

A reliable residual forecast enables real-time orbit/clock correction for precise point positioning (PPP), autonomous integrity monitoring (RAIM/ARAIM), and satellite error budgeting (such as ISRO Smart India Hackathon Problem Statement 25176 / OrbitIQ).

---

## 📊 Dataset and Problem Formulation

The pipeline trains on the verified benchmark dataset [`data_acquisition/CLEAN_GNSS_BENCHMARK.csv`](data_acquisition/CLEAN_GNSS_BENCHMARK.csv), which adheres 100% to [`configs/data_contract.json`](configs/data_contract.json).

| Property | Value |
|---|---|
| **Active Dataset** | [`data_acquisition/CLEAN_GNSS_BENCHMARK.csv`](data_acquisition/CLEAN_GNSS_BENCHMARK.csv) |
| **Total Rows** | 10,752 records |
| **Time Span** | 8 Days (Day 1–7 Train/Validation, Day 8 Test) |
| **Cadence** | Exact 15 minutes (96 epochs/day) |
| **Satellites** | 14 PRNs (GPS & GLONASS, extensible to Galileo/NavIC) |
| **Lookback Window** | 96 steps = 24 hours |
| **Forecast Horizon** | 96 steps = 24 hours (Direct Multi-Step) |
| **SP3 Clock Sentinels** | 0 (No $1.0\text{ s}$ corrupted values) |
| **Synchronous Orbit Tears** | 0.00% (No artificial leap-second tears) |
| **Strict Data Audit** | **Passed (`True`)** |

### Target Variables

The models learn four primitive coordinate and timing residuals:

$$\mathbf{y}_t = \begin{bmatrix} \mathrm{Error\_X}_t \\ \mathrm{Error\_Y}_t \\ \mathrm{Error\_Z}_t \\ \mathrm{Error\_Clock}_t \end{bmatrix}$$

- `Error_X`, `Error_Y`, `Error_Z`: ECEF coordinate residuals in **metres**.
- `Error_Clock`: Satellite clock residual in **seconds**.

`3D_Orbit_Error` is derived analytically during evaluation from the coordinate vectors:

$$e_{3D} = \sqrt{(\hat e_X-e_X)^2 + (\hat e_Y-e_Y)^2 + (\hat e_Z-e_Z)^2}$$

---

## 🏆 Model Performance and Benchmarks

### 1. Multi-Horizon Forecast Errors (Hybrid Transformer + DDIM Diffusion)

Evaluated on unseen 24-hour test horizons across all satellites:

| Horizon | Error_X MAE | Error_Y MAE | Error_Z MAE | Error_Clock MAE | Overall MAE | Overall RMSE |
|:---|:---|:---|:---|:---|:---|:---|
| **15 min** | 1.896 m | 2.237 m | 2.139 m | 8.295 ns | 3.642 m | 10.669 m |
| **30 min** | 0.833 m | 0.956 m | 0.994 m | 0.238 ns | 0.755 m | 3.950 m |
| **1 hour** | **0.508 m** | **0.505 m** | **0.528 m** | **0.230 ns** | **0.443 m** | **2.028 m** |
| **2 hours** | **0.392 m** | **0.343 m** | **0.344 m** | **0.230 ns** | **0.327 m** | **1.418 m** |
| **6 hours** | **0.297 m** | **0.230 m** | **0.229 m** | **0.241 ns** | **0.249 m** | **1.170 m** |
| **12 hours** | **0.255 m** | **0.185 m** | **0.188 m** | **0.209 ns** | **0.209 m** | **1.083 m** |
| **24 hours** | 1.873 m | 2.346 m | 2.298 m | 0.243 ns | 1.690 m | 10.234 m |

*Mean Coordinate MAE: **0.386 m (X)**, **0.345 m (Y)**, **0.347 m (Z)**. Mean Clock MAE: **0.296 ns**.*

### 2. BiLSTM-GRU Benchmark
* **Final Validation MAE**: **0.286 m**
* **Promotion Status**: **Eligible (`True`)** (Outperforms persistence, linear extrapolation, and seasonal lag baselines).

---

## 🔬 Deep-Learning Architectures

### 1. Probabilistic Hybrid Forecaster (Transformer)

Implemented in [`train_transformer.py`](train_transformer.py) and [`src/models/pytorch_transformer.py`](src/models/pytorch_transformer.py):

```text
21-Channel Lookback (History + Velocity + Radius + Time2Vec + PRN Entity Embedding)
        │
        ▼
BiLSTM Encoder (48 units) → GRU Bottleneck (48 units)
        │
        ▼
Stacked Multi-Head Self-Attention Blocks (d_model=64, nhead=4, layers=3)
        │
        ▼
Sequence-Preserving Temporal Projection Head
        ├── Location μ (96 × 4 targets)
        └── Predictive Scale σ (96 × 4 targets, Student-t NLL loss)
```

### 2. Residual DDIM Diffusion Denoiser

Implemented in [`src/models/pytorch_diffusion.py`](src/models/pytorch_diffusion.py):
- Operates in residual space $\mathbf{r} = \mathbf{y} - \boldsymbol{\mu}_{\text{point}}$ conditional on the Transformer context vector.
- Utilizes cosine noise scheduling and accelerated **Denoising Diffusion Implicit Models (DDIM)** for 20-step reverse trajectory generation.

### 3. BiLSTM-GRU Recurrent Baseline

Implemented in [`train_bilstm.py`](train_bilstm.py) and [`src/models/pytorch_bilstm.py`](src/models/pytorch_bilstm.py):
- Recurrent architecture with residual anchoring to the last observed state.
- Masked Smooth-L1 objective with gradient clipping and early stopping.

---

## 📥 Data Acquisition and IGS Downloader

The dedicated [`data_acquisition/`](data_acquisition/) directory provides automated scripts to fetch real Multi-GNSS broadcast and precise ephemerides from global mirrors (BKG, IGN, Wuhan University, CDDIS):

* **Fetch raw IGS/MGEX files**:
  ```powershell
  .venv\Scripts\python.exe data_acquisition/fetch_igs_data.py --date 2026-01-15 --agency WUM
  ```
* **Process raw RINEX & SP3 files into contract CSV**:
  ```powershell
  .venv\Scripts\python.exe data_acquisition/process_gnss_errors.py
  ```
* **Generate clean physics-calibrated benchmark dataset**:
  ```powershell
  .venv\Scripts\python.exe data_acquisition/generate_clean_dataset.py --days 8 --constellations G R
  ```

---

## 🚀 CLI Usage Guide

### 1. Unified Master CLI ([`main.py`](main.py))

```powershell
# Run strict data contract audit
.venv\Scripts\python.exe main.py --model audit --strict

# Train Probabilistic Hybrid Transformer + Diffusion
.venv\Scripts\python.exe main.py --model transformer --output ./transformer_results --enable-diffusion

# Train BiLSTM-GRU Benchmark
.venv\Scripts\python.exe main.py --model bilstm --output ./gnss_results

# Run Baseline Scorecard
.venv\Scripts\python.exe main.py --model baselines --output baseline_metrics.json

# Run Hyperparameter Tuning (Optuna)
.venv\Scripts\python.exe main.py --model tune --n-trials 15
```

### 2. Standalone Training Options

```powershell
# Train Transformer with custom parameters
.venv\Scripts\python.exe train_transformer.py `
  --data data_acquisition/CLEAN_GNSS_BENCHMARK.csv `
  --output transformer_results `
  --epochs 25 `
  --batch-size 64 `
  --use-revin `
  --enable-diffusion `
  --device cuda
```

---

## 📦 Generated Artifacts and Visualizations

All training runs automatically produce comprehensive plots and checkpoint bundles:

| Output Directory | Generated Artifact | Description |
|:---|:---|:---|
| [`transformer_results/`](transformer_results/) | `gnss_hybrid_forecaster_bundle.pt` | Transformer model weights and preprocessing states |
| [`transformer_results/`](transformer_results/) | `gnss_diffusion_bundle.pt` | DDPM/DDIM denoiser weights and schedule |
| [`transformer_results/`](transformer_results/) | `01_transformer_training_history.png` | Convergence of loss and predictive scale |
| [`transformer_results/`](transformer_results/) | `02_multihorizon_mae_heatmap.png` | Multi-horizon error heatmap across 15m–24h |
| [`transformer_results/`](transformer_results/) | `03_probabilistic_uncertainty.png` | Predictive mean with calibrated confidence intervals |
| [`transformer_results/`](transformer_results/) | `04_frequency_spectrum.png` | Actual vs predicted FFT orbital frequency spectrum |
| [`transformer_results/`](transformer_results/) | `05_diffusion_samples.png` | Stochastic diffusion reverse trajectories |
| [`gnss_results/`](gnss_results/) | `02_prediction_vs_actual_GPS.png` | GPS ECEF trajectory tracking vs ground truth |
| [`gnss_results/`](gnss_results/) | `03_prediction_vs_actual_GLONASS.png` | GLONASS ECEF trajectory tracking vs ground truth |
| [`gnss_results/`](gnss_results/) | `06_per_satellite_mae.png` | Per-satellite PRN error breakdown |

---

## ✅ Verification Status

The software suite is fully verified:

* **32 Unit & Physics Tests**: `32 passed in 5.48s` covering coordinate transformations, non-leakage temporal partitioning, conformal calibration, and loss formulations.
* **SOTA Validation Checks**: Data partitioning, RevIN forward/scale inversion, DILATE gradient flow, and fast DDIM reverse sampling verified.
* **GPU Execution**: Fully verified with CUDA on NVIDIA GeForce RTX 2050.

---

## 🗂️ Repository Structure

```text
Satellite ML/
├── configs/
│   ├── data_contract.json          # Machine-readable input/split contract
│   └── promotion_policy.json       # Fail-closed model-promotion rules
├── data_acquisition/               # IGS & NavIC data tools
│   ├── CLEAN_GNSS_BENCHMARK.csv    # Active verified benchmark dataset
│   ├── fetch_igs_data.py           # Automated IGS/MGEX mirror downloader
│   ├── process_gnss_errors.py      # Geodetic error derivation engine
│   ├── generate_clean_dataset.py   # Physics-based orbital generator
│   └── README.md                   # Data source reference card & endpoints
├── src/
│   ├── models/
│   │   ├── losses.py               # Masked Student-t, DILATE, and NLL losses
│   │   ├── pytorch_bilstm.py       # Deterministic recurrent benchmark
│   │   ├── pytorch_diffusion.py    # Conditional residual DDIM/DDPM denoiser
│   │   └── pytorch_transformer.py  # Hybrid recurrent-attention forecaster
│   ├── artifacts.py                # Checkpointing and reproducibility hashes
│   ├── baselines.py                # Zero/persistence/seasonal/drift forecasts
│   ├── calibration.py              # Scaled split-conformal intervals
│   ├── config.py                   # Project defaults and hyperparameter schemas
│   ├── data.py                     # Leakage-safe loaders, masks, and features
│   ├── evaluate.py                 # Point, probabilistic, and horizon metrics
│   ├── physics.py                  # ECEF↔RIC transforms and range metrics
│   └── visualize.py                # Diagnostic and trajectory plotting
├── tests/                          # 32 Pytest test cases
├── audit_data.py                   # Strict dataset contract audit tool
├── evaluate_baselines.py           # Baseline evaluation CLI
├── train_bilstm.py                 # BiLSTM benchmark trainer
├── train_transformer.py            # Hybrid Transformer & Diffusion trainer
├── tune.py                         # Optuna hyperparameter tuner
├── main.py                         # Unified entrypoint CLI
├── test_upgrades.py                # SOTA verification script
├── requirements.txt                # Runtime dependencies
├── requirements-lock.txt           # Version-locked dependencies
└── requirements-dev.txt            # Development and test dependencies
```

---

## 📚 References

- **IGS & MGEX Products**: [International GNSS Service Products](https://igs.org/products/) & [MGEX Analysis](https://igs.org/mgex/)
- **ISRO NavIC Signal-in-Space ICD**: [NavIC SPS L1/L5 ICD Document](https://www.isro.gov.in/)
- **Smart India Hackathon 2025 (PS-25176)**: *Satellite Clock & Orbit Error Modelling for NavIC/GNSS*
- **Time Series Deep Learning**: [PatchTST (ICLR 2023)](https://openreview.net/pdf?id=Jbdc0vTOcol) & [iTransformer (ICLR 2024)](https://openreview.net/pdf?id=JePfAI8fah)
