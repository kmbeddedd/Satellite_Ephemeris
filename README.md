# 🛰️ GNSS Satellite Orbit & Clock Error Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-green.svg)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework for **multi-horizon forecasting of GNSS broadcast ephemeris and satellite clock errors**. The system processes 15-minute telemetry across GPS and GLONASS constellations to predict Day-8 orbital errors ($X, Y, Z$, 3D Euclidean magnitude) and atomic clock drift over multi-step horizons (15 min to 24 hours).

---

## 📌 Table of Contents

- [Overview & Motivation](#-overview--motivation)
- [Dataset & Problem Formulation](#-dataset--problem-formulation)
- [Modular Python Script Architecture](#-modular-python-script-architecture)
- [Architecture & Methodology](#-architecture--methodology)
  - [1. Production BiLSTM + GRU Forecaster (`train_bilstm.py`)](#1-production-bilstm--gru-forecaster-train_bilstmpy)
  - [2. Deep Transformer & Diffusion Forecaster (`train_transformer.py`)](#2-deep-transformer--diffusion-forecaster-train_transformerpy)
  - [3. Bayesian Hyperparameter Tuning (`tune.py`)](#3-bayesian-hyperparameter-tuning-tunepy)
- [Performance & Benchmark Results](#-performance--benchmark-results)
- [Repository Structure](#-repository-structure)
- [Installation & Virtual Environment Setup](#-installation--virtual-environment-setup)
- [CLI Usage Guide](#-cli-usage-guide)
  - [1. Unified CLI Runner (`main.py`)](#1-unified-cli-runner-mainpy)
  - [2. Train BiLSTM + GRU Pipeline (`train_bilstm.py`)](#2-train-bilstm--gru-pipeline-train_bilstmpy)
  - [3. Train Transformer & Diffusion Pipeline (`train_transformer.py`)](#3-train-transformer--diffusion-pipeline-train_transformerpy)
  - [4. Hyperparameter Optimization (`tune.py`)](#4-hyperparameter-optimization-tunepy)
- [Generated Visualizations & Diagnostics](#-generated-visualizations--diagnostics)
- [License](#-license)

---

## 🌌 Overview & Motivation

Global Navigation Satellite Systems (GNSS) such as **GPS** and **GLONASS** broadcast real-time ephemeris and clock correction parameters. However, broadcast data exhibits secular drift and deviations from actual orbits due to:
- Gravitational perturbations (non-spherical Earth geopotential, lunar/solar third-body effects).
- Solar radiation pressure (SRP) and atmospheric drag.
- Relativistic effects and onboard atomic clock frequency instability.

Accurate multi-step forecasting of these error vectors directly enhances **Precise Point Positioning (PPP)**, autonomous satellite navigation, and receiver tracking performance during communication blackouts.

---

## 📊 Dataset & Problem Formulation

The dataset (`FINAL_Data.csv`) contains continuous 8-day satellite tracking telemetry at **15-minute intervals** (96 epochs per day, 768 timesteps per complete satellite).

| Parameter | Specification |
| :--- | :--- |
| **Observation Cadence** | 15 minutes ($96 \text{ steps/day}$) |
| **Time Span** | 8 days (Days 1-7: Training / Validation, Day 8: Out-of-sample Test) |
| **Constellations** | GPS (PRNs starting with `G`) and GLONASS (PRNs starting with `R`) |
| **Look-back Window ($L$)** | $96\text{ steps} = 24\text{ hours}$ |
| **Forecast Horizon ($H$)** | Direct multi-step prediction of $96\text{ steps} = 24\text{ hours}$ ahead |

### Target Variables

$$\mathbf{y}_t = \begin{bmatrix} \text{Error\_X}_t \\ \text{Error\_Y}_t \\ \text{Error\_Z}_t \\ \text{3D\_Orbit\_Error}_t \\ \text{Error\_Clock}_t \end{bmatrix}$$

- **$\text{Error\_X}, \text{Error\_Y}, \text{Error\_Z}$**: ECEF orbit coordinate differences between broadcast and modelled precise ephemeris ($\text{metres}$).
- **$\text{3D\_Orbit\_Error}$**: Euclidean norm of the coordinate errors: $\sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2}$ ($\text{metres}$).
- **$\text{Error\_Clock}$**: Satellite onboard atomic clock offset / bias ($\text{seconds}$).

---

## 🏗️ Modular Python Script Architecture

The codebase is refactored from notebooks into a clean, production-grade modular Python package:

```
Satellite ML/
├── src/
│   ├── __init__.py               # Package metadata
│   ├── config.py                 # Central configurations, targets, time parameters, default hyperparameters
│   ├── data.py                   # Ingestion, cleaning, outlier filtering, cyclical/rolling features, dataset builders
│   ├── models/
│   │   ├── __init__.py
│   │   ├── keras_bilstm.py       # TensorFlow / Keras BiLSTM + GRU model
│   │   ├── pytorch_bilstm.py     # PyTorch BiLSTM + GRU equivalent (dual-backend support)
│   │   ├── pytorch_transformer.py# Multi-Task Transformer Forecaster with Probabilistic & Spike heads
│   │   ├── pytorch_diffusion.py  # Conditional Diffusion Denoiser & reverse sampling
│   │   └── losses.py             # Custom losses (Gaussian NLL, BCE spike, smoothness, FFT frequency, clock acceleration)
│   ├── evaluate.py               # Multi-horizon (15m, 30m, 1h, 2h, 6h, 12h, 24h) and per-satellite metric calculation
│   └── visualize.py              # Diagnostic plotting (training curves, heatmaps, residuals, uncertainty bands, FFT spectra)
├── train_bilstm.py               # BiLSTM + GRU training & evaluation CLI entrypoint
├── train_transformer.py          # PyTorch Transformer & Diffusion training & evaluation CLI entrypoint
├── tune.py                       # Optuna Bayesian hyperparameter optimization CLI entrypoint
├── main.py                       # Master CLI runner
├── gnss_forecast.py              # Backward-compatible wrapper
├── FINAL_Data.csv                # 8-day 15-minute GNSS dataset (GPS + GLONASS)
└── requirements.txt              # Unified dependencies file
```

---

## 🔬 Architecture & Methodology

### 1. Production BiLSTM + GRU Forecaster (`train_bilstm.py`)

A single shared recurrent forecaster trained across all complete satellites to capture generalizable orbital dynamics:

```
Input: Lookback Window (96 timesteps × 5 target features)
  │
  ├──► Bidirectional LSTM Layer (32 units, Dropout=0.3)
  │      └── Captures forward & backward temporal dynamics
  │
  ├──► GRU Layer (64 units, Dropout=0.11)
  │      └── Compresses representation into final state vector
  │
  ├──► Layer Normalization
  │
  ├──► Dense Projection (64 units, ReLU)
  │
  └──► Output Layer: Dense(96 × 5) ──► Reshape(96, 5)
         └── Direct multi-step 24-hour prediction
```

- **Loss Function**: Huber Loss ($\delta = 1.0$) for outlier-robust training.
- **Normalization**: `StandardScaler` fitted exclusively on Day 1-7 training partitions.

### 2. Deep Transformer & Diffusion Forecaster (`train_transformer.py`)

A physics-informed multi-task Transformer architecture incorporating:
- **Satellite Entity Embeddings**: Learned vector representations for PRNs.
- **Context Compression & Future Query Decoder**: Self-attention pooling projecting global state onto future tokens.
- **Probabilistic Head**: Gaussian distribution parameters $(\mu, \sigma)$ with softplus activation.
- **Spike Detection Head**: Binary classification of perturbation events.
- **Physics-Informed Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{spike}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{accel}} + \lambda_{\text{multi}} \mathcal{L}_{\text{multi}} + \lambda_{\text{freq}} \mathcal{L}_{\text{FFT}} + \lambda_{\text{clock}} \mathcal{L}_{\text{drift}}$$
- **Conditional Diffusion Denoiser**: 100-step reverse diffusion sampling for stochastic uncertainty quantification.

### 3. Bayesian Hyperparameter Tuning (`tune.py`)

Automated Optuna optimization over:
- BiLSTM units $\in \{32, 64, 128\}$
- GRU units $\in \{16, 32, 64\}$
- Dropout rates $\in [0.1, 0.4]$
- Learning rate $\in [10^{-4}, 5 \times 10^{-3}]$ (log scale)
- Batch size $\in \{32, 64\}$

---

## 📈 Performance & Benchmark Results

Evaluated on out-of-sample Day 8 across **51 complete satellites** (31 GPS, 20 GLONASS):

### 24-Hour Horizon Aggregate Performance

| Target Metric | Mean MAE | Mean RMSE | Units |
| :--- | :---: | :---: | :---: |
| **Error_X** | `2,127.25` | `5,459.39` | metres |
| **Error_Y** | `2,132.81` | `5,487.25` | metres |
| **Error_Z** | `2,131.09` | `5,462.24` | metres |
| **3D_Orbit_Error** | `3,178.14` | `6,133.04` | metres |
| **Error_Clock** | `0.011956` | `0.101967` | seconds |

### Multi-Horizon MAE Breakdown

| Forecast Horizon | Error_X (m) | Error_Y (m) | Error_Z (m) | 3D Orbit Error (m) | Clock Error (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **15 min** | 7,207.65 | 10,015.80 | 7,609.16 | 4,790.71 | 0.998872 |
| **30 min** | 3,717.18 | 5,025.48 | 3,905.18 | 3,560.41 | 0.501114 |
| **1 hour** | 1,941.31 | 2,533.73 | 1,966.98 | 2,670.96 | 0.250930 |
| **2 hours** | 3,005.02 | 3,133.41 | 2,797.60 | 4,236.85 | 0.126044 |
| **6 hours** | 2,333.53 | 2,276.46 | 2,272.81 | 3,395.47 | 0.043405 |
| **12 hours** | 2,226.27 | 2,186.57 | 2,096.09 | 3,047.27 | 0.022487 |
| **24 hours** | 2,127.25 | 2,132.81 | 2,131.09 | 3,178.14 | 0.011956 |

---

## ⚙️ Installation & Virtual Environment Setup

### 1. Create Virtual Environment

```bash
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 CLI Usage Guide

### 1. Unified CLI Runner (`main.py`)

Run any pipeline via `main.py`:

```bash
# Run BiLSTM + GRU Forecaster
.venv\Scripts\python.exe main.py --model bilstm --data FINAL_Data.csv --output ./gnss_results

# Run PyTorch Transformer Forecaster with Conditional Diffusion
.venv\Scripts\python.exe main.py --model transformer --data FINAL_Data.csv --output ./transformer_results --enable-diffusion

# Run Optuna Hyperparameter Tuning
.venv\Scripts\python.exe main.py --model tune --data FINAL_Data.csv --n-trials 15
```

---

### 2. Train BiLSTM + GRU Pipeline (`train_bilstm.py`)

```bash
.venv\Scripts\python.exe train_bilstm.py --data FINAL_Data.csv --output ./gnss_results --epochs 60 --batch-size 64
```

**Key Arguments:**
- `--data`: Path to dataset CSV (default: `FINAL_Data.csv`).
- `--output`: Output directory for models and plots (default: `./gnss_results`).
- `--epochs`: Number of training epochs (default: `60`).
- `--batch-size`: Mini-batch size (default: `64`).
- `--lr`: Learning rate (default: `1.56e-3`).
- `--backend`: Execution backend (`auto`, `keras`, or `torch`).

---

### 3. Train Transformer & Diffusion Pipeline (`train_transformer.py`)

```bash
.venv\Scripts\python.exe train_transformer.py --data FINAL_Data.csv --output ./transformer_results --epochs 30 --enable-diffusion
```

**Key Arguments:**
- `--epochs`: Transformer training epochs (default: `30`).
- `--diffusion-epochs`: Diffusion training epochs (default: `80`).
- `--enable-diffusion`: Flag to train conditional diffusion denoiser.
- `--d-model`: Transformer hidden dimension (default: `64`).
- `--nhead`: Number of multi-head attention heads (default: `4`).
- `--device`: Target compute device (`cuda` or `cpu`).

---

### 4. Hyperparameter Optimization (`tune.py`)

```bash
.venv\Scripts\python.exe tune.py --data FINAL_Data.csv --n-trials 20
```

---

## 📊 Generated Visualizations & Diagnostics

All outputs generated by the training scripts are saved directly to the designated output folder:

| Artifact | Description |
| :--- | :--- |
| `01_training_history.png` | Loss and metric convergence profiles over training epochs. |
| `02_prediction_vs_actual_GPS.png` | 24-hour time-series overlay of predicted vs ground-truth errors for GPS satellites. |
| `03_prediction_vs_actual_GLONASS.png` | Time-series prediction overlay for GLONASS satellites. |
| `04_multihorizon_mae_heatmap.png` | Multi-horizon error heatmap across forecast intervals (15m to 24h). |
| `05_residual_distributions.png` | Residual histogram with fitted Gaussian curves verifying zero-mean unbiased predictions. |
| `06_per_satellite_mae.png` | Comparative bar chart showing individual MAE across all evaluated PRNs. |
| `metrics_summary.json` | Comprehensive machine-readable metrics report containing per-satellite and aggregate MAE/RMSE. |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
