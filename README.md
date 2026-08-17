# 🛰️ GNSS Satellite Orbit & Clock Error Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-optional-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Optuna](https://img.shields.io/badge/Optuna-seeded%20TPE-green.svg)](https://optuna.org/)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen.svg)](#-verification-status)

An end-to-end framework for **multi-horizon forecasting of GNSS broadcast
ephemeris and satellite clock residuals**. The project processes 15-minute GPS
and GLONASS time series and predicts the next 24 hours of ECEF orbit-coordinate
and clock error.

The codebase includes a deterministic BiLSTM-GRU benchmark, a probabilistic
recurrent-attention model, calibrated uncertainty, optional residual diffusion,
executable naive baselines, reproducible tuning, strict data auditing, and
fail-closed model-promotion checks.

> [!IMPORTANT]
> The corrected software pipeline is leakage-safe and unit-aware, but the
> bundled `FINAL_Data.csv` contains a confirmed upstream orbit-alignment defect.
> Its results are diagnostic—not scientific or operational GNSS accuracy claims.
> See [Data integrity and scientific status](#-data-integrity-and-scientific-status)
> and [DATA_AUDIT.md](DATA_AUDIT.md).

---

## 📌 Table of Contents

- [Overview and motivation](#-overview-and-motivation)
- [Dataset and problem formulation](#-dataset-and-problem-formulation)
- [Data integrity and scientific status](#-data-integrity-and-scientific-status)
- [Corrected data and evaluation pipeline](#-corrected-data-and-evaluation-pipeline)
- [Deep-learning architectures](#-deep-learning-architectures)
  - [BiLSTM-GRU benchmark](#1-bilstm-gru-benchmark)
  - [Probabilistic hybrid forecaster](#2-probabilistic-hybrid-forecaster)
  - [Residual diffusion](#3-residual-diffusion-optional)
  - [Optuna tuning](#4-reproducible-optuna-tuning)
- [Metrics, baselines, and promotion](#-metrics-baselines-and-promotion)
- [GPU acceleration and reproducibility](#-gpu-acceleration-and-reproducibility)
- [Installation](#-installation)
- [CLI usage guide](#-cli-usage-guide)
- [Generated artifacts and visualizations](#-generated-artifacts-and-visualizations)
- [Verification status](#-verification-status)
- [Research-backed next improvements](#-research-backed-next-improvements)
- [Repository structure](#-repository-structure)
- [References](#-references)

---

## 🌌 Overview and Motivation

GNSS satellites broadcast real-time ephemeris and clock parameters. Their
differences from precise reference products vary with orbital dynamics,
ephemeris age, force-model error, solar-radiation pressure, eclipse and yaw
conditions, satellite hardware, and onboard clock behavior.

The learning task is formulated as **residual forecasting**: estimate future
broadcast-minus-reference errors instead of predicting the complete satellite
position from scratch. A trustworthy residual forecast can eventually support
real-time orbit/clock correction and downstream SISRE or PPP experiments—once
the source-product alignment and operational availability contract are valid.

---

## 📊 Dataset and Problem Formulation

The bundled CSV contains eight days of nominal 15-minute observations:

| Property | Value |
|---|---|
| File | `FINAL_Data.csv` |
| Rows | 41,021 |
| Time span | 2026-01-01 through 2026-01-08 |
| Nominal cadence | 15 minutes, 96 epochs/day |
| Satellites | 54 GPS/GLONASS PRNs |
| Lookback | 96 steps = 24 hours |
| Forecast horizon | 96 steps = 24 hours |
| Corrected train windows | 19,260 |
| Corrected validation windows | 1,113 |
| Corrected test origins | 52 |
| Model inputs | 21 scaled historical/physical features |

### Target variables

The models learn only the four primitive residuals:

$$
\mathbf{y}_t =
\begin{bmatrix}
\mathrm{Error\_X}_t \\
\mathrm{Error\_Y}_t \\
\mathrm{Error\_Z}_t \\
\mathrm{Error\_Clock}_t
\end{bmatrix}
$$

- `Error_X`, `Error_Y`, `Error_Z`: ECEF coordinate residuals in metres.
- `Error_Clock`: satellite clock residual in seconds.

`3D_Orbit_Error` is not an independent target. Evaluation derives the physical
vector error from the coordinate forecast:

$$
e_{3D} = \sqrt{(\hat e_X-e_X)^2 + (\hat e_Y-e_Y)^2 + (\hat e_Z-e_Z)^2}.
$$

This guarantees that the reported 3D error is consistent with the predicted XYZ
trajectory and cannot become negative.

### Input features

The first four input channels are historical target residuals. When the source
columns are present, the corrected pipeline also includes:

- daily sine/cosine time encodings;
- causal rolling residual means;
- broadcast ECEF position and clock;
- causal finite-difference broadcast velocity and clock drift;
- broadcast orbital radius;
- broadcast phase sine/cosine.

All derivatives use only the current and previous epoch—never a centered or
future difference.

---

## 🚨 Data Integrity and Scientific Status

The executable audit found two critical source-data defects:

1. **497 missing SP3 clock values were converted to approximately one second.**
   The official SP3 missing-clock value is `999999.999999` microseconds. It is
   not a physical clock event. The corrected pipeline masks these clock labels
   while retaining the timestamp and valid orbit labels.
2. **The orbit residual contains a repeating synchronous kilometre-scale
   pattern.** About 20.08% of rows exceed 1 km; at 154 epochs, at least 95% of
   satellites exceed 1 km together. The event indicator has mean lag-5
   correlation 0.968, corresponding to a suspicious 75-minute cycle.

The audit also finds 14 non-15-minute per-satellite intervals. Corrected window
construction purges every sequence crossing one of those gaps.

Run the audit directly:

```powershell
.venv\Scripts\python.exe audit_data.py `
  --data FINAL_Data.csv `
  --report data_quality_report.json `
  --strict
```

The bundled CSV intentionally returns exit code `2` under `--strict`. A report
is still written. The contract is defined in
[`configs/data_contract.json`](configs/data_contract.json).

### Why the orbit issue is not filtered away

Deleting rows because their future target exceeds a threshold would censor the
hardest cases, create irregular row-index sequences, and leak target knowledge
into cohort selection. The corrected pipeline retains large targets, reports
them, and blocks scientific promotion until the upstream join is rebuilt.

The repository does not contain the original RINEX navigation, SP3/CLK source
files, download timestamps, or the upstream join/interpolation implementation.
Therefore the orbit values cannot be repaired honestly from this repository
alone. The required rebuild checklist is documented in
[DATA_AUDIT.md](DATA_AUDIT.md).

Historical `gnss_results/` and `transformer_results/` artifacts are preserved,
but their metrics are not comparable with corrected runs. The previous
Transformer targets were scaled twice and inverse-transformed once, and the old
BiLSTM evaluation silently reduced its test cohort to 35 satellites.

---

## 🧱 Corrected Data and Evaluation Pipeline

```text
CSV + source validity
        │
        ├── detect SP3 clock sentinel ──> per-target availability mask
        ├── validate duplicate keys and cadence
        └── retain large targets for honest scoring
        │
        ▼
causal feature engineering
        │
        ▼
raw-time chronological boundaries
        ├── training labels
        ├── validation labels (disjoint)
        └── final test labels (disjoint)
        │
        ▼
fit feature + target scalers on training block only
        │
        ▼
build exact-cadence windows and purge boundary-crossing targets
        │
        ▼
masked training ──> physical-unit evaluation ──> baseline/promotion gates
```

The main corrections are:

- no future-target row filtering;
- no future-completeness satellite selection;
- no random split of overlapping windows;
- no shared train/validation labels;
- no scaler fitting on validation or test rows;
- no feature-then-target double transformation;
- no independent `3D_Orbit_Error` prediction;
- no aggregate metric that combines metres and seconds;
- no pseudo-event loss from target-threshold labels;
- no promotion without baseline skill, calibration, coverage, and clean data.

---

## 🔬 Deep-Learning Architectures

### 1. BiLSTM-GRU Benchmark

Implemented in `train_bilstm.py` and `src/models/pytorch_bilstm.py`.

```text
21-channel lookback
  → bidirectional LSTM
  → dropout
  → GRU sequence encoder
  → attention pooling + final hidden state
  → LayerNorm and dense multi-horizon head
  → residual anchor from last observed target vector
  → 96 × 4 forecast
```

The benchmark uses masked Smooth-L1 loss, masked validation MAE, gradient
clipping, deterministic seeding, early stopping, and best-checkpoint restore.
It consumes the same chronological folds as the probabilistic model.

The legacy Keras architecture remains as reference code, but corrected training
rejects `--backend keras` because that path does not consume per-target
availability masks.

### 2. Probabilistic Hybrid Forecaster

Implemented in `train_transformer.py` and
`src/models/pytorch_transformer.py`.

```text
historical + physical features
  + relative Time2Vec encoding
  + PRN entity embedding
        │
        ▼
BiLSTM → GRU → configurable stacked self-attention + feed-forward blocks
        │
        ▼
sequence-preserving temporal projection
        │
        ├── location μ for 96 × 4 targets
        └── positive predictive scale for 96 × 4 targets
```

Key changes:

- `--num-layers` now controls actual stacked attention blocks.
- Target-channel indices are explicit instead of assumed.
- Student-t negative log-likelihood is the default robust objective.
- Gaussian NLL remains available through `--distribution gaussian`.
- Every likelihood and optional regularizer respects target masks.
- Absolute smoothing, FFT, DILATE, and event losses are off by default and are
  treated as ablations.
- RevIN is optional (`--use-revin`), and its uncertainty denormalization now
  reverses the learned affine scale correctly.
- Validation selects the checkpoint used for final evaluation.

### 3. Residual Diffusion (Optional)

The diffusion module models `target - point_forecast`, not the full trajectory.
Training and sampling now operate in the same residual space.

- cosine schedule reaches an approximately normal terminal state;
- reverse DDPM variance uses the posterior variance;
- DDIM includes the terminal reconstruction step;
- invalid training targets are masked;
- all test trajectories—not only the first example—are sampled and scored;
- outputs are evaluated with per-target empirical CRPS and the XYZ energy score.

Diffusion is disabled by default. It should be retained only if it beats the
Student-t/conformal alternative on proper scores and downstream decision risk.

### 4. Reproducible Optuna Tuning

`tune.py` uses the same hardened dataset bundle and target masks as training.
The TPE sampler, each trial, and DataLoader shuffling are seeded. Trials optimize
masked validation MAE on chronological validation labels and support pruning.

---

## 📈 Metrics, Baselines, and Promotion

### Deterministic scorecard

For every primitive target, the canonical evaluator reports:

- MAE and RMSE;
- median absolute error;
- p90, p95, and p99 absolute error;
- exact-lead metrics at 15 min, 30 min, 1 h, 2 h, 6 h, 12 h, and 24 h;
- cumulative steps 1 through each horizon, labeled separately;
- per-satellite and per-constellation slices;
- valid-label counts and coverage.

Orbit-vector error is derived in metres. Clock error is shown separately in
seconds, nanoseconds, and range-equivalent metres using $c\Delta t$.

### Probabilistic scorecard

The probabilistic evaluator reports Student-t or Gaussian NLL, central interval
coverage, and interval width per target and horizon. Scaled split-conformal
multipliers are fitted on validation only and evaluated on the final test block.

Required 80%, 90%, and 95% conformal coverages must fall within the configured
tolerance before promotion.

### Executable baselines

Run zero-correction, persistence, seasonal-naive, and drift forecasts with:

```powershell
.venv\Scripts\python.exe evaluate_baselines.py `
  --data FINAL_Data.csv `
  --output baseline_metrics.json
```

Current diagnostic results on the bundled, known-defective CSV are:

| Baseline | XYZ vector MAE | Clock MAE |
|---|---:|---:|
| Zero correction | 4,141 m | 15.33 ns |
| Persistence | 4,154 m | 17.48 ns |
| Seasonal naive, period 96 | 7,619 m | 15.11 ns |
| Drift | 4,472 m | 18.28 ns |

These values diagnose the dataset and establish the minimum skill threshold;
they are not satellite-accuracy claims.

### Promotion policy

A probabilistic candidate is not promotion-eligible unless:

1. the strict data audit passes;
2. it beats every configured baseline at every required horizon;
3. required interval coverage is calibrated;
4. test coverage and evaluated satellites are reported;
5. operational issue time, input-product latency, and resource budgets are
   declared.

The policy is machine-readable in
[`configs/promotion_policy.json`](configs/promotion_policy.json). Current runs
correctly remain ineligible while the source-data defect persists.

---

## ⚡ GPU Acceleration and Reproducibility

The trainers support NVIDIA CUDA and CPU execution.

- in-memory tensor minibatching with `FastGPUTensorLoader`;
- automatic mixed precision on CUDA;
- gradient clipping;
- deterministic seeds for Python, NumPy, PyTorch, CUDA, DataLoader shuffling,
  and Optuna;
- `cudnn.benchmark=False` for deterministic runs;
- `--nondeterministic` as an explicit speed-oriented opt-out;
- best-validation checkpoint selection;
- exact top-level dependency constraints in `requirements-lock.txt`.

Each checkpoint bundle includes:

- model and matching optimizer state;
- immutable model/training configuration;
- feature and target scaler state;
- feature/target schema and target-channel mapping;
- satellite vocabulary;
- split boundaries and data-quality metadata;
- source CSV SHA-256 and Git SHA;
- seed and Python/NumPy/PyTorch/runtime versions.

---

## 💻 Installation

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The lock file records the verified top-level versions. If the constrained
PyTorch wheel does not match the machine's CUDA runtime, install the appropriate
wheel using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

### 3. Install test dependencies

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

---

## 🚀 CLI Usage Guide

### 1. Master CLI

`main.py` dispatches all supported workflows:

```powershell
# Strict, read-only data audit
.venv\Scripts\python.exe main.py --model audit `
  --data FINAL_Data.csv --strict

# Executable baseline scorecard
.venv\Scripts\python.exe main.py --model baselines `
  --data FINAL_Data.csv --output baseline_metrics.json

# Deterministic benchmark
.venv\Scripts\python.exe main.py --model bilstm `
  --data FINAL_Data.csv --output gnss_results_new

# Probabilistic hybrid
.venv\Scripts\python.exe main.py --model transformer `
  --data FINAL_Data.csv --output transformer_results_new

# Reproducible tuning
.venv\Scripts\python.exe main.py --model tune `
  --data FINAL_Data.csv --n-trials 20
```

### 2. Train BiLSTM-GRU

```powershell
.venv\Scripts\python.exe train_bilstm.py `
  --data FINAL_Data.csv `
  --output gnss_results_new `
  --epochs 45 `
  --batch-size 128 `
  --lr 0.002 `
  --device cuda
```

Useful options:

- `--early-stopping-patience 10`
- `--bilstm-units 64`
- `--gru-units 64`
- `--skip-plots`
- `--nondeterministic`

### 3. Train Probabilistic Hybrid

```powershell
.venv\Scripts\python.exe train_transformer.py `
  --data FINAL_Data.csv `
  --output transformer_results_new `
  --epochs 30 `
  --batch-size 128 `
  --distribution student_t `
  --student-t-df 3 `
  --num-layers 3 `
  --device cuda
```

Optional ablations:

```powershell
# RevIN
--use-revin

# Shape loss; disabled by default
--lambda-dilate 0.05

# Residual diffusion with full-test proper scoring
--enable-diffusion `
--diffusion-epochs 20 `
--diffusion-eval-samples 20 `
--ddim-steps 20
```

The pseudo-event head cannot be enabled on the current dataset because genuine
maneuver/clock-event labels are not present.

### 4. Tune BiLSTM-GRU

```powershell
.venv\Scripts\python.exe tune.py `
  --data FINAL_Data.csv `
  --n-trials 20 `
  --epochs 12 `
  --device cuda `
  --output tuning_results.json
```

---

## 📦 Generated Artifacts and Visualizations

### Model and evaluation artifacts

| File | Purpose |
|---|---|
| `gnss_bilstm_bundle.pt` | BiLSTM model, preprocessing, provenance, and best optimizer state |
| `gnss_hybrid_forecaster_bundle.pt` | Probabilistic model and complete reload state |
| `gnss_diffusion_bundle.pt` | Optional denoiser, schedule, scaler, and provenance |
| `artifact_manifest.json` | Human-readable artifact metadata |
| `test_predictions.npz` | Physical-unit predictions, labels, masks, PRNs, and timestamps |
| `evaluation_report.json` | Model, uncertainty, conformal, baseline, coverage, quality, and promotion results |
| `diffusion_test_samples.npz` | All residual-diffusion test draws |

### Diagnostic figures

| Figure | Content |
|---|---|
| `01_training_history.png` | Training/validation convergence and predictive scale |
| `02_multihorizon_mae_heatmap.png` | Explicit exact-lead error by target and horizon |
| `02_prediction_vs_actual_GPS.png` | GPS trajectory comparisons for the BiLSTM path |
| `03_prediction_vs_actual_GLONASS.png` | GLONASS trajectory comparisons for the BiLSTM path |
| `03_probabilistic_uncertainty.png` | Location and predictive-scale interval diagnostic |
| `04_frequency_spectrum.png` | Actual-versus-predicted spectrum diagnostic |
| `05_residual_distributions.png` | Finite-label residual distributions |
| `05_diffusion_samples.png` | Optional stochastic residual trajectories |
| `06_per_satellite_mae.png` | Per-PRN deterministic error |

Visualization code handles masked/non-finite targets without treating them as
real observations.

---

## ✅ Verification Status

The final implementation was checked with:

- **32 passing focused tests** covering sentinel handling, scaling, temporal
  disjointness, cadence, masks, metric units/semantics, baselines, calibration,
  RevIN scale inversion, attention depth, RIC transforms, and diffusion space;
- compilation and import checks for all CLI modules;
- actual-data bundle construction;
- one-epoch CUDA smoke runs for both corrected trainers;
- seeded Optuna smoke tuning;
- strict audit exit behavior;
- baseline CLI generation;
- Transformer checkpoint reconstruction and state reload;
- optional diffusion training, sampling, CRPS, and energy-score evaluation.

The smoke models correctly fail promotion. They were used to validate the
pipeline, not to claim final model accuracy.

---

## 🔭 Research-Backed Next Improvements

After rebuilding the source data, the highest-value next iteration is:

1. **Acquire months or years of point-in-time products.** Use rolling forecast
   origins, an untouched final period, and held-out satellite/block tests.
2. **Forecast RIC/RAC residuals around a physical baseline.** The included
   `src/physics.py` supplies tested ECEF↔RIC transforms.
3. **Separate clock modeling.** Forecast clock first differences or frequency
   after polynomial/Kalman trend removal, reporting nanoseconds and $c\Delta t$.
4. **Add operational covariates.** Ephemeris age, Toe/IODE, state/velocity,
   orbital elements, Sun-beta/eclipses/yaw, EOP, block/clock type, health,
   prediction, maneuver, and clock-event flags.
5. **Benchmark compact alternatives first.** DLinear/NLinear, N-HiTS, TCN,
   PatchTST, and iTransformer should beat the strongest simple and physical
   baseline before increasing architecture complexity.
6. **Evaluate downstream impact.** Add RIC errors, datum-corrected clock, SISRE,
   p50/p95/p99, event slices, and PPP positioning impact.

Any real-time experiment must declare when precise historical residuals become
available. Using labels that arrive after forecast issue time is operational
leakage even when the train/test timestamps are chronologically separated.

---

## 🗂️ Repository Structure

```text
Satellite ML/
├── configs/
│   ├── data_contract.json          # Machine-readable input/split contract
│   └── promotion_policy.json       # Fail-closed model-promotion rules
├── src/
│   ├── models/
│   │   ├── keras_bilstm.py         # Legacy reference model
│   │   ├── losses.py               # Masked Student-t/Gaussian and ablation losses
│   │   ├── pytorch_bilstm.py       # Deterministic recurrent benchmark
│   │   ├── pytorch_diffusion.py    # Residual DDPM/DDIM module
│   │   └── pytorch_transformer.py  # Probabilistic recurrent-attention model
│   ├── artifacts.py                # Seeds, hashes, scaler serialization
│   ├── baselines.py                # Zero/persistence/seasonal/drift forecasts
│   ├── calibration.py              # Scaled split-conformal intervals
│   ├── config.py                   # Project defaults and schemas
│   ├── data.py                     # Contracts, features, folds, masks, loaders
│   ├── evaluate.py                 # Unit-aware point/probabilistic/sample metrics
│   ├── physics.py                  # ECEF↔RIC and clock-range utilities
│   └── visualize.py                # Mask-safe diagnostic plotting
├── tests/                          # Focused regression tests
├── audit_data.py                   # Read-only strict dataset audit
├── evaluate_baselines.py           # Baseline evaluation CLI
├── train_bilstm.py                 # Deterministic training CLI
├── train_transformer.py            # Probabilistic/diffusion training CLI
├── tune.py                         # Seeded Optuna CLI
├── main.py                         # Unified workflow dispatcher
├── gnss_forecast.py                # Backward-compatible BiLSTM wrapper
├── DATA_AUDIT.md                   # Detailed evidence and rebuild requirements
├── requirements.txt                # Runtime dependencies
├── requirements-lock.txt           # Verified top-level version constraints
├── requirements-dev.txt            # Test dependencies
└── FINAL_Data.csv                  # Bundled diagnostic dataset
```

---

## 📚 References

### Official products and formats

- [IGS products, accuracies, and latency](https://igs.org/products/)
- [IGS MGEX data products](https://igs.org/mgex/data-products/)
- [IGS SP3-d format specification](https://files.igs.org/pub/data/format/sp3d.pdf)
- [IGS satellite metadata](https://igs.org/mgex/metadata/)

### GNSS forecasting and evaluation

- [Physics-constrained RIC residual forecasting with BiLSTM/N-HiTS, ION 2026](https://www.ion.org/publications/abstract.cfm?articleID=20520)
- [Hybrid physical GNSS orbit propagation](https://elib.dlr.de/139980/)
- [Multi-constellation force-model strategy](https://www.sciencedirect.com/science/article/pii/S0094576523002138)
- [Clock decomposition and LSTM residual forecasting](https://link.springer.com/article/10.1007/s10291-021-01115-0)
- [Multi-GNSS SISRE evaluation](https://elib.dlr.de/92092/)

### Forecasting and uncertainty

- [N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854)
- [PatchTST](https://openreview.net/pdf?id=Jbdc0vTOcol)
- [iTransformer](https://openreview.net/pdf?id=JePfAI8fah)
- [Adaptive conformal prediction under distribution shift](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html)

---

## 📄 License

No license file is currently included in this repository. Add an explicit
license before distributing or accepting external contributions.
