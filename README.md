# GNSS orbit and clock residual forecasting

This project forecasts 24 hours of GPS/GLONASS broadcast orbit residuals
(`Error_X`, `Error_Y`, `Error_Z`) and clock residuals (`Error_Clock`) from a
24-hour lookback sampled every 15 minutes.

## Scientific status

The training and evaluation code is now leakage-safe and unit-aware, but the
bundled `FINAL_Data.csv` has a confirmed upstream orbit-alignment defect. It
also contains 497 converted SP3 missing-clock sentinels. The clock values are
masked correctly; the orbit source data must be rebuilt before model results
can support a scientific or operational claim. See [DATA_AUDIT.md](DATA_AUDIT.md).

Old result directories are retained as historical artifacts, but their metrics
must not be compared with new runs: the previous Transformer targets were
standardized twice and inverse-transformed once, while the old BiLSTM silently
evaluated only 35 of 51 selected satellites.

## What the corrected pipeline does

- masks SP3 missing-clock labels without deleting their rows;
- retains large targets and reports them instead of censoring future truth;
- requires exact 15-minute cadence inside every lookback/forecast window;
- creates chronological train/validation/test label blocks with no label
  overlap and fits scalers on training rows only;
- transforms feature and target arrays separately, eliminating double scaling;
- predicts four primitive targets and derives 3D vector error from XYZ;
- adds point-in-time broadcast position, causal velocity, orbital phase,
  broadcast clock, and clock-drift features when available;
- evaluates exact-lead and cumulative-to-lead metrics explicitly;
- reports MAE, RMSE, median, p90/p95/p99, per-satellite and constellation slices,
  clock seconds/nanoseconds/range-equivalent metres, and valid-label coverage;
- runs zero-correction, persistence, seasonal-naive, and drift baselines;
- trains a Student-t probabilistic head by default and calibrates intervals on
  the chronological validation fold with scaled split conformal prediction;
- treats diffusion as an optional residual ablation and scores all its test
  samples with CRPS and the XYZ energy score;
- saves reloadable bundles containing model state, scalers, feature/target
  schema, satellite vocabulary, split boundaries, data hash, code SHA, seed,
  runtime versions, and the best validation epoch;
- fails model promotion unless the candidate beats every configured baseline
  at every required horizon and the data audit passes.

## Installation

Python 3.10+ and PyTorch 2.x are required. Exact top-level package versions for
the verified run are constrained in `requirements-lock.txt`; the PyTorch wheel
still needs to match the machine's CUDA runtime.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Install the appropriate PyTorch CUDA wheel from the
[official selector](https://pytorch.org/get-started/locally/) if the default
wheel does not match the machine's CUDA runtime.

## Commands

Audit the data contract (expected to fail strictly for the bundled CSV):

```powershell
.venv\Scripts\python.exe main.py --model audit `
  --data FINAL_Data.csv --report data_quality_report.json --strict
```

Evaluate executable baselines:

```powershell
.venv\Scripts\python.exe main.py --model baselines `
  --data FINAL_Data.csv --output baseline_metrics.json
```

Train the deterministic BiLSTM-GRU benchmark:

```powershell
.venv\Scripts\python.exe main.py --model bilstm `
  --data FINAL_Data.csv --output gnss_results_new `
  --epochs 45 --batch-size 128 --device cuda
```

Train the probabilistic recurrent-attention model:

```powershell
.venv\Scripts\python.exe main.py --model transformer `
  --data FINAL_Data.csv --output transformer_results_new `
  --epochs 30 --batch-size 128 --distribution student_t --device cuda
```

Diffusion is off by default. Enable it only as a scored ablation:

```powershell
.venv\Scripts\python.exe train_transformer.py `
  --enable-diffusion --diffusion-epochs 20 `
  --diffusion-eval-samples 20 --ddim-steps 20
```

Run reproducible Optuna tuning on the same temporal folds:

```powershell
.venv\Scripts\python.exe main.py --model tune `
  --data FINAL_Data.csv --n-trials 20 --epochs 12 --device cuda
```

The legacy Keras model remains as reference code, but corrected training rejects
the Keras backend because it does not consume per-target availability masks.

## Outputs

Each corrected training run writes:

- `*_bundle.pt`: model plus immutable preprocessing/provenance state;
- `artifact_manifest.json`: human-readable bundle metadata;
- `test_predictions.npz`: physical-unit predictions, labels, masks, satellites,
  and label timestamps;
- `evaluation_report.json`: deterministic, probabilistic, conformal, baseline,
  coverage, data-quality, and promotion results;
- optional diagnostic figures.

No aggregate metric combines metres and seconds.

## Model choices

The default probabilistic model combines a BiLSTM, GRU, configurable stacked
self-attention blocks, PRN embeddings, and a direct multi-horizon residual head.
The head predicts Student-t location and scale; Gaussian training remains an
explicit option. RevIN and shape-warping losses are off by default and should be
treated as ablations. The pseudo spike task is disabled because target-threshold
labels are not genuine maneuver or clock-event annotations.

The utilities in `src/physics.py` provide ECEF↔RIC transforms for the next data
revision. Once clean source products are available, the recommended formulation
is a physical broadcast/numerical propagation baseline plus a learned RIC
residual, with a separately decomposed clock trend/frequency model.

## Evaluation and promotion

With the bundled CSV, the executable raw-data baselines produce approximately:

| Baseline | XYZ vector MAE | Clock MAE |
|---|---:|---:|
| Zero correction | 4,141 m | 15.33 ns |
| Persistence | 4,154 m | 17.48 ns |
| Seasonal naive (96) | 7,619 m | 15.11 ns |
| Drift | 4,472 m | 18.28 ns |

These numbers diagnose the current dataset; they are not GNSS accuracy claims.
One test day and roughly one origin per satellite are insufficient for model
selection. Acquire months or years, use rolling origins with purged targets,
reserve an untouched final period, and add held-out satellite/block tests.

Promotion requirements are machine-readable in
`configs/promotion_policy.json`. They include baseline skill at every required
horizon, calibrated intervals, all-satellite coverage, event/constellation
slices, and declared product latency and resource budgets.

## Research-backed next model revision

After rebuilding the data, prioritize:

1. RIC/RAC residuals around a numerical or broadcast-ephemeris propagator;
2. clock first-differences/frequency with polynomial or Kalman trend removal;
3. ephemeris age, Toe/IODE, state/velocity/elements, Sun-beta/eclipse/yaw, EOP,
   satellite block/clock type, health, prediction, maneuver, and event flags;
4. DLinear/NLinear and N-HiTS, then compact TCN/PatchTST/iTransformer ablations;
5. rolling conformal calibration and downstream SISRE/PPP impact evaluation.

Relevant primary sources:

- [IGS products and operational latency](https://igs.org/products/)
- [IGS MGEX data products](https://igs.org/mgex/data-products/)
- [SP3-d format, units, flags, and missing values](https://files.igs.org/pub/data/format/sp3d.pdf)
- [Physics-constrained RIC residual forecasting with BiLSTM/N-HiTS (ION 2026)](https://www.ion.org/publications/abstract.cfm?articleID=20520)
- [Hybrid physical GNSS orbit propagation](https://elib.dlr.de/139980/)
- [Multi-GNSS orbit/clock SISRE evaluation](https://elib.dlr.de/92092/)
- [Clock decomposition and LSTM residual forecasting](https://link.springer.com/article/10.1007/s10291-021-01115-0)
- [Adaptive conformal prediction under distribution shift](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html)
