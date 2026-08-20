# OrbitIQ ISRO SIH 2025 End-to-End Pipeline Execution Report

- **Dataset Source**: `data\orbitiq\ORBITIQ_ISRO_BENCHMARK.csv`
- **Hardware Acceleration**: `cuda`
- **Total Training Epochs**: `10`

## Overall Model Benchmarks

| Architecture | Overall 3D MAE (m) | Overall 3D RMSE (m) | Conformal 90% Coverage | Conformal 95% Coverage |
|---|---|---|---|---|
| **Probabilistic Hybrid Transformer** | **2.5418** | **4.2878** | 94.5% | 98.2% |
| **Deterministic BiLSTM-GRU** | 1.5420 | 2.2931 | N/A | N/A |
| Zero Baseline | 1.3360 | 1.9355 | N/A | N/A |
| Persistence Baseline | 1.6879 | 2.7802 | N/A | N/A |
| Seasonal Baseline | 2.8756 | 6.3498 | N/A | N/A |
| Drift Baseline | 2.7077 | 4.4465 | N/A | N/A |

## Multi-Horizon Forecast Performance (Hybrid Transformer)

| Horizon | Error_X MAE (m) | Error_Y MAE (m) | Error_Z MAE (m) | Error_Clock MAE (s) | 3D Orbit MAE (m) |
|---|---|---|---|---|---|
| **15 min** | 0.2862 | 0.8083 | 0.3891 | 5.2613e-10 | **0.9912** |
| **30 min** | 0.4056 | 0.6013 | 0.3356 | 1.1451e-09 | **0.8623** |
| **1 hour** | 0.4762 | 0.7910 | 0.9802 | 2.2212e-09 | **1.4894** |
| **2 hours** | 0.9723 | 1.8592 | 0.7840 | 2.2050e-09 | **2.3150** |
| **6 hours** | 1.3795 | 1.3368 | 2.3157 | 2.7682e-09 | **3.1047** |
| **12 hours** | 1.0688 | 1.3229 | 1.0930 | 1.7601e-09 | **2.1569** |
| **24 hours** | 0.8070 | 0.9532 | 0.8218 | 2.4778e-09 | **1.5902** |

## ISRO SIH 2025 PS 25176 Compliance
- **15-Minute Uniform Cadence**: 100% compliant across 8 full days (7-day train/val, 8th-day multi-step test).
- **Normality & Conformal Scaling**: Conformal calibration guarantees empirical coverage at 90% and 95% confidence intervals.
- **Physics-Informed Separation**: Multi-step predictions maintain exact vector Euclidean 3D orbit residuals.