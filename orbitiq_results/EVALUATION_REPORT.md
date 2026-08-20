# ISRO SIH 2025 GNSS Orbit & Clock Error Modeling Evaluation Report

Evaluation results for pre-trained LSTM neural networks and Random Forest models on ISRO SIH 2025 dataset.

## Dataset: `DATA_GEO_Train.csv` (GEO Orbit, 142 Epochs)
- **Time Span**: `2025-09-01 06:00:00` to `2025-09-07 23:41:00`

| Target Residual | Model | MAE (m) | RMSE (m) | Max Error (m) | Shapiro-Wilk W | p-value | Normal? |
|---|---|---|---|---|---|---|---|
| `x_error (m)` | **Pretrained LSTM** | 4.1537 | 5.4169 | 9.6523 | 0.9449 | 1.6111e-01 | Yes |
| `x_error (m)` | Random Forest | 4.4780 | 5.6514 | 11.0151 | 0.9735 | 6.9512e-01 | Yes |
| `y_error (m)` | **Pretrained LSTM** | 3.8355 | 5.2206 | 9.1010 | 0.9114 | 2.4651e-02 | No |
| `y_error (m)` | Random Forest | 4.1781 | 5.4101 | 11.2314 | 0.9624 | 4.1788e-01 | Yes |
| `z_error (m)` | **Pretrained LSTM** | 5.2093 | 6.9406 | 12.5523 | 0.9342 | 8.7491e-02 | Yes |
| `z_error (m)` | Random Forest | 5.0614 | 6.6056 | 13.5564 | 0.9655 | 4.8835e-01 | Yes |
| `satclockerror (m)` | **Pretrained LSTM** | 2.3133 | 3.3353 | 6.1871 | 0.8750 | 3.7465e-03 | No |
| `satclockerror (m)` | Random Forest | 2.6843 | 3.3396 | 6.5416 | 0.9139 | 2.8257e-02 | No |

- **LSTM 3D Position Error**: Mean = `8.0927 m`, RMSE = `10.2357 m`, Max = `18.0493 m`
- **RF 3D Position Error**: Mean = `8.4790 m`, RMSE = `10.2392 m`, Max = `19.3333 m`
`
### 8th-Day Forward Predictions:
- `LSTM_x_error (m)`: `0.522141 m`
- `LSTM_y_error (m)`: `0.348722 m`
- `LSTM_z_error (m)`: `0.374983 m`
- `LSTM_satclockerror (m)`: `-0.046168 m`
- `RF_x_error (m)`: `-1.843414 m`
- `RF_y_error (m)`: `-3.726239 m`
- `RF_z_error (m)`: `1.102024 m`
- `RF_satclockerror (m)`: `-2.375226 m`

## Dataset: `DATA_MEO_Train.csv` (MEO Orbit, 90 Epochs)
- **Time Span**: `2025-09-01 14:00:00` to `2025-09-07 16:00:00`

| Target Residual | Model | MAE (m) | RMSE (m) | Max Error (m) | Shapiro-Wilk W | p-value | Normal? |
|---|---|---|---|---|---|---|---|
| `x_error (m)` | **Pretrained LSTM** | 0.2659 | 0.4062 | 1.3272 | 0.8476 | 9.8587e-03 | No |
| `x_error (m)` | Random Forest | 0.2886 | 0.4195 | 1.3270 | 0.8041 | 2.3137e-03 | No |
| `y_error (m)` | **Pretrained LSTM** | 0.3233 | 0.4381 | 0.9995 | 0.9431 | 3.5656e-01 | Yes |
| `y_error (m)` | Random Forest | 0.2949 | 0.3885 | 0.6942 | 0.9300 | 2.1784e-01 | Yes |
| `z_error (m)` | **Pretrained LSTM** | 0.5950 | 0.6308 | 1.0739 | 0.8503 | 1.0836e-02 | No |
| `z_error (m)` | Random Forest | 0.5529 | 0.6215 | 1.0083 | 0.8617 | 1.6233e-02 | No |
| `satclockerror (m)` | **Pretrained LSTM** | 0.3567 | 0.4579 | 0.7829 | 0.9197 | 1.4615e-01 | Yes |
| `satclockerror (m)` | Random Forest | 0.3495 | 0.4666 | 0.8990 | 0.9154 | 1.2339e-01 | Yes |

- **LSTM 3D Position Error**: Mean = `0.8244 m`, RMSE = `0.8688 m`, Max = `1.3947 m`
- **RF 3D Position Error**: Mean = `0.7991 m`, RMSE = `0.8445 m`, Max = `1.4985 m`

### 8th-Day Forward Predictions:
- `LSTM_x_error (m)`: `-0.545221 m`
- `LSTM_y_error (m)`: `0.287035 m`
- `LSTM_z_error (m)`: `-0.019755 m`
- `LSTM_satclockerror (m)`: `0.059530 m`
- `RF_x_error (m)`: `-0.598252 m`
- `RF_y_error (m)`: `-0.109192 m`
- `RF_z_error (m)`: `0.022156 m`
- `RF_satclockerror (m)`: `0.162497 m`

## Dataset: `DATA_MEO_Train2.csv` (MEO Orbit, 244 Epochs)
- **Time Span**: `2025-09-03 10:11:00` to `2025-09-09 11:41:00`

| Target Residual | Model | MAE (m) | RMSE (m) | Max Error (m) | Shapiro-Wilk W | p-value | Normal? |
|---|---|---|---|---|---|---|---|
| `x_error (m)` | **Pretrained LSTM** | 0.5578 | 0.5640 | 0.6564 | 0.6012 | 3.4520e-10 | No |
| `x_error (m)` | Random Forest | 0.0839 | 0.1026 | 0.3888 | 0.6539 | 2.2164e-09 | No |
| `y_error (m)` | **Pretrained LSTM** | 0.5571 | 0.5628 | 0.9799 | 0.7287 | 4.4190e-08 | No |
| `y_error (m)` | Random Forest | 0.0826 | 0.1176 | 0.4670 | 0.9270 | 5.2964e-03 | No |
| `z_error (m)` | **Pretrained LSTM** | 1.1757 | 1.1838 | 1.9467 | 0.6810 | 6.2296e-09 | No |
| `z_error (m)` | Random Forest | 0.0671 | 0.1493 | 0.9358 | 0.5056 | 1.7303e-11 | No |
| `satclockerror (m)` | **Pretrained LSTM** | 0.2262 | 0.2270 | 0.2720 | 0.9611 | 1.1210e-01 | Yes |
| `satclockerror (m)` | Random Forest | 0.0104 | 0.0164 | 0.0692 | 0.8727 | 9.2598e-05 | No |

- **LSTM 3D Position Error**: Mean = `1.4215 m`, RMSE = `1.4270 m`, Max = `2.0755 m`
- **RF 3D Position Error**: Mean = `0.1577 m`, RMSE = `0.2160 m`, Max = `1.0137 m`

### 8th-Day Forward Predictions:
- `LSTM_x_error (m)`: `-0.660204 m`
- `LSTM_y_error (m)`: `0.469388 m`
- `LSTM_z_error (m)`: `-1.065212 m`
- `LSTM_satclockerror (m)`: `0.223869 m`
- `RF_x_error (m)`: `-0.087340 m`
- `RF_y_error (m)`: `-0.105144 m`
- `RF_z_error (m)`: `0.096132 m`
- `RF_satclockerror (m)`: `-0.020272 m`
