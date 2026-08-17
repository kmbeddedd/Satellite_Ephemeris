\## Bottom line



Do not make the network larger yet. The current project does not demonstrate reliable forecasting skill: data corruption, double scaling, validation leakage, and inconsistent metrics make the reported improvements unreliable. Fixing those issues should improve the system far more than adding layers.



I audited the current working tree and researched primary sources through 17 August 2026. I made no changes.



\## What the saved models actually achieve



On the same 35-satellite Day-8 cohort:



| Target | No-correction baseline | Saved BiLSTM | Hybrid after correcting its double inverse-scaling |

|---|---:|---:|---:|

| X | 2,016 m | 2,067 m | 2,483 m |

| Y | 2,033 m | 2,098 m | 3,735 m |

| Z | 2,008 m | 2,057 m | 3,152 m |

| Clock | 0.01042 s | 0.01144 s | 0.08407 s |



The physically meaningful mean XYZ vector error is:



\- No correction: \*\*4.024 km\*\*

\- BiLSTM: \*\*4.134 km\*\*

\- Hybrid: \*\*6.364 km\*\*



So both learned models currently lose to simply predicting zero correction.



The BiLSTM’s apparently better `3D\_Orbit\_Error` score is misleading: it predicts the norm independently from XYZ. Its predicted norm differs from `sqrt(X²+Y²+Z²)` by an average of \*\*3.61 km\*\*, and about 30% of its predicted norms are negative.



\## Critical problems found



1\. \*\*The clock target contains missing-data sentinels.\*\*



&#x20;  There are 497 rows where the precise clock is approximately `0.999999999999 s`; 357 occur among complete satellites. This is the SP3 missing-clock sentinel `999999.999999 µs`, which the \[official IGS SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf) says represents a bad or absent clock—not a one-second clock error. The model is currently trained to reproduce missing data.



2\. \*\*The orbit targets show a repeating ingestion/alignment artifact.\*\*



&#x20;  The median 3D error is only 0.028 m, but 7,900 of 39,168 complete-satellite rows—20.2%—exceed 1 km, with a strong five-sample periodic pattern. At 152 timestamps, all 51 satellites exceed 1 km together.



&#x20;  This is not plausible GNSS behaviour. A one-year multi-GNSS study reported average broadcast SISRE around 0.7 m for GPS and 1.9 m for GLONASS, although SISRE is not identical to this project’s ECEF metric. \[DLR/GPS Solutions study](https://elib.dlr.de/92092/)



3\. \*\*Hybrid targets are standardized twice.\*\*



&#x20;  \[src/data.py](/D:/Education/Project/Satellite%20ML/src/data.py:227) scales feature columns, including the four targets, and immediately scales those columns again at lines 230–231. \[train\_transformer.py](/D:/Education/Project/Satellite%20ML/train\_transformer.py:306) reverses only one scaling pass.



&#x20;  Consequently, values such as `0.471` in the hybrid metrics JSON are standardized values mislabeled as metres.



4\. \*\*Validation is heavily contaminated.\*\*



&#x20;  - BiLSTM: randomized overlapping windows; 100% of validation windows reuse timestamps present in training targets.

&#x20;  - Hybrid: every validation window shares training labels, averaging 61.9% of its 96 target steps.

&#x20;  - Scalers are fitted using the future validation period.



&#x20;  The scheduler, early stopping, and Optuna tuning therefore optimize against an optimistic validation set.



5\. \*\*Target-based filtering biases the test.\*\*



&#x20;  \[src/data.py](/D:/Education/Project/Satellite%20ML/src/data.py:43) deletes rows using future `3D\_Orbit\_Error < 50 km`. This creates irregular time gaps and removes 16 difficult satellites from Day-8 evaluation. \[train\_bilstm.py](/D:/Education/Project/Satellite%20ML/train\_bilstm.py:193) then silently evaluates only 35 of the 51 supposedly complete satellites.



6\. \*\*Metrics are inconsistent or dimensionally invalid.\*\*



&#x20;  \[src/evaluate.py](/D:/Education/Project/Satellite%20ML/src/evaluate.py:45) defines a BiLSTM horizon as cumulative steps `1..h`, while the hybrid scorer uses only endpoint `h`. `Overall\_MAE` also averages metres and seconds together.



7\. \*\*The diffusion component is not validly evaluated.\*\*



&#x20;  It trains on residuals `y−μ`, but sampling feeds `μ+x` to the denoiser. It also starts from `0.02×noise`, inconsistent with the training schedule. Diffusion never contributes to the reported test metrics; it only produces a plot. See \[pytorch\_diffusion.py](/D:/Education/Project/Satellite%20ML/src/models/pytorch\_diffusion.py:87).



\## Recommended improvement order



| Priority | Improvement | Expected value |

|---|---|---|

| P0 | Rebuild and validate the dataset | Essential |

| P1 | Replace the validation and evaluation protocol | Essential |

| P2 | Reformulate targets around GNSS physics | Very high |

| P3 | Establish strong simple baselines | Very high |

| P4 | Compare compact modern architectures | Medium–high |

| P5 | Add calibrated uncertainty | Medium |



\### P0 — Rebuild the data



Regenerate the CSV from point-in-time broadcast RINEX and precise SP3/CLK products:



\- Reject SP3 missing position/clock sentinels before unit conversion.

\- Preserve SP3 clock-event, prediction, maneuver, and accuracy flags.

\- Verify GPS/UTC time systems, leap seconds, epoch joins, interpolation, units, antenna reference points, clock datums, and differential code biases.

\- Keep an exact 15-minute grid with validity masks. Never delete rows and then pretend adjacent rows remain 15 minutes apart.

\- Record data source, product type, generation time and hash.



IGS/MGEX provides broadcast ephemerides and precise orbit/clock products going back many years, including 5-minute SP3 and 30-second CLK products. Train on months or years rather than seven days. \[IGS MGEX data and products](https://igs.org/mgex/data-products/)



Also define what information exists when a forecast is issued. IGS final products have roughly 12–19 days of latency, rapid products 17–41 hours, while ultra-rapid products contain observed and predicted halves. Historical “true error” features may therefore be unavailable for the intended real-time use case. \[Official IGS product accuracy and latency](https://igs.org/products/)



\### P1 — Make evaluation trustworthy



Use raw-time rolling-origin folds before scaling or windowing:



\- Train through one date, validate on the following untouched 24-hour target.

\- Ensure validation target timestamps never occur in training targets.

\- Fit scalers only on each fold’s training portion.

\- Use many forecast dates across seasons and retain a final untouched period.

\- Evaluate all satellites, plus held-out-satellite and held-out-block tests.

\- Report GPS/GLONASS, satellite, normal/event, eclipse and ephemeris-age slices.



Include zero, persistence, seasonal-96, drift, quadratic clock, ARIMA/Kalman, DLinear/NLinear and operational IGS ultra-rapid baselines. A complex model should not be promoted unless it beats the strongest of these at every required horizon.



\### P2 — Reformulate the prediction problem



For orbit:



\- Transform ECEF errors into radial/in-track/cross-track, or RAC/RIC.

\- Predict a correction to a broadcast or numerical propagation baseline.

\- Derive 3D error from predicted XYZ/RIC rather than predicting it separately.

\- Add ephemeris age, Toe/IODE, state and velocity, orbital elements, argument of latitude, Sun angle, beta angle, eclipse/yaw state, EOP, block type and maneuver flags.



A January 2026 ION study uses RIC residuals, direct multi-horizon BiLSTM/N-HiTS outputs, and a soft two-body plus J2 constraint—very close to the appropriate direction for this project. \[Physics-constrained GNSS forecasting study](https://www.ion.org/publications/abstract.cfm?articleID=20520)



For clock:



\- Use a separate model/head.

\- Work in nanoseconds or range-equivalent metres.

\- Predict first differences/frequency residuals around a polynomial or Kalman trend.

\- Slice evaluation by clock type and satellite block.



Recent GPS clock work uses 30-second IGS labels and compares Transformer forecasts against ARIMA, LSTM and N-BEATS, achieving its reported results only out to two hours—not automatically at this project’s 24-hour horizon. \[Measurement, June 2026](https://www.sciencedirect.com/science/article/pii/S0263224126013175)



\### P3/P4 — Architecture experiments



After the data and scorecard are fixed, test in this order:



1\. Linear, NLinear and DLinear.

2\. N-HiTS or a modest TCN.

3\. PatchTST or iTransformer with longer, multiday history.

4\. A compact physics-residual hybrid.

5\. The current BiLSTM–GRU–MHSA stack only if it wins controlled ablations.



Simple linear models have outperformed much more complex Transformers on several long-horizon benchmarks, making them necessary baselines. \[AAAI DLinear paper](https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089) N-HiTS is specifically designed for efficient multiscale long-horizon prediction and is already used in relevant GNSS work. \[AAAI N-HiTS paper](https://ojs.aaai.org/index.php/AAAI/article/view/25854)



Start with Huber/MAE or a clean per-channel likelihood. Add FFT, smoothness or DILATE terms one at a time only when an ablation proves they help.



\### P5 — Uncertainty



Use Student-t or quantile heads first, then calibrate them on a separate rolling temporal fold. Report CRPS, interval width and empirical 50/80/90/95% coverage per horizon. Multi-step adaptive conformal methods are designed for non-exchangeable time-series settings. \[Multi-step adaptive conformal inference](https://proceedings.mlr.press/v230/hallberg-szabadvary24a.html)



Keep diffusion only if corrected residual-space sampling beats those simpler methods on CRPS, energy score, coverage and downstream PPP performance.



A sensible promotion gate is: statistically significant improvement over both the strongest simple baseline and an operational GNSS baseline, at every required horizon, without important satellite/constellation/event regressions.



The workspace already contained uncommitted model changes and untracked test artifacts before this audit; I left them untouched.











































# PART-2

# 

Implemented all feasible repository changes.

Key outcomes:

Fixed SP3 clock sentinel masking, target double-scaling, cadence gaps, target-based filtering, validation leakage, and train-only scaler fitting in \[src/data.py (line 543)](/D:/Education/Project/Satellite ML/src/data.py:543).

Added unit-aware metrics, uncertainty scoring, satellite/constellation slices, executable baselines, and fail-closed promotion gates in \[src/evaluate.py (line 316)](/D:/Education/Project/Satellite ML/src/evaluate.py:316).

Hardened Transformer/BiLSTM training with masked losses, Student-t forecasts, conformal calibration, best checkpoints, reproducible artifacts, and corrected diffusion in \[train\_transformer.py (line 436)](/D:/Education/Project/Satellite ML/train\_transformer.py:436) and \[train\_bilstm.py (line 228)](/D:/Education/Project/Satellite ML/train\_bilstm.py:228).

Added seeded, leakage-safe Optuna tuning, ECEF↔RIC utilities, data contracts, locked dependencies, CLI audit/baseline commands, and comprehensive tests.

Replaced misleading performance claims with an honest project status in \[README.md (line 7)](/D:/Education/Project/Satellite ML/README.md:7) and \[DATA\_AUDIT.md (line 9)](/D:/Education/Project/Satellite ML/DATA\_AUDIT.md:9).

Verification:

32 passed tests.

Transformer, BiLSTM, Optuna, baseline, checkpoint reload, strict audit, and diffusion smoke paths all completed.

Corrected dataset: 19,260 train, 1,113 validation, and 52 test windows.

Promotion correctly fails for the smoke models.

Strict audit correctly exits 2.

One upstream limitation remains: the bundled CSV has 497 converted SP3 missing-clock sentinels and a repeating synchronous kilometre-scale orbit defect. The code now handles or detects these safely, but correcting the orbit values requires the original RINEX/SP3/CLK source products and ingestion pipeline. This conclusion follows the official SP3-d specification and IGS product guidance.

The production ML workflow directly shaped the explicit data contract, reproducible model bundles, measurable promotion gates, and fail-closed behavior. Existing historical result artifacts and unrelated untracked files were preserved.



