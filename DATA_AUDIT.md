# Data audit and scientific status

Audit date: 2026-08-17  
Dataset SHA-256: `244dd0ba6e0e9ef7f8ba5e8a3e38b235851c4c12c82f3cb6a22fb9e98c5fbbd1`

`FINAL_Data.csv` is usable for pipeline tests, but it is not currently fit for a
scientific model-performance claim.

## Confirmed defects

- 497 `Modelled_Clock` values are approximately 1 second. This is the converted
  SP3 missing-clock sentinel `999999.999999` microseconds, not a physical clock
  target. The corrected pipeline masks these labels and keeps the timestamps.
- 20.08% of rows have a three-dimensional orbit residual of at least 1 km.
  At 154 epochs, at least 95% of satellites cross that threshold together.
- The kilometre-event indicator has mean lag-5 correlation 0.968. At 15-minute
  cadence this is a 75-minute repeating pattern, strongly indicating an
  ingestion, interpolation, or epoch-alignment defect.
- There are 14 non-15-minute per-satellite intervals. Corrected window builders
  purge any lookback/label block crossing a gap.
- `3D_Orbit_Error` is the norm of `Error_X/Y/Z` to numerical precision. It is
  derived during evaluation and is no longer an independently learned target.

Run the executable audit with:

```powershell
.venv\Scripts\python.exe audit_data.py --data FINAL_Data.csv --strict
```

`--strict` intentionally returns exit code 2 for the bundled CSV. A JSON report
is still written unless `--report` points elsewhere.

## What was repaired in code

- SP3 clock sentinels become explicit target-availability masks.
- No target-magnitude row filtering occurs before training or evaluation.
- Feature and target scalers are fitted before the validation boundary only.
- Feature and target arrays are transformed separately exactly once.
- Train, validation, and test labels occupy disjoint chronological blocks.
- Boundary-crossing and non-contiguous windows are purged.
- The satellite vocabulary depends on usable training history, not future test
  completeness.
- Metrics retain physical units, derive vector 3D error, and report coverage.

On the bundled CSV this produces 19,260 training windows, 1,113 validation
windows, and 52 test origins. The report names every evaluated satellite; it no
longer silently drops difficult target rows.

## What cannot be repaired from this repository alone

The repository contains the derived CSV but not the source RINEX navigation,
SP3/CLK products, download timestamps, or the code that joined/interpolated
them. The orbit defect therefore cannot be corrected honestly by a downstream
filter. Rebuild the CSV from the source products and record:

1. source URLs, product issue/download time, and hashes;
2. GPS/UTC/GLONASS time-system conversions and leap seconds;
3. exact epoch join and interpolation/extrapolation policy;
4. SP3 units, missing values, accuracy fields, prediction/event/maneuver flags;
5. antenna phase-center/center-of-mass reference and applied biases;
6. what information was available at each forecast issue time.

The machine-readable contract is in `configs/data_contract.json`. The promotion
policy remains fail-closed in `configs/promotion_policy.json` until clean source
data and multiple future test days are available.

## Authoritative references

- [IGS SP3-d format specification](https://files.igs.org/pub/data/format/sp3d.pdf)
- [IGS product accuracy and latency](https://igs.org/products/)
- [IGS MGEX data products](https://igs.org/mgex/data-products/)
- [Multi-GNSS SISRE evaluation](https://elib.dlr.de/92092/)

