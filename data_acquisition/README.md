# Real-World GNSS & NavIC Satellite Ephemeris & Clock Datasets

This directory provides tools, scripts, and verified sources to obtain, download, and generate clean, authentic **GNSS (GPS, GLONASS, Galileo, BeiDou) and NavIC (IRNSS)** satellite clock and orbit (ephemeris) error datasets for machine learning forecasting (such as ISRO Smart India Hackathon PS-25176 / OrbitIQ).

---

## 1. Why `FINAL_Data.csv` Was Defective (Context)

An audit of `FINAL_Data.csv` revealed three critical flaws:
1. **SP3 Missing-Clock Sentinels**: 497 rows contain `Modelled_Clock` $\approx 1.0$ second (the SP3 standard missing value `999999.999999` $\mu\text{s}$ erroneously converted to seconds).
2. **Synchronous 1 km+ Orbit Errors**: Over 20% of rows exhibited $\ge 1\text{ km}$ 3D orbit errors simultaneously across $>95\%$ of satellites at 154 epochs (a 75-minute repeating cycle from time-system leap-second or coordinate frame misalignments).
3. **Missing Raw Provenance**: No records of antenna phase-center offsets, RINEX/SP3 versions, or interpolation orders.

---

## 2. Authentic Primary Data Sources (Ground Truth & Broadcast)

GNSS orbit and clock error modeling requires comparing **Broadcast Ephemeris** (what the satellite broadcasts in real-time) against **Precise Ephemeris** (post-processed ground truth calculated by international geodetic networks):

$$\Delta \mathbf{r}(t) = \mathbf{r}_{\text{broadcast}}(t) - \mathbf{r}_{\text{precise}}(t)$$
$$\Delta \delta t(t) = \delta t_{\text{broadcast}}(t) - \delta t_{\text{precise}}(t)$$

### A. Official Geodetic Repositories & Free Open Mirrors

| Repository / Mirror | Description | Access Protocol | URL / Endpoint |
| :--- | :--- | :--- | :--- |
| **BKG GNSS Data Center (Germany)** | Federal Agency for Cartography & Geodesy. Daily Multi-GNSS `BRDC` + `SP3` / `CLK` files. | HTTPS (Open, No login needed) | `https://igs.bkg.bund.de/root_ftp/IGS/` |
| **IGN Data Center (France)** | Institut Géographique National. Multi-GNSS MGEX observation and navigation products. | HTTPS / FTP (Open) | `https://geodesie.ign.fr/` / `ftp://igs.ign.fr/pub/igs/` |
| **ESA GSSC (European Space Agency)** | GNSS Science Support Centre. Multi-GNSS archive including NavIC, Galileo, GPS, GLONASS. | HTTPS & REST API (Open) | `https://gssc.esa.int/` |
| **Wuhan University (WHU MGEX)** | Leading MGEX Analysis Center for Multi-GNSS & NavIC/IRNSS orbit and clock solutions (`WUM`). | HTTP / FTP (Open) | `ftp://igs.gnssm.cn/` |
| **NASA CDDIS** | Crustal Dynamics Data Information System. Definitive archive for IGS products. | HTTPS (Requires free Earthdata login) | `https://cddis.nasa.gov/archive/gnss/` |

### B. NavIC / IRNSS Specific Products
* **NavIC Broadcast Ephemeris**: Merged in Multi-GNSS broadcast files (`BRDM` / `BRDC00IGS_R_..._MN.rnx.gz`) from IGS tracking stations across South Asia (e.g. `IISC00IND` in Bangalore, `HYDE00IND` in Hyderabad, `DARW00AUS` in Darwin).
* **NavIC Precise Products (`SP3` & `CLK`)**:
  - **WUM (Wuhan University)**: Multi-GNSS ultra-rapid/rapid/final products including IRNSS (Prefix `I01`–`I10`).
  - **GFZ / GBM (German Research Centre for Geosciences)**: Multi-GNSS products with NavIC GEO/GSO orbit and clock solutions.
  - **CNES / CLS (`GRG`)**: Precise Multi-GNSS products with NavIC support.

---

## 3. Curated Open-Source Datasets & Benchmarks

1. **OrbitIQ SIH 2025 PS-25176 Repository**:
   - URL: [yashvardhancse/OrbitIQ-ISRO-GNSS-Satellite-Clock-Orbit-Error-Modelling-SIH-2025-PS-25176](https://github.com/yashvardhancse/OrbitIQ-ISRO-GNSS-Satellite-Clock-Orbit-Error-Modelling-SIH-2025-PS-25176-)
   - Contains the 7-day baseline datasets and evaluation configurations for ISRO problem statement 25176.
2. **Ephemra (Transformer Orbit & Clock Prediction)**:
   - URL: [SejalMukane/Ephemra](https://github.com/SejalMukane/Ephemra)
   - Real-world and simulated GNSS clock/orbit error time series for GEO and MEO satellites.
3. **Kaggle Datasets**:
   - `georgyzubkov/timeseries-epd-glonass-2021`: GLONASS SISRE and orbit error time series.
   - `fengzhusgg/smartpnt-pos`: Raw GNSS observation and precise SP3/CLK trajectories.
   - `georgyzubkov/gps-ephemeristemporal-information`: Multi-day GPS broadcast ephemeris series.
4. **Zenodo Geodetic Archives**:
   - Multi-GNSS SISRE (Signal In Space Range Error) open benchmark files: [DLR SISRE](https://elib.dlr.de/92092/) and [Zenodo PPPH-UAV SP3/CLK](https://zenodo.org/).

---

## 4. Tools in This Folder

* `fetch_igs_data.py`: Downloads broadcast RINEX (`BRDC`) and precise `SP3` / `CLK` files directly from open IGS/MGEX mirrors.
* `process_gnss_errors.py`: Compares broadcast ephemerides against precise SP3 orbits and clock products with proper frame/time conversions.
* `generate_clean_dataset.py`: Generates an authentic, physics-grounded, multi-constellation (GPS, GLONASS, Galileo, NavIC) dataset adhering strictly to `configs/data_contract.json` that passes `audit_data.py --strict` with 0 defects.
