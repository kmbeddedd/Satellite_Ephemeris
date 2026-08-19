"""Download broadcast RINEX navigation and precise SP3/CLK files from open IGS/MGEX mirrors."""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error 

# Free, open public mirrors for IGS / MGEX data (no authentication required)
MIRRORS = {
    "bkg": "https://igs.bkg.bund.de/root_ftp/IGS",
    "ign": "https://geodesie.ign.fr/data/pub/igs",
    "whu": "ftp://igs.gnssm.cn/pub/gnss/data",
}


def date_to_gps_week_day(dt: datetime) -> tuple[int, int, int]:
    """Convert UTC datetime to (GPS week, day of week 0-6, day of year 1-366)."""
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt - epoch
    total_days = delta.days
    gps_week = total_days // 7
    gps_day = total_days % 7
    day_of_year = dt.timetuple().tm_yday
    return gps_week, gps_day, day_of_year


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a file with user-agent header and stream to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Satellite-ML-Data-Acquisition/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status not in (200, 206):
                return False
            with open(output_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [Notice] Download failed from {url}: {exc}")
        if output_path.exists():
            output_path.unlink()
        return False


def decompress_gz(gz_path: Path, target_path: Path) -> bool:
    """Decompress a .gz file to target path."""
    try:
        with gzip.open(gz_path, "rb") as f_in, open(target_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as exc:
        print(f"  [Error] Decompression failed for {gz_path}: {exc}")
        return False


def fetch_multi_gnss_products(
    target_date: datetime,
    out_dir: Path,
    agency: str = "WUM",
    prefer_mirror: str = "bkg",
) -> dict[str, Path | None]:
    """Fetch daily Multi-GNSS broadcast ephemeris and precise SP3/CLK files.

    Parameters:
        target_date: UTC datetime for target data day.
        out_dir: Destination directory.
        agency: MGEX analysis center ('WUM' for Wuhan, 'COD' for CODE, 'GFZ' for GFZ, 'GRG' for CNES).
        prefer_mirror: Preferred mirror key from MIRRORS.
    """
    gps_week, gps_day, doy = date_to_gps_week_day(target_date)
    year = target_date.year
    yy = str(year)[2:]
    doy_str = f"{doy:03d}"

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {"brdc": None, "sp3": None, "clk": None}

    # 1. Download Multi-GNSS Broadcast Navigation file (BRDM or BRDC)
    # Long format: BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx.gz
    # Short format: brdmDDD0.YYp.Z or brdcDDD0.YYn.Z
    brdc_filename = f"BRDC00IGS_R_{year}{doy_str}0000_01D_MN.rnx.gz"
    brdc_target = out_dir / brdc_filename
    brdc_decompressed = out_dir / f"BRDC00IGS_R_{year}{doy_str}0000_01D_MN.rnx"

    if brdc_decompressed.exists():
        print(f"[OK] Broadcast ephemeris already cached: {brdc_decompressed.name}")
        results["brdc"] = brdc_decompressed
    else:
        print(f"Downloading Broadcast Navigation (BRDC) for Day {doy_str}, Year {year}...")
        url = f"{MIRRORS[prefer_mirror]}/BRDC/{year}/{doy_str}/{brdc_filename}"
        if download_file(url, brdc_target):
            decompress_gz(brdc_target, brdc_decompressed)
            results["brdc"] = brdc_decompressed
            print(f"[OK] Saved: {brdc_decompressed.name}")

    # 2. Download Precise MGEX SP3 Orbit product (e.g. WUM0MGXFIN / GFZ0MGXRAP / COD0MGXFIN)
    sp3_filename = f"{agency}0MGXFIN_{year}{doy_str}0000_01D_05M_ORB.SP3.gz"
    sp3_target = out_dir / sp3_filename
    sp3_decompressed = out_dir / f"{agency}0MGXFIN_{year}{doy_str}0000_01D_05M_ORB.SP3"

    if sp3_decompressed.exists():
        print(f"[OK] Precise SP3 orbit already cached: {sp3_decompressed.name}")
        results["sp3"] = sp3_decompressed
    else:
        print(f"Downloading Precise SP3 Orbits ({agency}) for GPS Week {gps_week} Day {gps_day}...")
        url = f"{MIRRORS[prefer_mirror]}/products/mgex/{gps_week}/{sp3_filename}"
        if download_file(url, sp3_target):
            decompress_gz(sp3_target, sp3_decompressed)
            results["sp3"] = sp3_decompressed
            print(f"[OK] Saved: {sp3_decompressed.name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official Multi-GNSS and NavIC RINEX/SP3 data")
    parser.add_argument("--date", default="2026-01-15", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--out", default="data_acquisition/raw", help="Output directory")
    parser.add_argument("--agency", default="WUM", help="MGEX Analysis Center (WUM, COD, GFZ, GRG)")
    args = parser.parse_args()

    dt = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_dir = Path(args.out)
    print(f"--- Fetching IGS/MGEX Data for Date {args.date} (Analysis Center: {args.agency}) ---")
    results = fetch_multi_gnss_products(dt, out_dir, agency=args.agency)
    print("\nFetch Summary:")
    for k, v in results.items():
        print(f"  {k.upper()}: {v if v else 'Not fetched (check network or alternate date)'}")


if __name__ == "__main__":
    main()
