"""
One-time data setup for SocialMap.

Downloads and processes:
  1. US county boundaries (Census Bureau, 500k simplified shapefile)
  2. Canada census division boundaries (Statistics Canada)
  3. US population (Census Bureau ACS 5-year estimates)
  4. Facebook Social Connectedness Index (user-supplied TSV)

Usage:
  python data/setup.py                         # geography + population only
  python data/setup.py path/to/county_county.tsv  # + SCI data

Facebook SCI download (free, registration required):
  https://data.humdata.org/dataset/social-connectedness-index
  File needed: county_county.tsv
"""

import io
import json
import math
import os
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "socialmap.db"
GEOJSON_PATH = DATA_DIR / "counties.geojson"
TMP_DIR = DATA_DIR / "tmp"

US_COUNTY_SHP_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"
)
CENSUS_POP_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/counties/totals/co-est2023-alldata.csv"
)
CANADA_DIV_URL = (
    "https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/"
    "boundary-limites/files-fichiers/lcd_000b21a_e.zip"
)

# Province FIPS-style prefix (first 2 digits of CDUID) → name
PROVINCE_NAMES = {
    "10": "Newfoundland and Labrador",
    "11": "Prince Edward Island",
    "12": "Nova Scotia",
    "13": "New Brunswick",
    "24": "Quebec",
    "35": "Ontario",
    "46": "Manitoba",
    "47": "Saskatchewan",
    "48": "Alberta",
    "59": "British Columbia",
    "60": "Yukon",
    "61": "Northwest Territories",
    "62": "Nunavut",
}

# US state FIPS → name (50 states + DC)
STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

# Territories to exclude from US counties
EXCLUDED_STATE_FIPS = {"60", "66", "69", "72", "78"}


def download_bytes(url: str, desc: str) -> io.BytesIO:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
            pbar.update(len(chunk))
    buf.seek(0)
    return buf


def load_us_counties() -> gpd.GeoDataFrame:
    print("\n[1/4] Downloading US county boundaries...")
    buf = download_bytes(US_COUNTY_SHP_URL, "US shapefile")

    us_tmp = TMP_DIR / "us"
    us_tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(us_tmp)

    shp = next(us_tmp.glob("*.shp"))
    gdf = gpd.read_file(shp).to_crs("EPSG:4326")
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDED_STATE_FIPS)].copy()

    gdf["fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]
    gdf["name"] = gdf["NAME"]
    gdf["state"] = gdf["STATEFP"].map(STATE_NAMES).fillna("Unknown")
    gdf["country"] = "US"

    print(f"   {len(gdf)} US counties loaded.")
    return gdf[["fips", "name", "state", "country", "geometry"]]


def load_us_population() -> pd.DataFrame:
    print("\n[2/4] Fetching US population from Census Bureau population estimates...")
    resp = requests.get(CENSUS_POP_URL, timeout=60)
    resp.raise_for_status()
    import io
    df = pd.read_csv(io.BytesIO(resp.content), encoding="latin-1")
    # COUNTY == 0 rows are state totals; exclude them
    df = df[df["COUNTY"] != 0].copy()
    df["fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(str).str.zfill(3)
    df["population"] = pd.to_numeric(df["POPESTIMATE2023"], errors="coerce").fillna(0).astype(int)
    print(f"   {len(df)} county populations fetched.")
    return df[["fips", "population"]]


def load_canada_divisions() -> gpd.GeoDataFrame | None:
    print("\n[3/4] Downloading Canada census division boundaries...")
    try:
        buf = download_bytes(CANADA_DIV_URL, "Canada shapefile")
        ca_tmp = TMP_DIR / "ca"
        ca_tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buf) as z:
            z.extractall(ca_tmp)

        shp = next(ca_tmp.glob("*.shp"))
        gdf = gpd.read_file(shp).to_crs("EPSG:4326")

        gdf["fips"] = "CA" + gdf["CDUID"].astype(str)
        gdf["name"] = gdf["CDNAME"].astype(str)
        gdf["state"] = gdf["CDUID"].astype(str).str[:2].map(PROVINCE_NAMES).fillna("Unknown")
        gdf["country"] = "CA"

        print(f"   {len(gdf)} Canadian census divisions loaded.")
        return gdf[["fips", "name", "state", "country", "geometry"]]
    except Exception as exc:
        print(f"   Warning: could not load Canada data ({exc}). Skipping.")
        return None


def build_geojson_and_db(
    us_gdf: gpd.GeoDataFrame,
    us_pop: pd.DataFrame,
    ca_gdf: gpd.GeoDataFrame | None,
):
    print("\n[4/4] Building GeoJSON and database...")

    # Merge population into US
    us = us_gdf.merge(us_pop, on="fips", how="left")
    us["population"] = us["population"].fillna(0).astype(int)

    frames = [us]
    if ca_gdf is not None:
        ca = ca_gdf.copy()
        ca["population"] = 0  # Stats Canada population not yet fetched
        frames.append(ca)

    combined: gpd.GeoDataFrame = pd.concat(frames, ignore_index=True)  # type: ignore[assignment]

    # Simplify geometry for web performance (~0.01 degree tolerance ≈ 1 km)
    combined["geometry"] = combined["geometry"].simplify(0.01, preserve_topology=True)

    # Write GeoJSON
    features = []
    for _, row in combined.iterrows():
        if row["geometry"] is None or row["geometry"].is_empty:
            continue
        features.append({
            "type": "Feature",
            "id": row["fips"],
            "properties": {
                "fips": row["fips"],
                "name": row["name"],
                "state": row["state"],
                "population": int(row["population"]),
                "country": row["country"],
            },
            "geometry": row["geometry"].__geo_interface__,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(GEOJSON_PATH, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))
    print(f"   GeoJSON saved: {len(features)} features → {GEOJSON_PATH.name}")

    # Write counties table to SQLite
    conn = sqlite3.connect(DB_PATH)
    counties_df = combined[["fips", "name", "state", "population", "country"]].copy()
    counties_df.to_sql("counties", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_counties_name ON counties(name COLLATE NOCASE)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_counties_fips ON counties(fips)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sci (
            fips_a TEXT NOT NULL,
            fips_b TEXT NOT NULL,
            sci     REAL NOT NULL,
            PRIMARY KEY (fips_a, fips_b)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sci_a ON sci(fips_a)")
    conn.commit()
    conn.close()
    print(f"   Database created: {DB_PATH.name}")

    return counties_df


def _detect_format(path: Path) -> tuple[str, str, str]:
    """Return (separator, fips_a_col, fips_b_col) by inspecting the header."""
    with open(path, newline="", encoding="utf-8") as f:
        header = f.readline().strip()
    sep = "\t" if "\t" in header else ","
    cols = header.split(sep)
    if "user_loc" in cols:
        return sep, "user_loc", "fr_loc"
    if "user_region" in cols:
        return sep, "user_region", "friend_region"
    raise ValueError(f"Unrecognised SCI file columns: {cols}")


def load_sci_data(sci_file: str):
    """
    Ingest Facebook SCI data (CSV or TSV) into the database.

    Supported column layouts:
      - user_loc / fr_loc / scaled_sci          (original county_county.tsv)
      - user_region / friend_region / scaled_sci (us_counties.csv with country cols)

    Scores are log-normalised to 0-100.  Both pair directions are kept so
    that querying by fips_a alone returns all connections for a county.
    Self-pairs (fips_a == fips_b) are dropped.
    """
    path = Path(sci_file)
    if not path.exists():
        print(f"\nERROR: SCI file not found: {sci_file}")
        print("Download from: https://data.humdata.org/dataset/social-connectedness-index")
        return

    sep, col_a, col_b = _detect_format(path)
    print(f"\nLoading SCI data from {path.name}  (sep={'TAB' if sep == chr(9) else 'COMMA'}, "
          f"fips cols: {col_a}/{col_b})")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sci")
    conn.commit()

    chunk_size = 200_000
    has_country_cols = "user_country" in open(path, encoding="utf-8").readline()

    # Pass 1: find max raw SCI across US-US pairs (excluding self-pairs)
    print("   Pass 1/2: scanning for max SCI value...")
    raw_max = 0.0
    for chunk in pd.read_csv(path, sep=sep, chunksize=chunk_size):
        if has_country_cols:
            chunk = chunk[(chunk["user_country"] == "US") & (chunk["friend_country"] == "US")]
        chunk = chunk[chunk[col_a] != chunk[col_b]]
        vals = pd.to_numeric(chunk["scaled_sci"], errors="coerce").fillna(0)
        if not vals.empty and vals.max() > raw_max:
            raw_max = vals.max()

    if raw_max == 0:
        print("   ERROR: no valid SCI values found. Check file format.")
        conn.close()
        return

    log_max = math.log1p(raw_max)
    print(f"   Max raw SCI: {raw_max:,.0f}  →  log-normalising to 0–100")

    # Pass 2: normalise and insert
    print("   Pass 2/2: inserting normalised scores...")
    total_rows = 0
    for chunk in tqdm(
        pd.read_csv(path, sep=sep, chunksize=chunk_size),
        desc="   chunks",
    ):
        if has_country_cols:
            chunk = chunk[(chunk["user_country"] == "US") & (chunk["friend_country"] == "US")]
        chunk = chunk[chunk[col_a] != chunk[col_b]].copy()
        chunk["fips_a"] = chunk[col_a].astype(str).str.zfill(5)
        chunk["fips_b"] = chunk[col_b].astype(str).str.zfill(5)
        chunk["sci"] = pd.to_numeric(chunk["scaled_sci"], errors="coerce").fillna(0)
        chunk["sci"] = (np.log1p(chunk["sci"]) / log_max * 100).round(2)
        chunk[["fips_a", "fips_b", "sci"]].to_sql(
            "sci", conn, if_exists="append", index=False,
            method="multi", chunksize=5_000,
        )
        total_rows += len(chunk)

    conn.commit()
    conn.close()
    print(f"   Loaded {total_rows:,} SCI records.")


if __name__ == "__main__":
    TMP_DIR.mkdir(exist_ok=True)

    us_gdf = load_us_counties()
    us_pop = load_us_population()
    ca_gdf = load_canada_divisions()
    build_geojson_and_db(us_gdf, us_pop, ca_gdf)

    if len(sys.argv) > 1:
        load_sci_data(sys.argv[1])
    else:
        print("\nSkipping SCI data (no file provided).")
        print("Re-run with: python data/setup.py path/to/county_county.tsv")

    # Clean up temp files
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\nSetup complete. Run: python app.py")
