"""
data_clean.py
Phase 2 — Build the unified monthly master dataset.

Loads the five raw CSVs produced by data_fetch.py, aligns them on a monthly
DatetimeIndex spanning 2010-01 through 2025-10, computes real exchange rates
(2015=100), tags shock periods, and writes data/processed/master_monthly.csv.
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
INDEX_START = "2010-01-01"
INDEX_END = "2025-10-01"
BASE_YEAR = 2015

# nominal TRY rate ↔ foreign-country CPI used to deflate it
FX_TO_FOREIGN_CPI = {
    "EUR_TRY": "DE_TUFE",
    "GBP_TRY": "GB_TUFE",
    "USD_TRY": "US_TUFE",
    "RUB_TRY": "RU_TUFE",
}


def _load_indexed_csv(path: str, index_col: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col, parse_dates=True)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df


def load_raw() -> dict:
    evds = _load_indexed_csv(f"{RAW_DIR}/evds_raw.csv")
    fred = _load_indexed_csv(f"{RAW_DIR}/fred_raw.csv")
    trends = _load_indexed_csv(f"{RAW_DIR}/trends_raw.csv")
    tuik_monthly = pd.read_csv(f"{RAW_DIR}/tuik_monthly.csv", parse_dates=["date"])
    tuik_monthly = tuik_monthly.set_index("date").rename(columns={"arrivals": "arrivals_total"})
    tuik_monthly.index = tuik_monthly.index.to_period("M").to_timestamp()
    return {"evds": evds, "fred": fred, "trends": trends, "tuik_monthly": tuik_monthly}


def compute_real_fx(merged: pd.DataFrame, base_year: int = BASE_YEAR) -> pd.DataFrame:
    """Add real_<CCY>_TRY = nominal * (foreign_CPI / TR_TUFE), rebased to base_year=100.

    NaN propagates where either CPI is missing — notably RU after 2022-03.
    """
    out = merged.copy()
    for fx_col, foreign_cpi in FX_TO_FOREIGN_CPI.items():
        if fx_col not in out or foreign_cpi not in out or "TR_TUFE" not in out:
            continue
        real = out[fx_col] * (out[foreign_cpi] / out["TR_TUFE"])
        base = real.loc[str(base_year)].mean()
        if pd.isna(base) or base == 0:
            continue
        out[f"real_{fx_col}"] = real / base * 100
    return out


def add_shock_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags: covid (2020-03..2021-06), russia_war (2022-02..),
    mideast (2023-10..)."""
    out = df.copy()
    idx = out.index
    out["covid"] = ((idx >= "2020-03-01") & (idx <= "2021-06-01")).astype(int)
    out["russia_war"] = (idx >= "2022-02-01").astype(int)
    out["mideast"] = (idx >= "2023-10-01").astype(int)
    return out


def build_master() -> pd.DataFrame:
    sources = load_raw()
    full_index = pd.date_range(INDEX_START, INDEX_END, freq="MS")

    merged = pd.concat(
        [
            sources["evds"].reindex(full_index),
            sources["fred"].reindex(full_index),
            sources["trends"].reindex(full_index),
            sources["tuik_monthly"].reindex(full_index),
        ],
        axis=1,
    )
    merged.index.name = "date"

    merged = compute_real_fx(merged)
    merged = add_shock_flags(merged)
    return merged


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master = build_master()
    out_path = f"{PROCESSED_DIR}/master_monthly.csv"
    master.to_csv(out_path)
    print(f"→ {out_path}")
    print(f"shape: {master.shape}")
    print(f"date range: {master.index.min().date()} → {master.index.max().date()}")
    print()
    print("columns:")
    for c in master.columns:
        n = master[c].count()
        print(f"  {c:35s} non-null={n:>3}/{len(master)}")
    print()
    print("head:")
    print(master.head().to_string())
