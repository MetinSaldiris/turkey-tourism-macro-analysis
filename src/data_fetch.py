"""
data_fetch.py
Tüm veri kaynaklarından çekim fonksiyonları.
Çalıştırmadan önce: .env dosyasında EVDS_API_KEY ve FRED_API_KEY tanımlı olmalı.
"""

import os
import time
import unicodedata
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EVDS_KEY = os.getenv("EVDS_API_KEY")
FRED_KEY  = os.getenv("FRED_API_KEY")

START_DATE = "01-01-2010"
END_DATE   = "31-10-2025"
FRED_START = "2010-01-01"
FRED_END   = "2025-10-31"
TRENDS_TIMEFRAME = "2010-01-01 2025-10-31"

# ─────────────────────────────────────────
# 1. EVDS — Kur Serileri & TR TÜFE
# ─────────────────────────────────────────

EVDS_SERIES = {
    "EUR_TRY": "TP.DK.EUR.A.YTL",
    "GBP_TRY": "TP.DK.GBP.A.YTL",
    "USD_TRY": "TP.DK.USD.A.YTL",
    "RUB_TRY": "TP.DK.RUB.A.YTL",
    "TR_TUFE": "TP.FE.OKTG01",
}

def fetch_all_evds() -> pd.DataFrame:
    from evds import evdsAPI
    evds = evdsAPI(EVDS_KEY)
    series_codes = list(EVDS_SERIES.values())
    print(f"  Çekiliyor: {series_codes}")

    df = evds.get_data(
        series_codes,
        startdate=START_DATE,
        enddate=END_DATE,
        frequency=5,
        aggregation_types="avg"
    )

    # EVDS API replaces dots with underscores in returned column names.
    rename_map = {v.replace(".", "_"): k for k, v in EVDS_SERIES.items()}
    df = df.rename(columns=rename_map)

    if "Tarih" in df.columns:
        df["Tarih"] = pd.to_datetime(df["Tarih"], format="%Y-%m")
        df = df.set_index("Tarih")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    print(f"  EVDS OK — {df.shape}")
    return df


# ─────────────────────────────────────────
# 2. FRED — Kaynak Ülke TÜFE Serileri
# ─────────────────────────────────────────

FRED_SERIES = {
    # DE: Eurostat HICP, aylık endeks (base = Index 2025=100). compute_real_fx
    # 2015 = 100'e yeniden bazladığı için baz-yıl farkı sorun değil. Bu seri
    # OECD MEI'nin (DEUCPIALLMINMEI, 2025-03'te durdu) yerini alıyor.
    "DE_TUFE": "CP0000DEM086NEST",
    # GB: OECD MEI aylık endeksi (Index 2015=100). FRED üzerinde ONS kaynaklı,
    # halihazırda güncel bir aylık UK CPI ENDEKSİ serisi bulunamadığı için bu
    # seride kalıyoruz; yayın 2025-03'te durduğundan real_GBP_TRY o tarihte
    # kesiliyor. Bu limitasyon README/Limitations'da belgelendi.
    "GB_TUFE": "GBRCPIALLMINMEI",
    # US: BLS CPI-U all items, aylık endeks (Index 1982-84=100). Güncel.
    "US_TUFE": "CPIAUCSL",
    # OECD discontinued Russian data after 2022-03 (sanctions); series ends there.
    "RU_TUFE": "RUSCPIALLMINMEI",
}

def fetch_all_fred() -> pd.DataFrame:
    from fredapi import Fred
    fred = Fred(api_key=FRED_KEY)
    frames = []

    for name, code in FRED_SERIES.items():
        print(f"  FRED — {name}")
        try:
            s = fred.get_series(code, observation_start=FRED_START, observation_end=FRED_END)
            s = s.resample("MS").mean()
            s.name = name
            frames.append(s)
        except Exception as e:
            print(f"  HATA — {name}: {e}")

    df = pd.concat(frames, axis=1).sort_index()
    print(f"  FRED OK — {df.shape}")
    return df


# ─────────────────────────────────────────
# 3. Google Trends — Arama Hacmi
# ─────────────────────────────────────────

TRENDS_QUERIES = {
    "DE": ["Türkei Urlaub", "Antalya Hotel"],
    "GB": ["Turkey holiday", "Antalya hotel"],
    "SA": ["السياحة في تركيا"],
    "AE": ["سياحة تركيا"],
    "RU": ["Турция отдых", "Анталья отель"],
}

def _trends_with_retry(pytrends, keywords, geo, timeframe, max_attempts=5):
    """Build payload and fetch interest_over_time with exponential backoff on 429."""
    import random
    for attempt in range(1, max_attempts + 1):
        try:
            pytrends.build_payload(keywords, geo=geo, timeframe=timeframe)
            df = pytrends.interest_over_time()
            return df
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "Too Many Requests" in msg or "TooManyRequestsError" in type(e).__name__
            if attempt == max_attempts or not is_429:
                raise
            wait = (2 ** attempt) * 30 + random.uniform(0, 15)
            print(f"    429 — {geo} attempt {attempt}/{max_attempts}, sleeping {wait:.0f}s")
            time.sleep(wait)
    return None


def fetch_google_trends() -> pd.DataFrame:
    from pytrends.request import TrendReq
    # retries=0: pytrends' internal urllib3 retry uses the removed method_whitelist
    # kwarg (urllib3>=1.26 dropped it). We do our own retry in _trends_with_retry.
    pytrends = TrendReq(hl="en-US", tz=120, retries=0, timeout=(10, 30))
    timeframe = TRENDS_TIMEFRAME
    frames = []

    for geo, keywords in TRENDS_QUERIES.items():
        print(f"  Google Trends — {geo}")
        try:
            df = _trends_with_retry(pytrends, keywords, geo, timeframe)
            if df is None or df.empty:
                print(f"  UYARI — {geo} boş, atlanıyor")
                continue
            df = df.drop(columns=["isPartial"], errors="ignore")
            df.columns = [f"trends_{geo}_{c.replace(' ', '_')}" for c in df.columns]
            df = df.resample("MS").mean()
            frames.append(df)
            time.sleep(60)
        except Exception as e:
            print(f"  HATA — {geo}: {e}")
            time.sleep(60)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ─────────────────────────────────────────
# 4. TÜİK — Turist Girişi (Manuel Excel)
# ─────────────────────────────────────────

_TR_TO_ASCII = str.maketrans("ıİşŞğĞüÜöÖçÇâÂîÎûÛ", "iIsSgGuUoOcCaAiIuU")


def _norm_tr(s: str) -> str:
    """Fold a Turkish string to lowercase ASCII for keyword matching.

    Handles two Python quirks:
      1) "İ".lower() yields "i̇" (i + combining dot), not "i".
      2) "ı" is a single codepoint with no NFD decomposition.
    The translate table covers (1) and (2) along with ş/ğ/ü/ö/ç/â/î/û.
    """
    return unicodedata.normalize(
        "NFC", str(s).translate(_TR_TO_ASCII).lower()
    )


TUIK_MONTHS_TR = {
    _norm_tr(name): num for name, num in [
        ("ocak", 1), ("şubat", 2), ("mart", 3), ("nisan", 4),
        ("mayıs", 5), ("haziran", 6), ("temmuz", 7), ("ağustos", 8),
        ("eylül", 9), ("ekim", 10), ("kasım", 11), ("aralık", 12),
    ]
}


def _as_year(v):
    """Return v as a 4-digit year if it looks like one, else None.

    TÜİK stores year headers as floats (2010.0). Handle int, float, and str.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        if float(v).is_integer():
            n = int(v)
            return n if 2000 <= n <= 2030 else None
        return None
    s = str(v).strip()
    if s.isdigit() and 2000 <= int(s) <= 2030:
        return int(s)
    return None


def _to_number(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("\xa0", "").replace(" ", "").replace(".", "").replace(",", ".")
    if s in {"", "-", "..", "nan"}:
        return pd.NA
    try:
        return float(s)
    except ValueError:
        return pd.NA


def _parse_country_year_sheet(filepath: str, sheet: str) -> pd.DataFrame:
    """Sheet C-Mil Göre G.Yabancı: country × year matrix → long form.

    Layout: row 0 title, row 1 blank, row 2 = [_, "MİLLİYET", 2010, 2011, ...],
    data from row 3. Year cells are floats.
    """
    raw = pd.read_excel(filepath, sheet_name=sheet, header=None, engine="openpyxl")

    header_idx = None
    for i in range(min(10, len(raw))):
        years_in_row = [(c, _as_year(v)) for c, v in raw.iloc[i].items() if _as_year(v) is not None]
        if len(years_in_row) >= 3:
            header_idx = i
            year_cols = dict(years_in_row)
            break
    if header_idx is None:
        raise ValueError("Year header row not found")

    country_col = next(c for c in raw.columns if c not in year_cols and raw.iloc[header_idx + 1:][c].notna().any())

    body = raw.iloc[header_idx + 1:].reset_index(drop=True)
    out = pd.DataFrame({"country": body[country_col].astype(str).str.strip()})
    for col, year in year_cols.items():
        out[year] = body[col].map(_to_number)

    out = out[out["country"].str.len() > 0]
    out = out[~out["country"].map(_norm_tr).isin({"toplam", "total", "nan", ""})]
    long = out.melt(id_vars="country", var_name="year", value_name="arrivals")
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["arrivals"]).reset_index(drop=True)
    return long


def _parse_monthly_sheet(filepath: str, sheet: str) -> pd.DataFrame:
    """Sheet A-Yıl-Aya Göre G. Ziyaretçi: monthly totals.

    Sheet contains one-or-more stacked blocks. Each block has a year row
    (years like 2016, 2017, ...), a sub-header row (Yabancı/Vatandaş/Toplam),
    then 12 month-name rows. We extract the Toplam column per year, per block.
    """
    raw = pd.read_excel(filepath, sheet_name=sheet, header=None, engine="openpyxl")

    # Find every row that looks like a year header (≥ 3 year cells).
    year_rows = []
    for i in range(len(raw)):
        positions = [(c, _as_year(v)) for c, v in raw.iloc[i].items() if _as_year(v) is not None]
        if len(positions) >= 3:
            year_rows.append((i, positions))
    if not year_rows:
        raise ValueError("No year header row found in monthly sheet")

    records = []
    for block_idx, (year_row_idx, year_positions) in enumerate(year_rows):
        subheader = raw.iloc[year_row_idx + 1].astype(str).str.strip().map(_norm_tr)

        sorted_pos = sorted(year_positions, key=lambda t: t[0])
        bands = []
        for k, (start_col, year) in enumerate(sorted_pos):
            end_col = sorted_pos[k + 1][0] if k + 1 < len(sorted_pos) else max(raw.columns) + 1
            toplam_col = None
            for c in range(start_col, end_col):
                if c in subheader.index and subheader[c] == "toplam":
                    toplam_col = c
                    break
            if toplam_col is None:
                toplam_col = start_col
            bands.append((year, toplam_col))

        # Block body runs from year_row+2 until the next year row (or EOF).
        body_end = year_rows[block_idx + 1][0] if block_idx + 1 < len(year_rows) else len(raw)
        body = raw.iloc[year_row_idx + 2:body_end]

        month_col = None
        for c in body.columns:
            col_vals = body[c].astype(str).str.strip().map(_norm_tr)
            if col_vals.isin(TUIK_MONTHS_TR.keys()).sum() >= 6:
                month_col = c
                break
        if month_col is None:
            continue

        for _, row in body.iterrows():
            label = _norm_tr(str(row[month_col]).strip())
            if label not in TUIK_MONTHS_TR:
                continue
            month = TUIK_MONTHS_TR[label]
            for year, col in bands:
                val = _to_number(row[col])
                if pd.notna(val):
                    records.append((year, month, val))

    df = pd.DataFrame(records, columns=["year", "month", "arrivals"])
    df = df.drop_duplicates(subset=["year", "month"])
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df[["date", "arrivals"]].sort_values("date").reset_index(drop=True)


def load_tuik_excel(filepath: str) -> dict:
    """Parse TÜİK tourism Excel. Returns {} if file missing."""
    if not os.path.exists(filepath):
        print(f"  UYARI — TÜİK dosyası bulunamadı: {filepath}")
        return {}

    print(f"  TÜİK yükleniyor: {filepath}")
    out = {}
    try:
        out["country_year"] = _parse_country_year_sheet(filepath, "C-Mil Göre G.Yabancı")
        print(f"  country_year — {out['country_year'].shape}")
    except Exception as e:
        print(f"  HATA — country_year sheet: {e}")

    try:
        out["monthly"] = _parse_monthly_sheet(filepath, "A-Yıl-Aya Göre G. Ziyaretçi")
        print(f"  monthly — {out['monthly'].shape}")
    except Exception as e:
        print(f"  HATA — monthly sheet: {e}")

    return out


# ─────────────────────────────────────────
# Ana Çekim
# ─────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    print("\n[1/4] EVDS")
    try:
        evds_df = fetch_all_evds()
        evds_df.to_csv("data/raw/evds_raw.csv")
        print("  → data/raw/evds_raw.csv")
    except Exception as e:
        print(f"  BAŞARISIZ: {e}")

    print("\n[2/4] FRED")
    try:
        fred_df = fetch_all_fred()
        fred_df.to_csv("data/raw/fred_raw.csv")
        print("  → data/raw/fred_raw.csv")
    except Exception as e:
        print(f"  BAŞARISIZ: {e}")

    print("\n[3/4] Google Trends")
    try:
        trends_df = fetch_google_trends()
        if not trends_df.empty:
            trends_df.to_csv("data/raw/trends_raw.csv")
            print("  → data/raw/trends_raw.csv")
    except Exception as e:
        print(f"  BAŞARISIZ: {e}")

    print("\n[4/4] TÜİK")
    try:
        tuik = load_tuik_excel("data/raw/tuik_turizm.xlsx")
        for name, df in tuik.items():
            path = f"data/raw/tuik_{name}.csv"
            df.to_csv(path, index=False)
            print(f"  → {path} ({df.shape})")
    except Exception as e:
        print(f"  BAŞARISIZ: {e}")

    print("\nFaz 1 bitti.")
