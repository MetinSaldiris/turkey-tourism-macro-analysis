"""Build notebooks/01_eda.ipynb programmatically and execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 2 — Exploratory Data Analysis\n"
    "\n"
    "Five plots over `data/processed/master_monthly.csv` to ground the demand-modelling "
    "phase: tourist-arrival dynamics, real-vs-nominal FX, the FX/arrivals relationship, "
    "Google-Trends search demand vs. realised arrivals, and a correlation overview "
    "across macro + trends + arrivals. All figures are saved to `outputs/figures/`."
))

cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.dates as mdates\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path('..').resolve()\n"
    "FIG_DIR = ROOT / 'outputs' / 'figures'\n"
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "sns.set_theme(style='whitegrid', context='notebook')\n"
    "\n"
    "df = pd.read_csv(ROOT / 'data' / 'processed' / 'master_monthly.csv',\n"
    "                 index_col=0, parse_dates=True)\n"
    "df.index.name = 'date'\n"
    "print(df.shape)\n"
    "df.head(2)"
))

# Plot 1 — arrivals with shaded shocks
cells.append(nbf.v4.new_markdown_cell(
    "## 1. Monthly arrivals with shock periods\n"
    "\n"
    "Total monthly tourist arrivals (TÜİK, 2016–2025) with the three structural-shock "
    "flags overlaid as shaded bands: **COVID** (2020-03 → 2021-06), **Russia–Ukraine war** "
    "(2022-02 onwards), **Middle-East conflict** (2023-10 onwards). The plot answers two "
    "questions at a glance: how deep was the COVID trough, and did the geopolitical "
    "shocks visibly bend the post-COVID recovery."
))
cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(12, 5))\n"
    "arrivals = df['arrivals_total'].dropna() / 1e6\n"
    "ax.plot(arrivals.index, arrivals.values, color='#1f4e79', lw=1.8, label='Monthly arrivals (millions)')\n"
    "\n"
    "shocks = [\n"
    "    ('covid',      '#cc4c4c', 'COVID-19'),\n"
    "    ('russia_war', '#7e57c2', 'Russia–Ukraine war'),\n"
    "    ('mideast',    '#e0a32a', 'Middle-East conflict'),\n"
    "]\n"
    "for flag, color, label in shocks:\n"
    "    on = df.index[df[flag] == 1]\n"
    "    if len(on):\n"
    "        ax.axvspan(on.min(), on.max(), color=color, alpha=0.18, label=label)\n"
    "\n"
    "ax.set_xlim(arrivals.index.min(), arrivals.index.max())\n"
    "ax.set_xlabel('Date')\n"
    "ax.set_ylabel('Arrivals (millions / month)')\n"
    "ax.set_title('Turkey — Monthly foreign-visitor arrivals (2016–2025) with shock overlays')\n"
    "ax.legend(loc='upper left', framealpha=0.95)\n"
    "ax.xaxis.set_major_locator(mdates.YearLocator())\n"
    "ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "fig.tight_layout()\n"
    "fig.savefig(FIG_DIR / '01_arrivals_with_shocks.png', dpi=150)\n"
    "plt.show()"
))

# Plot 2 — nominal vs real EUR_TRY
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Nominal vs. real EUR/TRY (2010–2025)\n"
    "\n"
    "Two series on the same time axis: **nominal EUR/TRY** (left-axis units: TRY per EUR) "
    "and **real EUR/TRY** indexed to 2015 = 100 (right-axis, dimensionless). The gap between "
    "the two captures the cumulative inflation differential between Türkiye and Germany. "
    "When real EUR/TRY drifts above 100, TRY has become *cheaper for Eurozone tourists in "
    "purchasing-power terms* — the key intuition for the demand model."
))
cells.append(nbf.v4.new_code_cell(
    "fig, ax1 = plt.subplots(figsize=(12, 5))\n"
    "ax2 = ax1.twinx()\n"
    "\n"
    "nom = df['EUR_TRY'].dropna()\n"
    "real = df['real_EUR_TRY'].dropna()\n"
    "ax1.plot(nom.index, nom.values, color='#1f4e79', lw=1.7, label='Nominal EUR/TRY')\n"
    "ax2.plot(real.index, real.values, color='#cc4c4c', lw=1.7, ls='--', label='Real EUR/TRY (2015=100)')\n"
    "ax2.axhline(100, color='#cc4c4c', lw=0.7, ls=':', alpha=0.6)\n"
    "\n"
    "ax1.set_xlabel('Date')\n"
    "ax1.set_ylabel('Nominal EUR/TRY', color='#1f4e79')\n"
    "ax2.set_ylabel('Real EUR/TRY (2015=100)', color='#cc4c4c')\n"
    "ax1.tick_params(axis='y', labelcolor='#1f4e79')\n"
    "ax2.tick_params(axis='y', labelcolor='#cc4c4c')\n"
    "ax1.set_title('Nominal vs. real EUR/TRY')\n"
    "\n"
    "lines1, labels1 = ax1.get_legend_handles_labels()\n"
    "lines2, labels2 = ax2.get_legend_handles_labels()\n"
    "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)\n"
    "ax1.xaxis.set_major_locator(mdates.YearLocator(2))\n"
    "ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "fig.tight_layout()\n"
    "fig.savefig(FIG_DIR / '02_eur_try_nominal_vs_real.png', dpi=150)\n"
    "plt.show()"
))

# Plot 3 — scatter real_EUR_TRY vs arrivals, COVID highlighted
cells.append(nbf.v4.new_markdown_cell(
    "## 3. Real EUR/TRY vs. arrivals, 2016–2025 (COVID highlighted)\n"
    "\n"
    "Cross-sectional view of the same period as plot 1, plotting each month as one dot: "
    "real EUR/TRY on the x-axis (cheaper TRY → right), arrivals on the y-axis. COVID "
    "months are coloured red. The expected pattern is a positive slope on the non-COVID "
    "points (cheaper real TRY → more arrivals) with the COVID cluster sitting far below "
    "the curve — a temporary demand collapse unrelated to FX."
))
cells.append(nbf.v4.new_code_cell(
    "sub = df.loc['2016':'2025', ['real_EUR_TRY', 'arrivals_total', 'covid']].dropna(\n"
    "    subset=['real_EUR_TRY', 'arrivals_total']\n"
    ")\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "for flag_val, color, label in [(0, '#1f4e79', 'Non-COVID'), (1, '#cc4c4c', 'COVID')]:\n"
    "    m = sub['covid'] == flag_val\n"
    "    ax.scatter(sub.loc[m, 'real_EUR_TRY'], sub.loc[m, 'arrivals_total'] / 1e6,\n"
    "               color=color, alpha=0.75, s=42, edgecolor='white', label=label)\n"
    "\n"
    "ax.set_xlabel('Real EUR/TRY (2015=100, higher = TRY cheaper)')\n"
    "ax.set_ylabel('Arrivals (millions / month)')\n"
    "ax.set_title('Real EUR/TRY vs. monthly arrivals — 2016–2025')\n"
    "ax.legend(loc='upper left', framealpha=0.95)\n"
    "fig.tight_layout()\n"
    "fig.savefig(FIG_DIR / '03_real_eur_vs_arrivals_scatter.png', dpi=150)\n"
    "plt.show()"
))

# Plot 4 — Trends DE vs arrivals, dual axis
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Google-Trends (Germany) vs. realised arrivals\n"
    "\n"
    "Search-intent leading indicator: the Trends index for *Türkei Urlaub* (Germany) on "
    "the left axis, monthly Turkish arrivals on the right axis. If German tourists plan "
    "trips two-to-three months ahead, search peaks should sit slightly to the left of "
    "arrival peaks. Use this plot to eyeball both contemporaneous and lagged comovement "
    "ahead of formal lag analysis in the modelling phase."
))
cells.append(nbf.v4.new_code_cell(
    "sub = df.loc['2016':'2025', ['trends_DE_Türkei_Urlaub', 'arrivals_total']].dropna()\n"
    "fig, ax1 = plt.subplots(figsize=(12, 5))\n"
    "ax2 = ax1.twinx()\n"
    "\n"
    "ax1.plot(sub.index, sub['trends_DE_Türkei_Urlaub'],\n"
    "         color='#1f4e79', lw=1.6, label='Trends DE — \"Türkei Urlaub\"')\n"
    "ax2.plot(sub.index, sub['arrivals_total'] / 1e6,\n"
    "         color='#cc4c4c', lw=1.6, ls='--', label='Arrivals (millions)')\n"
    "\n"
    "ax1.set_xlabel('Date')\n"
    "ax1.set_ylabel('Google-Trends index (0–100)', color='#1f4e79')\n"
    "ax2.set_ylabel('Monthly arrivals (millions)', color='#cc4c4c')\n"
    "ax1.tick_params(axis='y', labelcolor='#1f4e79')\n"
    "ax2.tick_params(axis='y', labelcolor='#cc4c4c')\n"
    "ax1.set_title('Google-Trends (DE, \"Türkei Urlaub\") vs. Turkish arrivals — 2016–2025')\n"
    "\n"
    "lines1, labels1 = ax1.get_legend_handles_labels()\n"
    "lines2, labels2 = ax2.get_legend_handles_labels()\n"
    "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)\n"
    "ax1.xaxis.set_major_locator(mdates.YearLocator())\n"
    "ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
    "fig.tight_layout()\n"
    "fig.savefig(FIG_DIR / '04_trends_de_vs_arrivals.png', dpi=150)\n"
    "plt.show()"
))

# Plot 5 — correlation heatmap
cells.append(nbf.v4.new_markdown_cell(
    "## 5. Correlation overview\n"
    "\n"
    "Pearson correlations across arrivals, all four real-FX series, and the DE/GB Trends "
    "columns. Pairwise computation drops missing values per pair — important because the "
    "series have very different coverage windows (real_RUB_TRY ends 2022-03, arrivals "
    "starts 2016-01, real_EUR/GBP end 2025-03). The heatmap is the starting point for "
    "deciding which regressors belong in the demand model.\n"
    "\n"
    "_Note: no US Trends column exists in `master_monthly.csv` (the original fetch did "
    "not query the US geo). Add it to `data_fetch.py` if a US Trends regressor is wanted._"
))
cells.append(nbf.v4.new_code_cell(
    "cols = [\n"
    "    'arrivals_total',\n"
    "    'real_EUR_TRY', 'real_GBP_TRY', 'real_USD_TRY', 'real_RUB_TRY',\n"
    "    'trends_DE_Türkei_Urlaub', 'trends_DE_Antalya_Hotel',\n"
    "    'trends_GB_Turkey_holiday', 'trends_GB_Antalya_hotel',\n"
    "]\n"
    "corr = df[cols].corr(method='pearson')\n"
    "\n"
    "short = {\n"
    "    'arrivals_total':         'Arrivals',\n"
    "    'real_EUR_TRY':           'rEUR/TRY',\n"
    "    'real_GBP_TRY':           'rGBP/TRY',\n"
    "    'real_USD_TRY':           'rUSD/TRY',\n"
    "    'real_RUB_TRY':           'rRUB/TRY',\n"
    "    'trends_DE_Türkei_Urlaub':  'DE_TürkeiUrlaub',\n"
    "    'trends_DE_Antalya_Hotel':  'DE_AntalyaHotel',\n"
    "    'trends_GB_Turkey_holiday': 'GB_TurkeyHoliday',\n"
    "    'trends_GB_Antalya_hotel':  'GB_AntalyaHotel',\n"
    "}\n"
    "corr = corr.rename(index=short, columns=short)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(9, 7))\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,\n"
    "            center=0, square=True, cbar_kws={'shrink': 0.8}, ax=ax)\n"
    "ax.set_title('Pearson correlations — arrivals, real FX, search intent')\n"
    "fig.tight_layout()\n"
    "fig.savefig(FIG_DIR / '05_correlation_heatmap.png', dpi=150)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Next\n"
    "\n"
    "All five figures saved under `outputs/figures/`. From here, the demand-model phase "
    "can proceed: build per-source-country regressions (or a panel) with arrivals as the "
    "dependent variable and real FX + Trends + shock flags as regressors."
))

nb["cells"] = cells

out_path = Path(__file__).parent / "01_eda.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
