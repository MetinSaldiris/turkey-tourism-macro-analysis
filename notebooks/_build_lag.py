"""Build notebooks/02_lag_analysis.ipynb programmatically and execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 2 — Lag analysis (CCF)\n"
    "\n"
    "For each of DE, GB, RU, we compute the cross-correlation function (CCF) "
    "between monthly arrivals and (a) the market's Google-Trends keyword and "
    "(b) the corresponding real exchange rate, over lags **-6 to +6 months**.\n"
    "\n"
    "_Sign convention used throughout:_ a **positive lag k** means the regressor "
    "leads arrivals by k months — that is, $\\rho(k) = \\text{corr}(x_{t-k}, y_t)$, "
    "so the regressor's value k months ago is correlated with arrivals today. A "
    "**negative lag** means the regressor lags arrivals.\n"
    "\n"
    "**Caveat — raw-series CCF.** Tourism arrivals are dominated by a strong "
    "annual seasonal cycle; Google-Trends search intent has the same seasonal "
    "shape. The CCF on the raw series will largely reflect this shared "
    "seasonality, which biases the peak-correlation lag toward 0. Treat the "
    "headline numbers as a first pass; a YoY-percent or seasonally-differenced "
    "CCF (12-month difference) is a sensible follow-up if a sharper lead/lag "
    "estimate is needed."
))

cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
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
    "\n"
    "MARKETS = {\n"
    "    'DE': {'trends': 'trends_DE_Türkei_Urlaub',  'real_fx': 'real_EUR_TRY', 'color': '#1f4e79'},\n"
    "    'GB': {'trends': 'trends_GB_Turkey_holiday', 'real_fx': 'real_GBP_TRY', 'color': '#2e7d32'},\n"
    "    'RU': {'trends': 'trends_RU_Турция_отдых',   'real_fx': 'real_RUB_TRY', 'color': '#7e57c2'},\n"
    "}\n"
    "MAX_LAG = 6"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Cross-correlation helper\n"
    "\n"
    "`ccf_two_sided` computes Pearson correlation between $x_{t-k}$ and $y_t$ "
    "for $k = -L, \\ldots, +L$. Inputs are aligned to their overlapping period "
    "and NaNs are dropped pairwise. The asymptotic 95% white-noise band at "
    "each lag is $\\pm 1.96 / \\sqrt{N_{\\text{eff}}(k)}$ where $N_{\\text{eff}}(k) = N - |k|$ "
    "is the number of usable observations at that lag."
))

cells.append(nbf.v4.new_code_cell(
    "def ccf_two_sided(x: pd.Series, y: pd.Series, max_lag: int = MAX_LAG):\n"
    "    \"\"\"Return (lags, ccf_values, conf_band, n_overlap).\n"
    "    \n"
    "    Positive lag k: x leads y by k months (corr of x_{t-k} with y_t).\n"
    "    Negative lag k: x lags y by |k| months.\n"
    "    \"\"\"\n"
    "    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()\n"
    "    n = len(pair)\n"
    "    xv = (pair['x'] - pair['x'].mean()) / pair['x'].std()\n"
    "    yv = (pair['y'] - pair['y'].mean()) / pair['y'].std()\n"
    "    \n"
    "    lags = np.arange(-max_lag, max_lag + 1)\n"
    "    vals = np.full(len(lags), np.nan)\n"
    "    for i, k in enumerate(lags):\n"
    "        if k >= 0:\n"
    "            a = xv.iloc[:n - k].values\n"
    "            b = yv.iloc[k:].values\n"
    "        else:\n"
    "            a = xv.iloc[-k:].values\n"
    "            b = yv.iloc[:n + k].values\n"
    "        if len(a) > 2:\n"
    "            vals[i] = np.corrcoef(a, b)[0, 1]\n"
    "    \n"
    "    conf = 1.96 / np.sqrt(np.maximum(n - np.abs(lags), 1))\n"
    "    return lags, vals, conf, n\n"
    "\n"
    "\n"
    "def plot_ccf(ax, lags, vals, conf, n, title, color):\n"
    "    ax.bar(lags, vals, color=color, alpha=0.85, edgecolor='white')\n"
    "    ax.plot(lags, conf, color='#cc4c4c', ls='--', lw=1, label='95% band')\n"
    "    ax.plot(lags, -conf, color='#cc4c4c', ls='--', lw=1)\n"
    "    ax.axhline(0, color='black', lw=0.5)\n"
    "    ax.axvline(0, color='gray', lw=0.5, ls=':')\n"
    "    ax.set_xlabel('Lag k (months)  — k>0: x leads arrivals')\n"
    "    ax.set_ylabel('Correlation')\n"
    "    ax.set_ylim(-1.05, 1.05)\n"
    "    ax.set_title(f'{title}  (N={n})')\n"
    "    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)"
))

# Per-market sections.
for mkt, spec in [
    ("DE", {"trends": "trends_DE_Türkei_Urlaub",  "real_fx": "real_EUR_TRY"}),
    ("GB", {"trends": "trends_GB_Turkey_holiday", "real_fx": "real_GBP_TRY"}),
    ("RU", {"trends": "trends_RU_Турция_отдых",   "real_fx": "real_RUB_TRY"}),
]:
    cells.append(nbf.v4.new_markdown_cell(
        f"## {mkt}\n"
        f"\n"
        f"CCF of `{spec['trends']}` and `{spec['real_fx']}` versus "
        f"`arrivals_total`."
    ))
    cells.append(nbf.v4.new_code_cell(
        f"mkt = '{mkt}'\n"
        f"spec = MARKETS[mkt]\n"
        f"\n"
        f"lags_t, vals_t, conf_t, n_t = ccf_two_sided(df[spec['trends']],  df['arrivals_total'])\n"
        f"lags_f, vals_f, conf_f, n_f = ccf_two_sided(df[spec['real_fx']], df['arrivals_total'])\n"
        f"\n"
        f"fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        f"plot_ccf(axes[0], lags_t, vals_t, conf_t, n_t,\n"
        f"         f'{{mkt}} — Trends vs. arrivals', spec['color'])\n"
        f"plot_ccf(axes[1], lags_f, vals_f, conf_f, n_f,\n"
        f"         f'{{mkt}} — real FX vs. arrivals', spec['color'])\n"
        f"fig.tight_layout()\n"
        f"fig.savefig(FIG_DIR / f'lag_{{mkt}}_ccf.png', dpi=150)\n"
        f"plt.show()"
    ))

# Summary table.
cells.append(nbf.v4.new_markdown_cell(
    "## Summary — lag of peak correlation\n"
    "\n"
    "For each (market, regressor) pair: the lag $k^*$ at which $|\\rho(k)|$ is "
    "maximised, the value $\\rho(k^*)$, the 95% band edge at $k^*$, and whether "
    "the peak is statistically distinct from white noise.\n"
    "\n"
    "Re-read the seasonality caveat at the top before drawing inferences from "
    "the lag column — peaks at $k = 0$ on the Trends rows are almost certainly "
    "the shared seasonal cycle, not a true contemporaneous causal link."
))

cells.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for mkt, spec in MARKETS.items():\n"
    "    for kind, col in [('Trends', spec['trends']), ('real FX', spec['real_fx'])]:\n"
    "        lags, vals, conf, n = ccf_two_sided(df[col], df['arrivals_total'])\n"
    "        absvals = np.abs(vals)\n"
    "        if np.all(np.isnan(absvals)):\n"
    "            continue\n"
    "        i = int(np.nanargmax(absvals))\n"
    "        rows.append({\n"
    "            'market':    mkt,\n"
    "            'regressor': kind,\n"
    "            'column':    col,\n"
    "            'n_overlap': n,\n"
    "            'peak_lag':  int(lags[i]),\n"
    "            'rho_peak':  round(float(vals[i]), 3),\n"
    "            'conf_95':   round(float(conf[i]), 3),\n"
    "            'significant': bool(abs(vals[i]) > conf[i]),\n"
    "        })\n"
    "summary = pd.DataFrame(rows)\n"
    "summary"
))

cells.append(nbf.v4.new_code_cell(
    "summary.to_csv(ROOT / 'outputs' / 'lag_summary.csv', index=False)\n"
    "print(f\"→ outputs/lag_summary.csv\")"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Next\n"
    "\n"
    "If the peak-lag values look mostly seasonality-driven (typical signature: "
    "every Trends row peaks at $k=0$ with $\\rho > 0.8$), the natural next step "
    "before modelling is to rebuild the CCF on **year-over-year percent change** "
    "or **seasonal first difference** ($\\Delta_{12}$). That removes the shared "
    "cycle and isolates the lead/lag in the *deviation* from each series' typical "
    "monthly path."
))

nb["cells"] = cells

out_path = Path(__file__).parent / "02_lag_analysis.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
