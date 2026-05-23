"""Build notebooks/03_regression.ipynb programmatically and execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 2 — Demand regression (first-differenced, Newey–West)\n"
    "\n"
    "End-to-end demand-model fitting on `master_monthly.csv`:\n"
    "\n"
    "1. ADF unit-root tests on arrivals and three real-FX series in **levels**.\n"
    "2. First-differencing and a second ADF round on the differenced series.\n"
    "3. OLS of $\\Delta \\text{arrivals}_t$ on $\\Delta \\text{real EUR/TRY}_t$, "
    "$\\Delta \\text{real GBP/TRY}_t$, lag-1 DE & GB Trends, and the three shock "
    "flags. Standard errors are **Newey–West (HAC, 4 lags)** to keep inference "
    "robust to monthly autocorrelation that survives differencing.\n"
    "4. Coefficient interpretation, with economic reading of the real-FX channel."
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.api as sm\n"
    "from statsmodels.tsa.stattools import adfuller\n"
    "from statsmodels.stats.stattools import durbin_watson\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path('..').resolve()\n"
    "df = pd.read_csv(ROOT / 'data' / 'processed' / 'master_monthly.csv',\n"
    "                 index_col=0, parse_dates=True)\n"
    "df.index.name = 'date'\n"
    "print(df.shape)"
))

# Step 1 — ADF in levels.
cells.append(nbf.v4.new_markdown_cell(
    "## 1. ADF — levels\n"
    "\n"
    "Augmented Dickey–Fuller with constant (no trend) on each level series. "
    "The null hypothesis is **unit root** (non-stationary); reject if $p < 0.05$. "
    "`maxlag` is chosen by AIC."
))

cells.append(nbf.v4.new_code_cell(
    "def adf_row(name: str, s: pd.Series) -> dict:\n"
    "    s = s.dropna()\n"
    "    stat, pval, used_lag, nobs, crit, _icbest = adfuller(\n"
    "        s, regression='c', autolag='AIC')\n"
    "    return {\n"
    "        'series':     name,\n"
    "        'n':          int(nobs),\n"
    "        'used_lag':   int(used_lag),\n"
    "        'adf_stat':   round(float(stat), 3),\n"
    "        'p_value':    round(float(pval), 4),\n"
    "        'crit_5pct':  round(float(crit['5%']), 3),\n"
    "        'conclusion': 'stationary' if pval < 0.05 else 'non-stationary',\n"
    "    }\n"
    "\n"
    "LEVEL_SERIES = ['arrivals_total', 'real_EUR_TRY', 'real_GBP_TRY', 'real_USD_TRY']\n"
    "adf_levels = pd.DataFrame([adf_row(n, df[n]) for n in LEVEL_SERIES])\n"
    "adf_levels"
))

# Step 2 — first-difference + re-test.
cells.append(nbf.v4.new_markdown_cell(
    "## 2. First-difference, re-test\n"
    "\n"
    "Series flagged as non-stationary in step 1 are first-differenced "
    "($\\Delta x_t = x_t - x_{t-1}$) and the ADF is re-run. The differenced "
    "series feed the regression in step 3."
))

cells.append(nbf.v4.new_code_cell(
    "non_stat = adf_levels.loc[adf_levels['conclusion'] == 'non-stationary', 'series'].tolist()\n"
    "print('Non-stationary in levels:', non_stat)\n"
    "\n"
    "for col in non_stat:\n"
    "    df[f'd_{col}'] = df[col].diff()\n"
    "\n"
    "adf_diff = pd.DataFrame([adf_row(f'd_{c}', df[f'd_{c}']) for c in non_stat])\n"
    "adf_diff"
))

# Step 3 — OLS with Newey-West.
cells.append(nbf.v4.new_markdown_cell(
    "## 3. OLS regression with Newey–West SE\n"
    "\n"
    "**Specification**\n"
    "\n"
    "$$\\Delta \\text{arrivals}_t = \\alpha\n"
    "  + \\beta_1 \\Delta \\text{rEUR/TRY}_t\n"
    "  + \\beta_2 \\Delta \\text{rGBP/TRY}_t\n"
    "  + \\beta_3 \\text{Trends}^{DE}_{t-1}\n"
    "  + \\beta_4 \\text{Trends}^{GB}_{t-1}\n"
    "  + \\gamma_1 \\text{covid}_t + \\gamma_2 \\text{war}_t + \\gamma_3 \\text{mideast}_t\n"
    "  + \\varepsilon_t$$\n"
    "\n"
    "Trends regressors enter at **lag 1** (so they reflect prior-month search "
    "intent, consistent with the CCF peaks). Standard errors are HAC with "
    "`maxlags=4` — captures up to a quarter of monthly autocorrelation."
))

cells.append(nbf.v4.new_code_cell(
    "model_df = df.assign(\n"
    "    trends_DE_lag1 = df['trends_DE_Türkei_Urlaub'].shift(1),\n"
    "    trends_GB_lag1 = df['trends_GB_Turkey_holiday'].shift(1),\n"
    ")\n"
    "\n"
    "regressors = [\n"
    "    'd_real_EUR_TRY', 'd_real_GBP_TRY',\n"
    "    'trends_DE_lag1', 'trends_GB_lag1',\n"
    "    'covid', 'russia_war', 'mideast',\n"
    "]\n"
    "y_col = 'd_arrivals_total'\n"
    "\n"
    "sample = model_df[[y_col] + regressors].dropna()\n"
    "y = sample[y_col]\n"
    "X = sm.add_constant(sample[regressors])\n"
    "print(f'Sample: {sample.index.min().date()} → {sample.index.max().date()}  (N={len(sample)})')\n"
    "\n"
    "model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})\n"
    "print(model.summary())\n"
    "print()\n"
    "print(f'Durbin–Watson on residuals: {durbin_watson(model.resid):.3f}')"
))

# Step 4 — interpretation.
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Interpretation\n"
    "\n"
    "The cell below derives the prose conclusions from the fitted model — sign / "
    "significance / economic reading are computed from `model.params` and "
    "`model.pvalues`, not hand-written, so the text stays accurate if the "
    "underlying CSV is refreshed."
))

cells.append(nbf.v4.new_code_cell(
    "alpha = 0.05\n"
    "params = model.params\n"
    "pvals = model.pvalues\n"
    "\n"
    "print('Significant regressors (p < 0.05):')\n"
    "sig = pvals[pvals < alpha]\n"
    "if sig.empty:\n"
    "    print('  (none)')\n"
    "else:\n"
    "    for name in sig.index:\n"
    "        print(f'  {name:18s}  beta={params[name]:+.3e}  p={pvals[name]:.4f}')\n"
    "\n"
    "print()\n"
    "print('Real EUR/TRY channel — economic reading:')\n"
    "b = params.get('d_real_EUR_TRY', np.nan)\n"
    "p = pvals.get('d_real_EUR_TRY', np.nan)\n"
    "if np.isnan(b):\n"
    "    print('  (regressor not in model)')\n"
    "elif p >= alpha:\n"
    "    print(f'  Coefficient {b:+.0f} is NOT significant at 5% (p={p:.3f}).')\n"
    "    print('  The data do not support a same-month real-FX → arrivals channel for the Eurozone in this spec.')\n"
    "elif b > 0:\n"
    "    print(f'  Coefficient {b:+.0f} (p={p:.3f}): a one-unit rise in real EUR/TRY')\n"
    "    print('  (TRY one index point cheaper vs. EUR in purchasing-power terms) is associated')\n"
    "    print(f'  with ~{b:+.0f} additional arrivals in the same month — the expected substitution channel.')\n"
    "else:\n"
    "    print(f'  Coefficient {b:+.0f} (p={p:.3f}) is significant but NEGATIVE — opposite of the textbook prior.')\n"
    "    print('  Likely interpretations: monthly differencing introduces noise that swamps the FX signal,')\n"
    "    print('  or the differenced regressor is picking up reverse-causal short-run dynamics.')\n"
    "\n"
    "print()\n"
    "print(f'R-squared: {model.rsquared:.3f}   Adj R²: {model.rsquared_adj:.3f}')\n"
    "dw = durbin_watson(model.resid)\n"
    "print(f'Durbin–Watson: {dw:.3f}  '\n"
    "      f'(≈2 = no autocorr; <1.5 = positive autocorr; >2.5 = negative autocorr)')\n"
    "if dw < 1.5:\n"
    "    print('  → residuals show positive autocorrelation; the Newey–West SE already accounts for this,')\n"
    "    print('    but a richer dynamic spec (AR term or lagged dependent) would tighten inference.')"
))

nb["cells"] = cells

out_path = Path(__file__).parent / "03_regression.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
