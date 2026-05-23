"""Build notebooks/05_collinearity_diag.ipynb programmatically and execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 2 — Collinearity diagnostic\n"
    "\n"
    "Notebook 04 produced a wrong-sign, insignificant coefficient on "
    "`yoy_real_EUR_TRY` (β = −5.16, p = 0.13) and a right-sign, significant "
    "coefficient on `yoy_real_GBP_TRY` (β = +6.95, p = 0.04). statsmodels "
    "flagged a **condition number ≈ 2,560**, which strongly suggests the "
    "two FX series are collinear and OLS is splitting the elasticity between "
    "them in a numerically unstable way.\n"
    "\n"
    "This notebook isolates the problem and re-fits three alternative specs:\n"
    "\n"
    "1. **GBP-only** — drop EUR.\n"
    "2. **EUR-only** — drop GBP.\n"
    "3. **Average real-FX index** — replace EUR+GBP with their mean.\n"
    "\n"
    "Comparing the coefficient on the FX regressor across the four specs "
    "(including the no-FE base from notebook 04) tells us whether the EUR/GBP "
    "elasticity is a stable economic signal or an artefact of which collinear "
    "regressor happens to be in the model."
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.api as sm\n"
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n"
    "from statsmodels.stats.stattools import durbin_watson\n"
    "from pathlib import Path\n"
    "\n"
    "ROOT = Path('..').resolve()\n"
    "df = pd.read_csv(ROOT / 'data' / 'processed' / 'master_monthly.csv',\n"
    "                 index_col=0, parse_dates=True)\n"
    "df.index.name = 'date'\n"
    "\n"
    "def yoy(s): return np.log(s) - np.log(s.shift(12))\n"
    "\n"
    "df['yoy_arrivals']     = yoy(df['arrivals_total'])\n"
    "df['yoy_real_EUR_TRY'] = yoy(df['real_EUR_TRY'])\n"
    "df['yoy_real_GBP_TRY'] = yoy(df['real_GBP_TRY'])\n"
    "df['yoy_real_USD_TRY'] = yoy(df['real_USD_TRY'])\n"
    "df['trends_DE_lag1']   = df['trends_DE_Türkei_Urlaub'].shift(1)\n"
    "df['trends_GB_lag1']   = df['trends_GB_Turkey_holiday'].shift(1)\n"
    "df['yoy_real_EUGB_avg'] = (df['yoy_real_EUR_TRY'] + df['yoy_real_GBP_TRY']) / 2"
))

# Correlations + VIF.
cells.append(nbf.v4.new_markdown_cell(
    "## 1. Pair correlations and VIF\n"
    "\n"
    "If EUR/GBP move together, their pairwise correlation will be near 1 and "
    "their VIFs in any joint regression will be large (rule of thumb: VIF > 10 "
    "is severe multicollinearity)."
))

cells.append(nbf.v4.new_code_cell(
    "fx_cols = ['yoy_real_EUR_TRY', 'yoy_real_GBP_TRY', 'yoy_real_USD_TRY']\n"
    "print('Pairwise correlations among YoY real-FX series:')\n"
    "print(df[fx_cols].corr().round(3).to_string())"
))

cells.append(nbf.v4.new_code_cell(
    "# VIF using the same regressor set as notebook 04's base spec.\n"
    "regressors_base = [\n"
    "    'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY',\n"
    "    'trends_DE_lag1', 'trends_GB_lag1',\n"
    "    'covid', 'russia_war', 'mideast',\n"
    "]\n"
    "sample = df[['yoy_arrivals'] + regressors_base].dropna()\n"
    "X = sm.add_constant(sample[regressors_base])\n"
    "vif = pd.DataFrame({\n"
    "    'regressor': X.columns,\n"
    "    'VIF':       [round(variance_inflation_factor(X.values, i), 2)\n"
    "                  for i in range(X.shape[1])],\n"
    "})\n"
    "vif"
))

# Spec comparison.
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Re-fit under three alternatives\n"
    "\n"
    "Each spec keeps `trends_DE_lag1`, `trends_GB_lag1`, `covid`, `russia_war`, "
    "`mideast` and swaps only the FX regressor(s). Newey–West HAC SEs with 4 "
    "lags throughout, same sample window per spec (constrained by FX coverage)."
))

cells.append(nbf.v4.new_code_cell(
    "def fit_spec(fx_cols_in_spec, label):\n"
    "    regs = list(fx_cols_in_spec) + [\n"
    "        'trends_DE_lag1', 'trends_GB_lag1',\n"
    "        'covid', 'russia_war', 'mideast',\n"
    "    ]\n"
    "    sub = df[['yoy_arrivals'] + regs].dropna()\n"
    "    X = sm.add_constant(sub[regs])\n"
    "    y = sub['yoy_arrivals']\n"
    "    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})\n"
    "    return label, sub, m\n"
    "\n"
    "specs = {\n"
    "    'base (EUR+GBP)': ['yoy_real_EUR_TRY', 'yoy_real_GBP_TRY'],\n"
    "    'GBP-only':       ['yoy_real_GBP_TRY'],\n"
    "    'EUR-only':       ['yoy_real_EUR_TRY'],\n"
    "    'avg(EUR,GBP)':   ['yoy_real_EUGB_avg'],\n"
    "}\n"
    "results = {label: fit_spec(cols, label) for label, cols in specs.items()}\n"
    "\n"
    "rows = []\n"
    "for label, (_, sub, m) in results.items():\n"
    "    row = {\n"
    "        'spec':       label,\n"
    "        'N':          int(m.nobs),\n"
    "        'R2':         round(m.rsquared, 3),\n"
    "        'AdjR2':      round(m.rsquared_adj, 3),\n"
    "        'CondNo':     round(m.condition_number, 0),\n"
    "        'DW':         round(durbin_watson(m.resid), 3),\n"
    "    }\n"
    "    for fx_col in ['yoy_real_EUR_TRY', 'yoy_real_GBP_TRY', 'yoy_real_EUGB_avg']:\n"
    "        if fx_col in m.params.index:\n"
    "            row[f'b_{fx_col}'] = round(float(m.params[fx_col]), 2)\n"
    "            row[f'p_{fx_col}'] = round(float(m.pvalues[fx_col]), 3)\n"
    "        else:\n"
    "            row[f'b_{fx_col}'] = np.nan\n"
    "            row[f'p_{fx_col}'] = np.nan\n"
    "    rows.append(row)\n"
    "comparison = pd.DataFrame(rows).set_index('spec')\n"
    "comparison"
))

# Verbose summaries for the two cleaner specs.
cells.append(nbf.v4.new_markdown_cell(
    "## 3. Full OLS summaries — single-FX and average-FX specs\n"
    "\n"
    "The summaries below are the two specs that drop the collinear pair: GBP-only "
    "and EUR-only. Compare each coefficient's sign and 95% interval against the "
    "joint base spec to see which side the elasticity \"belongs\" to."
))

cells.append(nbf.v4.new_code_cell(
    "for label in ['GBP-only', 'EUR-only', 'avg(EUR,GBP)']:\n"
    "    _, sub, m = results[label]\n"
    "    print(f'=== {label}  N={int(m.nobs)}  R²={m.rsquared:.3f}  AdjR²={m.rsquared_adj:.3f} ===')\n"
    "    print(m.summary().tables[1])\n"
    "    print()"
))

# Diagnosis.
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Verdict\n"
    "\n"
    "Cell below picks the spec with the lowest condition number and prints a "
    "plain-English read on whether the EUR/GBP collinearity was masking a real "
    "elasticity or whether the FX channel is genuinely weak at this aggregation."
))

cells.append(nbf.v4.new_code_cell(
    "alpha = 0.05\n"
    "best = comparison.sort_values('CondNo').index[0]\n"
    "best_label, (_, _, best_model) = best, results[best]\n"
    "_, _, best_model = results[best_label]\n"
    "\n"
    "fx_in_best = [c for c in ['yoy_real_EUR_TRY', 'yoy_real_GBP_TRY', 'yoy_real_EUGB_avg']\n"
    "              if c in best_model.params.index]\n"
    "fx = fx_in_best[0]\n"
    "b, p = best_model.params[fx], best_model.pvalues[fx]\n"
    "ci_lo, ci_hi = best_model.conf_int().loc[fx]\n"
    "\n"
    "print(f'Lowest-condition spec: {best_label}  (condition number {best_model.condition_number:.0f})')\n"
    "print(f'FX regressor: {fx}')\n"
    "print(f'  beta = {b:+.3f}   p = {p:.4f}   95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]')\n"
    "print()\n"
    "if p < alpha and b > 0:\n"
    "    print('Reading: once the collinear EUR/GBP pair is replaced with a single')\n"
    "    print(f'regressor, the FX coefficient is {b:+.2f} and significant at 5%.')\n"
    "    print(f'A 1% real TRY depreciation is associated with a {b:+.2f}% change in YoY arrivals.')\n"
    "    print('The wrong-sign EUR result in notebook 04 was a collinearity artefact.')\n"
    "elif p < alpha and b < 0:\n"
    "    print(f'Single-FX spec also yields a negative significant FX coefficient ({b:+.2f}).')\n"
    "    print('Collinearity was not the cause — the elasticity itself is wrong-signed in this sample.')\n"
    "    print('Likely culprits: aggregate arrivals masks country-level heterogeneity, or omitted')\n"
    "    print('variables (Russian relocation flows, regional safety perception) dominate the macro signal.')\n"
    "else:\n"
    "    print(f'Single-FX spec yields beta = {b:+.2f}, p = {p:.3f} — not significant at 5%.')\n"
    "    print('Collinearity was contributing to instability, but the underlying FX signal is also')\n"
    "    print('weak. To pin the elasticity down, the natural next steps are: (a) per-source-country')\n"
    "    print('panel regression instead of aggregate arrivals, (b) richer dynamic spec (AR / DL terms),')\n"
    "    print('(c) longer / wider FX measure (e.g., REER basket against a broader peer set).')"
))

nb["cells"] = cells

out_path = Path(__file__).parent / "05_collinearity_diag.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
