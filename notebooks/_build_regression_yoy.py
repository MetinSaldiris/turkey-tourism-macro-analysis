"""Build notebooks/04_regression_yoy.ipynb programmatically and execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 2 — YoY-log demand regression\n"
    "\n"
    "Notebook 03 ran OLS on first-differenced monthly data and produced an "
    "R² of 0.09 with no individually-significant regressors — the symptom of "
    "shared annual seasonality surviving $\\Delta_1$ differencing. Here we "
    "rebuild on **year-over-year log change**, $\\Delta_{12} \\log x = "
    "\\log x_t - \\log x_{t-12}$, which removes the calendar cycle and keeps "
    "the slow macro signal intact.\n"
    "\n"
    "Sequence:\n"
    "\n"
    "1. Build YoY-log series for arrivals and the three real-FX rates.\n"
    "2. ADF on the YoY series to confirm stationarity.\n"
    "3. OLS with Newey–West HAC (4 lags), Trends entering at lag 1.\n"
    "4. Re-fit with month-of-year fixed effects; compare $R^2$.\n"
    "5. Economic reading of the real-EUR/TRY elasticity."
))

cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import statsmodels.api as sm\n"
    "import statsmodels.formula.api as smf\n"
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

# Step 1 — YoY transform.
cells.append(nbf.v4.new_markdown_cell(
    "## 1. YoY log-change transform\n"
    "\n"
    "$yoy\\_x_t = \\log x_t - \\log x_{t-12}$, expressed as a decimal "
    "(0.10 ≈ +10% YoY). Applied to arrivals and to the three real-FX series."
))

cells.append(nbf.v4.new_code_cell(
    "def yoy_log(s: pd.Series) -> pd.Series:\n"
    "    return np.log(s) - np.log(s.shift(12))\n"
    "\n"
    "df['yoy_arrivals']     = yoy_log(df['arrivals_total'])\n"
    "df['yoy_real_EUR_TRY'] = yoy_log(df['real_EUR_TRY'])\n"
    "df['yoy_real_GBP_TRY'] = yoy_log(df['real_GBP_TRY'])\n"
    "df['yoy_real_USD_TRY'] = yoy_log(df['real_USD_TRY'])\n"
    "\n"
    "yoy_cols = ['yoy_arrivals', 'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY', 'yoy_real_USD_TRY']\n"
    "df[yoy_cols].describe().round(3)"
))

# Step 2 — ADF on YoY.
cells.append(nbf.v4.new_markdown_cell(
    "## 2. ADF on the YoY series\n"
    "\n"
    "Same ADF (constant, AIC-selected lag, 5% threshold) on `yoy_arrivals` "
    "and `yoy_real_EUR_TRY`."
))

cells.append(nbf.v4.new_code_cell(
    "def adf_row(name, s):\n"
    "    s = s.dropna()\n"
    "    stat, pval, used_lag, nobs, crit, _ = adfuller(s, regression='c', autolag='AIC')\n"
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
    "pd.DataFrame([\n"
    "    adf_row('yoy_arrivals',     df['yoy_arrivals']),\n"
    "    adf_row('yoy_real_EUR_TRY', df['yoy_real_EUR_TRY']),\n"
    "])"
))

# Step 3 — OLS.
cells.append(nbf.v4.new_markdown_cell(
    "## 3. OLS — base specification\n"
    "\n"
    "$$ yoy\\_arrivals_t = \\alpha\n"
    "  + \\beta_1\\, yoy\\_rEUR/TRY_t + \\beta_2\\, yoy\\_rGBP/TRY_t\n"
    "  + \\beta_3\\, Trends^{DE}_{t-1} + \\beta_4\\, Trends^{GB}_{t-1}\n"
    "  + \\gamma_1\\,covid + \\gamma_2\\,war + \\gamma_3\\,mideast + \\varepsilon_t $$\n"
    "\n"
    "$\\beta_1$ is an elasticity: a 1% real TRY depreciation against EUR is "
    "associated with $\\beta_1\\%$ change in YoY arrivals."
))

cells.append(nbf.v4.new_code_cell(
    "model_df = df.assign(\n"
    "    trends_DE_lag1 = df['trends_DE_Türkei_Urlaub'].shift(1),\n"
    "    trends_GB_lag1 = df['trends_GB_Turkey_holiday'].shift(1),\n"
    "    month          = df.index.month,\n"
    ")\n"
    "\n"
    "regressors = [\n"
    "    'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY',\n"
    "    'trends_DE_lag1', 'trends_GB_lag1',\n"
    "    'covid', 'russia_war', 'mideast',\n"
    "]\n"
    "y_col = 'yoy_arrivals'\n"
    "\n"
    "sample = model_df[[y_col, 'month'] + regressors].dropna()\n"
    "print(f'Sample: {sample.index.min().date()} → {sample.index.max().date()}  (N={len(sample)})')\n"
    "\n"
    "X_base = sm.add_constant(sample[regressors])\n"
    "y_base = sample[y_col]\n"
    "model_base = sm.OLS(y_base, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': 4})\n"
    "print(model_base.summary())\n"
    "print()\n"
    "print(f'Durbin–Watson: {durbin_watson(model_base.resid):.3f}')"
))

# Step 4 — month FE.
cells.append(nbf.v4.new_markdown_cell(
    "## 4. Add month fixed effects, compare R²\n"
    "\n"
    "Month-of-year dummies absorb any residual calendar effect not removed "
    "by the YoY transform (e.g., shifting Eid/Easter, school-break timing). "
    "The comparison cell prints $R^2$ and the real-EUR/TRY coefficient under "
    "each spec."
))

cells.append(nbf.v4.new_code_cell(
    "formula = (\n"
    "    'yoy_arrivals ~ yoy_real_EUR_TRY + yoy_real_GBP_TRY'\n"
    "    ' + trends_DE_lag1 + trends_GB_lag1'\n"
    "    ' + covid + russia_war + mideast'\n"
    "    ' + C(month)'\n"
    ")\n"
    "model_fe = smf.ols(formula, data=sample).fit(cov_type='HAC', cov_kwds={'maxlags': 4})\n"
    "print(model_fe.summary())\n"
    "print()\n"
    "print(f'Durbin–Watson: {durbin_watson(model_fe.resid):.3f}')"
))

cells.append(nbf.v4.new_code_cell(
    "comp = pd.DataFrame({\n"
    "    'no_FE': {\n"
    "        'R2':         round(model_base.rsquared, 3),\n"
    "        'Adj_R2':     round(model_base.rsquared_adj, 3),\n"
    "        'beta_EUR':   round(model_base.params['yoy_real_EUR_TRY'], 3),\n"
    "        'p_EUR':      round(model_base.pvalues['yoy_real_EUR_TRY'], 4),\n"
    "        'DW':         round(durbin_watson(model_base.resid), 3),\n"
    "    },\n"
    "    'month_FE': {\n"
    "        'R2':         round(model_fe.rsquared, 3),\n"
    "        'Adj_R2':     round(model_fe.rsquared_adj, 3),\n"
    "        'beta_EUR':   round(model_fe.params['yoy_real_EUR_TRY'], 3),\n"
    "        'p_EUR':      round(model_fe.pvalues['yoy_real_EUR_TRY'], 4),\n"
    "        'DW':         round(durbin_watson(model_fe.resid), 3),\n"
    "    },\n"
    "})\n"
    "comp"
))

# Step 5 — interpretation.
cells.append(nbf.v4.new_markdown_cell(
    "## 5. Economic interpretation — real-EUR/TRY elasticity\n"
    "\n"
    "Cell below picks the **preferred spec** (higher adjusted-$R^2$) and "
    "translates its $\\beta$ on `yoy_real_EUR_TRY` into the plain-English "
    "statement: *a 1% real TRY depreciation against EUR is associated with "
    "$\\beta$% change in YoY arrivals*."
))

cells.append(nbf.v4.new_code_cell(
    "preferred_name, preferred = max(\n"
    "    [('no-FE', model_base), ('month FE', model_fe)],\n"
    "    key=lambda kv: kv[1].rsquared_adj,\n"
    ")\n"
    "b   = preferred.params['yoy_real_EUR_TRY']\n"
    "p   = preferred.pvalues['yoy_real_EUR_TRY']\n"
    "ci  = preferred.conf_int().loc['yoy_real_EUR_TRY']\n"
    "alpha = 0.05\n"
    "\n"
    "print(f'Preferred spec: {preferred_name}  (adj R² = {preferred.rsquared_adj:.3f})')\n"
    "print(f'beta_EUR = {b:+.3f}   p = {p:.4f}   95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]')\n"
    "print()\n"
    "verdict = 'significant at 5%' if p < alpha else 'NOT significant at 5%'\n"
    "print(f'Result: real-EUR/TRY coefficient is {verdict}.')\n"
    "print()\n"
    "# In a log–log spec, beta is itself the elasticity:\n"
    "# yoy_arrivals and yoy_real_EUR/TRY are both log-decimals, so a +1pp change\n"
    "# in the regressor (i.e. Δ = 0.01) maps to a beta * 0.01 change in y, which\n"
    "# reads as beta% in the dependent variable. So we print beta directly.\n"
    "if p < alpha and b > 0:\n"
    "    print('Economic reading:')\n"
    "    print(f'  A 1% real TRY depreciation against EUR (yoy_real_EUR/TRY rises by 0.01)')\n"
    "    print(f'  is associated with a {b:+.2f}% change in YoY arrivals.')\n"
    "    print(f'  95% CI on that elasticity: [{ci[0]:+.2f}%, {ci[1]:+.2f}%].')\n"
    "    print('  Sign matches the textbook substitution channel: cheaper TRY → more EU arrivals.')\n"
    "elif p < alpha and b < 0:\n"
    "    print(f'Sign is negative — a 1% real TRY depreciation is associated with {b:+.2f}% YoY arrivals.')\n"
    "    print('  Counter to the textbook prior; likely picking up reverse-causal or omitted-variable dynamics.')\n"
    "else:\n"
    "    print(f'Point estimate: 1% real TRY depreciation ↔ {b:+.2f}% YoY arrivals.')\n"
    "    print(f'  But the 95% CI [{ci[0]:+.2f}%, {ci[1]:+.2f}%] straddles zero — the data')\n"
    "    print('  cannot rule out no FX effect at conventional levels. Interpret with caution.')\n"
    "\n"
    "print()\n"
    "print('Other regressors at 5%:')\n"
    "sig = preferred.pvalues[preferred.pvalues < alpha].drop(['yoy_real_EUR_TRY'], errors='ignore')\n"
    "if sig.empty:\n"
    "    print('  (no other regressor clears p < 0.05)')\n"
    "else:\n"
    "    for name in sig.index:\n"
    "        if name.startswith('C(month)'): continue\n"
    "        print(f'  {name:25s}  beta={preferred.params[name]:+.3e}  p={sig[name]:.4f}')"
))

nb["cells"] = cells

out_path = Path(__file__).parent / "04_regression_yoy.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
