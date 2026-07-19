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
    "1. Build YoY-log series for arrivals, the three real-FX rates, and the DE/GB Trends.\n"
    "2. ADF and KPSS on the YoY series to cross-check stationarity.\n"
    "3. OLS with Newey–West HAC (12 lags), Trends entering at YoY lag 1.\n"
    "4. Re-fit with month-of-year fixed effects; compare $R^2$.\n"
    "5. Economic reading of the real-EUR/TRY elasticity.\n"
    "6. Robustness: drop the COVID window (2020-03 → 2022-02) and re-fit."
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

# Defensive pulse-dummy derivation.
cells.append(nbf.v4.new_markdown_cell(
    "## 0. Defensive pulse-dummy derivation\n"
    "\n"
    "The four 12-month pulse dummies (`war_pulse`, `mideast_pulse`, "
    "`covid_crash_pulse`, `covid_rebound_pulse`) are produced by "
    "`src/data_clean.add_shock_flags`. Older versions of `master_monthly.csv` "
    "do not have them, so the cell below derives them from the index if the "
    "columns are missing. Logic mirrors `data_clean.py` exactly."
))

cells.append(nbf.v4.new_code_cell(
    "idx = df.index\n"
    "if 'war_pulse' not in df.columns:\n"
    "    df['war_pulse']            = ((idx >= '2022-02-01') & (idx <= '2023-01-01')).astype(int)\n"
    "    df['mideast_pulse']        = ((idx >= '2023-10-01') & (idx <= '2024-09-01')).astype(int)\n"
    "    df['covid_crash_pulse']    = ((idx >= '2020-03-01') & (idx <= '2021-02-01')).astype(int)\n"
    "    df['covid_rebound_pulse']  = ((idx >= '2021-03-01') & (idx <= '2022-02-01')).astype(int)\n"
    "    print('pulse dummies derived from index (old master_monthly.csv)')\n"
    "else:\n"
    "    print('pulse dummies loaded from master_monthly.csv')"
))

# Step 1 — YoY transform.
cells.append(nbf.v4.new_markdown_cell(
    "## 1. YoY log-change transform\n"
    "\n"
    "$yoy\\_x_t = \\log x_t - \\log x_{t-12}$, expressed as a decimal "
    "(0.10 ≈ +10% YoY). Applied to arrivals, the three real-FX series, and "
    "the DE/GB Trends indices."
))

cells.append(nbf.v4.new_code_cell(
    "def yoy_log(s: pd.Series) -> pd.Series:\n"
    "    return np.log(s) - np.log(s.shift(12))\n"
    "\n"
    "def yoy_log_safe(s: pd.Series) -> pd.Series:\n"
    "    # Trends indices can hit zero; log(0) is undefined -> drop those obs.\n"
    "    s = s.replace(0, np.nan)\n"
    "    return np.log(s) - np.log(s.shift(12))\n"
    "\n"
    "df['yoy_arrivals']     = yoy_log(df['arrivals_total'])\n"
    "df['yoy_real_EUR_TRY'] = yoy_log(df['real_EUR_TRY'])\n"
    "df['yoy_real_GBP_TRY'] = yoy_log(df['real_GBP_TRY'])\n"
    "df['yoy_real_USD_TRY'] = yoy_log(df['real_USD_TRY'])\n"
    "\n"
    "df['yoy_trends_DE'] = yoy_log_safe(df['trends_DE_Türkei_Urlaub'])\n"
    "df['yoy_trends_GB'] = yoy_log_safe(df['trends_GB_Turkey_holiday'])\n"
    "\n"
    "yoy_cols = ['yoy_arrivals', 'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY',\n"
    "            'yoy_real_USD_TRY', 'yoy_trends_DE', 'yoy_trends_GB']\n"
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
    "adf_tab = pd.DataFrame([\n"
    "    adf_row('yoy_arrivals',     df['yoy_arrivals']),\n"
    "    adf_row('yoy_real_EUR_TRY', df['yoy_real_EUR_TRY']),\n"
    "])\n"
    "adf_tab"
))

# Step 2b — KPSS cross-check.
cells.append(nbf.v4.new_markdown_cell(
    "## 2b. KPSS cross-check\n"
    "\n"
    "ADF's null is a unit root, KPSS's null is (level) stationarity — so the "
    "two tests reverse the burden of proof. When they agree, the verdict is "
    "clean. When ADF says non-stationary and KPSS says stationary, the "
    "reading is that ADF has low power on samples this short (~100 obs on "
    "the YoY-differenced sample). When they both say non-stationary, "
    "inference on levels-like regressors becomes fragile and must be flagged "
    "in Limitations."
))

cells.append(nbf.v4.new_code_cell(
    "from statsmodels.tsa.stattools import kpss\n"
    "import warnings\n"
    "def kpss_row(name, s):\n"
    "    s = s.dropna()\n"
    "    with warnings.catch_warnings():\n"
    "        warnings.simplefilter('ignore')  # kpss warns when p is outside table bounds\n"
    "        stat, pval, lags, crit = kpss(s, regression='c', nlags='auto')\n"
    "    return {'series': name, 'kpss_stat': round(float(stat), 3),\n"
    "            'p_value': round(float(pval), 3),\n"
    "            'conclusion': 'stationary' if pval > 0.05 else 'non-stationary'}\n"
    "\n"
    "kpss_tab = pd.DataFrame([\n"
    "    kpss_row('yoy_arrivals',     df['yoy_arrivals']),\n"
    "    kpss_row('yoy_real_EUR_TRY', df['yoy_real_EUR_TRY']),\n"
    "])\n"
    "kpss_tab"
))

cells.append(nbf.v4.new_markdown_cell(
    "**Interpretation.** The verdict cell below joins the ADF and KPSS "
    "columns. In this sample, KPSS fails to reject stationarity on both "
    "`yoy_arrivals` (KPSS stat 0.10, p ≈ 0.10) and `yoy_real_EUR_TRY` "
    "(KPSS stat 0.17, p ≈ 0.10). ADF rejects the unit root cleanly on "
    "`yoy_arrivals` (p = 0.005) but does not reject on `yoy_real_EUR_TRY` "
    "(p = 0.25). The two tests reverse the null, so the plain reading is: "
    "the YoY transform has removed the trend / unit-root component from both "
    "series, and ADF's non-rejection on the FX series is consistent with its "
    "well-known low power on small (~100 obs) samples. Both series are "
    "treated as stationary for the OLS below; the caveat is flagged in the "
    "README's Limitations bullet on stationarity."
))

cells.append(nbf.v4.new_code_cell(
    "verdict = adf_tab[['series','conclusion']].rename(columns={'conclusion':'adf'}).merge(\n"
    "    kpss_tab[['series','conclusion']].rename(columns={'conclusion':'kpss'}),\n"
    "    on='series')\n"
    "verdict"
))

# Step 3 — OLS.
cells.append(nbf.v4.new_markdown_cell(
    "## 3. OLS — base specification\n"
    "\n"
    "$$ yoy\\_arrivals_t = \\alpha\n"
    "  + \\beta_1\\, yoy\\_rEUR/TRY_t + \\beta_2\\, yoy\\_rGBP/TRY_t\n"
    "  + \\beta_3\\, yoy\\_Trends^{DE}_{t-1} + \\beta_4\\, yoy\\_Trends^{GB}_{t-1}\n"
    "  + \\gamma_1\\,covid\\_crash\\_pulse + \\gamma_2\\,covid\\_rebound\\_pulse\n"
    "  + \\gamma_3\\,war\\_pulse + \\gamma_4\\,mideast\\_pulse + \\varepsilon_t $$\n"
    "\n"
    "**Notes on this specification.**\n"
    "\n"
    "- Shock controls are **12-month pulse dummies**, not step dummies. A permanent "
    "  level shift shows up in a $\\Delta_{12}$-transformed dependent variable only "
    "  for the first 12 months (base-effect window), so YoY regressions need pulse, "
    "  not step, dummies to identify the shocks correctly.\n"
    "- Trends enter as **YoY log change at lag 1**, not the raw 0–100 index. "
    "  The dependent variable is a deseasonalised YoY change; a raw seasonal 0–100 "
    "  level regressor is on the wrong frequency and is mechanically biased toward a "
    "  near-zero coefficient. Placing both sides in YoY space removes that bias. "
    "  Zeros in the Trends index (log undefined) are dropped, which trims a small "
    "  number of observations from the estimation sample; the effective N is printed "
    "  below.\n"
    "- Trends enter at **lag 1** deliberately. The YoY CCF in notebook 02 peaks at "
    "  lag 0 for GB with ρ = +0.459 (and near-zero peak for DE, ρ = +0.622 at lag −6 "
    "  with a broad monotonic profile), so using lag 0 would risk same-month "
    "  simultaneity between search intent and arrivals in a single observation. "
    "  Lag +1 on the GB YoY CCF is still above the 95% white-noise band "
    "  (ρ = +0.404, band ±0.191), so the trade-off is a possibly attenuated "
    "  coefficient in exchange for a cleaner leading-indicator interpretation.\n"
    "- Standard errors are **Newey–West HAC with maxlags = 12**. The $\\Delta_{12}$ "
    "  transform on monthly data mechanically induces an MA(11) error structure "
    "  from overlapping differences, so the HAC bandwidth must be at least 11; we "
    "  use 12 to also absorb residual AR dynamics (DW is approximately 0.5 across "
    "  every spec).\n"
    "- $\\beta_1$ is an elasticity: a 1% real TRY depreciation against EUR is "
    "  associated with $\\beta_1\\%$ change in YoY arrivals."
))

cells.append(nbf.v4.new_code_cell(
    "model_df = df.assign(\n"
    "    trends_DE_lag1 = df['yoy_trends_DE'].shift(1),\n"
    "    trends_GB_lag1 = df['yoy_trends_GB'].shift(1),\n"
    "    month          = df.index.month,\n"
    ")\n"
    "\n"
    "regressors = [\n"
    "    'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY',\n"
    "    'trends_DE_lag1', 'trends_GB_lag1',\n"
    "    'covid_crash_pulse', 'covid_rebound_pulse', 'war_pulse', 'mideast_pulse',\n"
    "]\n"
    "y_col = 'yoy_arrivals'\n"
    "\n"
    "sample = model_df[[y_col, 'month'] + regressors].dropna()\n"
    "print(f'Sample: {sample.index.min().date()} → {sample.index.max().date()}  (N={len(sample)})')\n"
    "\n"
    "X_base = sm.add_constant(sample[regressors])\n"
    "y_base = sample[y_col]\n"
    "model_base = sm.OLS(y_base, X_base).fit(cov_type='HAC', cov_kwds={'maxlags': 12})\n"
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
    "    ' + covid_crash_pulse + covid_rebound_pulse + war_pulse + mideast_pulse'\n"
    "    ' + C(month)'\n"
    ")\n"
    "model_fe = smf.ols(formula, data=sample).fit(cov_type='HAC', cov_kwds={'maxlags': 12})\n"
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
    "print('Trends coefficients (YoY log at lag 1):')\n"
    "for trend_name in ['trends_DE_lag1', 'trends_GB_lag1']:\n"
    "    if trend_name in preferred.params.index:\n"
    "        bt = preferred.params[trend_name]\n"
    "        pt = preferred.pvalues[trend_name]\n"
    "        cit = preferred.conf_int().loc[trend_name]\n"
    "        sig = 'sig' if pt < alpha else 'ns'\n"
    "        print(f'  {trend_name:18s}  beta={bt:+.3f}  p={pt:.4f}  95% CI=[{cit[0]:+.3f}, {cit[1]:+.3f}]  ({sig})')\n"
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

# Step 6 — ex-COVID robustness.
cells.append(nbf.v4.new_markdown_cell(
    "## 6. Robustness: excluding the COVID window\n"
    "\n"
    "The YoY arrivals series shows a very large positive swing over 2020–2022 as "
    "the pandemic collapse rolls off the 12-month base. The two COVID pulse dummies "
    "control for it in the full sample, but a second look — dropping the whole "
    "window (2020-03 → 2022-02) and re-fitting — asks whether the elasticity is "
    "*driven by* those months or *survives* them.\n"
    "\n"
    "In this subsample the COVID pulses are all zero, so we drop them from the "
    "regressor list (otherwise the design matrix is singular). `war_pulse` and "
    "`mideast_pulse` are kept — both fall partly outside the excluded window."
))

cells.append(nbf.v4.new_code_cell(
    "covid_mask = (sample.index >= '2020-03-01') & (sample.index <= '2022-02-01')\n"
    "sample_exc = sample.loc[~covid_mask].copy()\n"
    "print(f'ex-COVID sample: {sample_exc.index.min().date()} → {sample_exc.index.max().date()}  (N={len(sample_exc)})')\n"
    "\n"
    "regressors_exc = [\n"
    "    'yoy_real_EUR_TRY', 'yoy_real_GBP_TRY',\n"
    "    'trends_DE_lag1', 'trends_GB_lag1',\n"
    "    'war_pulse', 'mideast_pulse',\n"
    "]\n"
    "X_exc = sm.add_constant(sample_exc[regressors_exc])\n"
    "y_exc = sample_exc[y_col]\n"
    "model_exc = sm.OLS(y_exc, X_exc).fit(cov_type='HAC', cov_kwds={'maxlags': 12})\n"
    "print(model_exc.summary())\n"
    "print()\n"
    "print(f'Durbin–Watson: {durbin_watson(model_exc.resid):.3f}')"
))

cells.append(nbf.v4.new_code_cell(
    "def row(name, m, fx='yoy_real_EUR_TRY'):\n"
    "    ci = m.conf_int().loc[fx]\n"
    "    return {\n"
    "        'spec':      name,\n"
    "        'N':         int(m.nobs),\n"
    "        f'beta_{fx}': round(float(m.params[fx]), 3),\n"
    "        f'p_{fx}':    round(float(m.pvalues[fx]), 4),\n"
    "        'ci_lo':     round(float(ci[0]), 3),\n"
    "        'ci_hi':     round(float(ci[1]), 3),\n"
    "        'R2':        round(float(m.rsquared), 3),\n"
    "    }\n"
    "\n"
    "cmp_covid = pd.DataFrame([row('full', model_base), row('ex-COVID', model_exc)]).set_index('spec')\n"
    "cmp_covid"
))

cells.append(nbf.v4.new_markdown_cell(
    "**Reading the comparison.** The verdict cell below prints whether the "
    "real-EUR/TRY point estimate survives dropping the COVID window and "
    "whether its p-value stays below 5%. If both hold, the headline elasticity "
    "is not a pandemic artefact. If the point estimate collapses or the "
    "p-value blows up, the finding is COVID-window dependent and the "
    "README/article must report it as such."
))

cells.append(nbf.v4.new_code_cell(
    "b_full,  p_full  = model_base.params['yoy_real_EUR_TRY'], model_base.pvalues['yoy_real_EUR_TRY']\n"
    "b_exc,   p_exc   = model_exc.params['yoy_real_EUR_TRY'],  model_exc.pvalues['yoy_real_EUR_TRY']\n"
    "print(f'full     : beta={b_full:+.3f}  p={p_full:.4f}')\n"
    "print(f'ex-COVID : beta={b_exc:+.3f}  p={p_exc:.4f}')\n"
    "print()\n"
    "if p_exc < 0.05 and b_exc * b_full > 0:\n"
    "    print('Verdict: the EUR/TRY elasticity survives the ex-COVID cut with the same sign at 5%.')\n"
    "    print('The full-sample finding is not driven by pandemic swings.')\n"
    "elif b_exc * b_full > 0:\n"
    "    print('Verdict: same sign, but the ex-COVID p-value is above 5%.')\n"
    "    print('The full-sample significance is at least partly COVID-window dependent.')\n"
    "else:\n"
    "    print('Verdict: the sign of the EUR/TRY elasticity flips (or its magnitude collapses)')\n"
    "    print('when the COVID window is dropped. The full-sample headline is COVID-window dependent')\n"
    "    print('and must be reported as such.')"
))

nb["cells"] = cells

out_path = Path(__file__).parent / "04_regression_yoy.ipynb"
out_path.write_text(nbf.writes(nb))
print(f"wrote {out_path}")
