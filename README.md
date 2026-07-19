# Turkey Tourism Demand: A Macro Analysis

This project tries to answer a simple question: what actually moves the number of foreign tourists who come to Turkey each month? Exchange rates? Google searches? Geopolitical shocks? I pulled monthly data from four sources back to 2010 and ran the analysis to find out.

The question matters because tourism is one of the largest sources of foreign currency for Turkey. Every percentage point of real depreciation against the euro is worth real money if it really does pull more visitors in. The headline result, with the caveats spelled out below, is that once the specification is tightened (correct HAC bandwidth for YoY, deseasonalised Trends regressors, pulse dummies for the shocks), the FX effect is directionally positive but no longer statistically significant at conventional levels; the strongest surviving demand signal is a UK-Trends elasticity.

## Data sources

- **EVDS** (Central Bank of Türkiye): monthly TRY exchange rates against EUR, GBP, USD, RUB, and the Turkish CPI.
- **FRED** (St. Louis Fed): monthly CPI indices for Germany (Eurostat HICP), the UK, the US, and Russia.
- **Google Trends**: monthly search-intent indices for Turkish-tourism keywords across DE, GB, SA, AE, RU.
- **TÜİK** (Turkish Statistical Institute): monthly arrivals totals and annual arrivals broken down by country of origin, parsed from the official Excel workbook.

## Project structure

```
turkey-tourism-macro-analysis/
├── src/
│   ├── data_fetch.py        # phase 1, pull from all four sources
│   └── data_clean.py        # phase 2, build the unified monthly master CSV
├── notebooks/
│   ├── 01_eda.ipynb              # plots: arrivals, FX, Trends, correlations, country shares
│   ├── 02_lag_analysis.ipynb     # cross-correlation functions by market (raw and YoY, RU pre/post split)
│   ├── 03_regression.ipynb       # OLS in first differences
│   ├── 04_regression_yoy.ipynb   # OLS in YoY log changes, KPSS cross-check, ex-COVID robustness
│   └── 05_collinearity_diag.ipynb # why EUR and GBP cannot both be in the model
├── tests/
│   └── test_parsers.py      # unit tests for the TÜİK Excel parser helpers
├── data/
│   ├── raw/                 # pulled from APIs and TÜİK Excel (gitignored)
│   └── processed/
│       └── master_monthly.csv   # joined monthly panel, 2010-01 to 2025-10
├── outputs/
│   ├── figures/             # all notebook plots, saved as PNG
│   ├── lag_summary.csv      # peak-correlation table from notebook 02 (raw and YoY variants)
│   └── portfolio_article.md
├── LICENSE
└── requirements.txt
```

### Repository layout notes

The five notebooks under `notebooks/` are build artefacts, not source. Each one is generated programmatically by its matching `_build_*.py` script (via `nbformat`); the notebooks are then executed with `jupyter nbconvert --execute --inplace`. To change a notebook, edit the `_build_*.py` script and re-run it, then re-execute the notebook. Do not edit the `.ipynb` files directly, changes there are overwritten on the next build.

## How to run it

1. Install Python dependencies. A clean virtualenv works fine.

    ```bash
    pip install -r requirements.txt
    ```

2. Add your API keys to a `.env` file at the project root.

    ```bash
    cp .env.example .env
    # edit .env and fill in EVDS_API_KEY and FRED_API_KEY
    ```

3. Drop the TÜİK workbook at `data/raw/tuik_turizm.xlsx`. The file is not redistributed here; you can download it from the TÜİK tourism statistics page.

4. Pull the raw data. The Google Trends step takes about five minutes because of the rate-limit handling.

    ```bash
    python src/data_fetch.py
    ```

5. Build the unified monthly dataset.

    ```bash
    python src/data_clean.py
    ```

6. Run the notebooks in order. Each one regenerates its figures into `outputs/figures/`.

    ```bash
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_lag_analysis.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_regression.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_regression_yoy.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/05_collinearity_diag.ipynb
    ```

7. Run the parser tests.

    ```bash
    python -m pytest
    ```

## What I found

All numbers below come from executing `notebooks/04_regression_yoy.ipynb` and `notebooks/05_collinearity_diag.ipynb` on the committed `data/processed/master_monthly.csv`. YoY sample: 2017-01 to 2025-03, N = 99. Standard errors are Newey-West HAC with maxlags = 12 (the Delta-12 transform mechanically induces an MA(11) error structure from overlapping differences, so the HAC bandwidth must be at least 11).

- **Arrivals follow a strong seasonal pattern.** July and August are roughly five to six times bigger than January and February. The shape is so dominant that the first thing any model has to do is take it out, or every result will just be a story about summer.

- **The raw-CCF claim that Trends leads arrivals by one month is a seasonality artefact.** On the raw series, DE Trends peaks at lag +1 with rho = +0.509 and GB Trends at lag +1 with rho = +0.691 (N = 118). On the deseasonalised YoY-CCF, that lead structure does not survive: DE peaks at lag -6 with rho = +0.622 and GB peaks at lag 0 with rho = +0.459 (N = 106); the correlation at lag +1 on the YoY-DE panel is only +0.292. Once the shared summer cycle is removed, Trends does not cleanly lead arrivals by one month for DE or GB. Figure: `outputs/figures/lag_yoy_ccf.png`.

- **The one-percent-real-depreciation-buys-two-percent-more-arrivals claim does not survive tighter inference.** In the preferred YoY OLS (no month-FE, higher adjusted R2), the real-EUR/TRY coefficient is beta = +2.05 with p = 0.083 and a 95 percent confidence interval of [-0.27, +4.37]. The point estimate points in the textbook direction, but the interval straddles zero, so the effect is not statistically significant at 5 percent. Dropping the COVID window (2020-03 through 2022-02) flips the sign entirely: beta = -0.59 with p = 0.163 on N = 75. The full-sample headline is COVID-window dependent, not a robust structural elasticity.

- **UK YoY Trends is the strongest surviving demand signal.** In the same base spec, YoY-log GB search intent at lag 1 has beta = +0.848 with p = 0.022 and a 95 percent CI of [+0.125, +1.571]; it is the only individually significant regressor other than the two COVID pulses. The DE-Trends coefficient is borderline (beta = -0.461, p = 0.054); its sign flips positive (beta = +0.28, p = 0.004) once the COVID window is dropped, which strongly suggests the DE result is confounded by the pandemic. On the specification-picked collinearity diagnostic, the picture is the same: after replacing the collinear EUR/GBP pair with a single regressor, the FX coefficient is around +0.7 to +0.8 with p between 0.09 and 0.18 (EUR-only spec: beta = +0.831, p = 0.094; GBP-only: beta = +0.697, p = 0.184; average of EUR/GBP: beta = +0.787, p = 0.132). None clear 5 percent.

- **The Russia-Ukraine war shows up as a clean structural break in Russian tourism dynamics.** The raw RU CCF splits sharply at 2022-02: pre-war (N = 73) it peaks at lag +1 with rho = +0.721 (Russian search intent leads arrivals in the holiday regime), post-war (N = 45) it peaks at lag -5 with rho = -0.727 (arrivals rise while search falls). The sign reversal between sub-periods is the break itself. The most plausible reading is that post-February 2022 Russian flows to Turkey stopped being vacation traffic and started being relocations, decoupling from search intent. Figure: `outputs/figures/lag_RU_split_ccf.png`.

- **EUR and GBP real exchange rates cannot both be in the model.** Their YoY correlation is 0.864 and their VIFs in the joint regression are 14.97 (EUR) and 15.80 (GBP), well past the textbook threshold. In the joint spec EUR keeps the correct positive sign (beta = +2.05, p = 0.083) but GBP takes the wrong-sign hit (beta = -1.34, p = 0.291) as OLS splits the joint elasticity across the two collinear regressors. Isolating either currency lowers the condition number from around 45 to around 10 and returns a stably positive but insignificant FX coefficient (EUR-only beta = +0.83, GBP-only beta = +0.70), which is exactly the collinearity pattern: the wrong-sign was an artefact of the joint spec, not a real feature of either bilateral rate.

## Limitations

- **Russian CPI ends in March 2022.** OECD stopped publishing after sanctions and FRED has no successor monthly series. Anything involving real RUB/TRY effectively stops there.

- **UK CPI ends in March 2025.** The FRED-hosted OECD MEI series `GBRCPIALLMINMEI` was discontinued at 2025-03; FRED has no ONS-sourced monthly UK CPI **index** series that is currently updated (the Eurostat companion for the UK stops at 2020-11 because of Brexit). Rather than silently switching to a rate series that would break the real-FX construction, the pipeline keeps the OECD MEI series and accepts that `real_GBP_TRY` is truncated at 2025-03. Effective YoY regression sample ends 2025-03 (N = 99). The DE series was migrated to Eurostat HICP (`CP0000DEM086NEST`), which is currently updated and monthly.

- **TÜİK monthly arrivals only go back to 2016.** The country-by-year sheet covers 2010 to 2025, but the monthly totals start in 2016. That is the binding constraint on every monthly model and limits the post-COVID YoY sample to 99 observations.

- **Aggregate arrivals hide country-level variation.** I am regressing total foreign arrivals on a small set of macros, which averages over 240-odd source countries. The natural next step is a country-level panel that uses the per-country FX and CPI for each origin.

- **Residual autocorrelation and stationarity nuance.** Durbin-Watson sits around 0.9 to 1.2 across specifications; a richer dynamic spec (an AR term or a lagged dependent variable) would tighten inference. ADF and KPSS disagree on the YoY real-EUR/TRY series (ADF does not reject a unit root at p = 0.25, KPSS does not reject stationarity at p = 0.10). The two-test verdict, given ADF's known low power on samples around 100 observations, is that the series is stationary, but inference on the FX coefficient should be read as fragile.

- **Google Trends is not bit-reproducible.** Trends indices are re-scaled per API request against the queried window, so re-running `data_fetch.py` produces different index values every time. `data/processed/master_monthly.csv` is therefore not a deterministic output of the pipeline; the committed CSV is the analysis dataset of record and every number in this README and in `outputs/portfolio_article.md` is tied to that snapshot.
