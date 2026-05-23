# Turkey Tourism Demand — A Macro Analysis

This project tries to answer a simple question: what actually moves the number of foreign tourists who come to Turkey each month? Exchange rates? Google searches? Geopolitical shocks? I pulled monthly data from four sources back to 2010 and ran the analysis to find out.

The question matters because tourism is one of the largest sources of foreign currency for Turkey. Every percentage point of real depreciation against the euro is worth real money if it really does pull more visitors in. The headline result, with the caveats spelled out below, is that the effect is there but smaller and noisier than the textbook picture suggests.

## Data sources

- **EVDS** (Central Bank of Türkiye) — monthly TRY exchange rates against EUR, GBP, USD, RUB, and the Turkish CPI.
- **FRED** (St. Louis Fed) — monthly CPI for Germany, the UK, the US, and Russia, drawn from the OECD MEI series.
- **Google Trends** — monthly search-intent indices for Turkish-tourism keywords across DE, GB, SA, AE, and RU.
- **TÜİK** (Turkish Statistical Institute) — monthly arrivals totals and annual arrivals broken down by country of origin, parsed from the official Excel workbook.

## Project structure

```
turkey-tourism-macro-analysis/
├── src/
│   ├── data_fetch.py        # phase 1 — pull from all four sources
│   └── data_clean.py        # phase 2 — build the unified monthly master CSV
├── notebooks/
│   ├── 01_eda.ipynb              # plots: arrivals, FX, Trends, correlations
│   ├── 02_lag_analysis.ipynb     # cross-correlation functions by market
│   ├── 03_regression.ipynb       # OLS in first differences
│   ├── 04_regression_yoy.ipynb   # OLS in YoY log changes
│   └── 05_collinearity_diag.ipynb # why EUR and GBP can't both be in the model
├── data/
│   ├── raw/                 # pulled from APIs / TÜİK Excel (gitignored)
│   └── processed/
│       └── master_monthly.csv   # joined monthly panel, 2010-01 to 2025-10
├── outputs/
│   ├── figures/             # all notebook plots, saved as PNG
│   └── lag_summary.csv      # peak-correlation table from notebook 02
└── requirements.txt
```

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

3. Drop the TÜİK workbook at `data/raw/tuik_turizm.xlsx`. The file isn't redistributed here; you can download it from the TÜİK tourism statistics page.

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

## What I found

- **Arrivals follow a strong seasonal pattern.** July and August are roughly five to six times bigger than January and February. The shape is so dominant that the first thing any model has to do is take it out, or every result will just be a story about summer.

- **Google Trends leads arrivals by one month.** The cross-correlation peaks at lag +1 for both Germany (ρ = 0.51) and the UK (ρ = 0.69). When German or British search interest in Turkish holidays jumps, arrivals follow about a month later. That fits with how people actually book trips.

- **A 1% real TRY depreciation is associated with roughly +2% more arrivals year-on-year.** This is the elasticity I get from the YoY-log specification once the collinear EUR / GBP pair is replaced with a single regressor. The effect points in the textbook direction but it sits right at the edge of statistical significance (p ≈ 0.05–0.13). The signal is real, just not as loud as it would need to be to drive policy.

- **The Russia–Ukraine war pushed Russian arrivals structurally higher.** The CCF for the Russian search-intent keyword peaks at lag −4 with ρ = −0.47 — Russian searches for Turkish holidays fell at the same time Russian arrivals to Turkey *rose*. The most plausible reading is relocations and alternative travel routes after sanctions: arrivals stopped being driven by vacation search and started being driven by people moving.

- **EUR and GBP real exchange rates can't both be in the model.** Their YoY correlation is 0.864 and their VIFs in the joint regression are both above 12. When you put them in together, EUR comes out with the wrong sign and GBP soaks up all the elasticity. Once you isolate either one, both behave properly. This is a textbook collinearity story, and a useful reminder to look at condition numbers before reading coefficients.

## Limitations

- **Russian CPI ends in March 2022.** The OECD stopped publishing it after the sanctions, and FRED doesn't have a successor monthly series. Anything involving real RUB/TRY effectively stops there.

- **TÜİK monthly data only goes back to 2016.** The country-by-year sheet covers 2010–2025, but the monthly totals only start in 2016. That's the binding constraint on every monthly model and limits the post-COVID sample to about 100 observations.

- **Aggregate arrivals hide country-level variation.** I'm regressing total foreign arrivals on a small set of macros, which means I'm averaging over 240-odd source countries. The natural next step would be a country-level panel that uses the per-country FX and CPI for each origin.

- **The model still has dynamics it isn't capturing.** Durbin–Watson sits around 0.5 across every specification I tried. The Newey–West standard errors absorb that into the inference, but the residuals are still telling me a richer spec (an AR term, a lagged dependent, or distributed FX lags) would be worth fitting.
