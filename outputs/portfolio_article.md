# Turkey Tourism Demand: What Actually Moves the Needle

Tourism is one of Turkey's largest sources of foreign currency, and every conversation about it eventually circles back to the same intuition: "the lira is cheap, so more people will come." I wanted to test that intuition properly. Pulling 15 years of monthly data from four very different sources, I set out to answer one question: what really drives foreign arrivals to Turkey, the exchange rate, intent (Google searches), or something else entirely? The honest answer turned out to be less flattering than the intuition, and the path to it was a small masterclass in why real-world data work looks nothing like a Kaggle CSV.

[SCREENSHOT: Monthly foreign arrivals to Turkey 2016 to 2025, with COVID and Russia-Ukraine war annotated as shocks, `01_arrivals_with_shocks.png`]

## The data challenge

The analysis touches four sources, and each one fought back in its own way:

- **EVDS** (Central Bank of Türkiye) for TRY exchange rates against EUR, GBP, USD, RUB and the Turkish CPI. The EVDS Python client silently rewrites column names: every `.` in the series code becomes `_` on the way back, so my first rename map quietly produced empty columns until I figured out what was happening.
- **FRED** for monthly CPI indices for Germany (Eurostat HICP), the UK, the US, and Russia. Two operational wrinkles here. The OECD-MEI Russian series ends in March 2022 because OECD stopped publishing after sanctions, so anything involving real RUB/TRY is structurally capped at that date. The OECD-MEI Germany series was discontinued at 2025-03 too, so I migrated DE_TUFE to the Eurostat HICP index (`CP0000DEM086NEST`), which is monthly and currently updated. FRED has no ONS-sourced monthly UK-CPI index that is currently updated (the Eurostat UK companion stops at 2020-11 because of Brexit), so the UK series is kept on OECD MEI and the sample ends at 2025-03. That is documented rather than papered over.
- **Google Trends** for search-intent indices on Turkish-tourism keywords across DE, GB, SA, AE, and RU. Trends rate-limits hard, and pytrends's own retry logic uses a urllib3 kwarg that was removed in 1.26, so it crashes instead of backing off. I disabled its retries, wrote my own exponential backoff with jitter on 429s, and accepted that the fetch step takes about five minutes per run. There is a deeper problem though: Trends indices are re-scaled per API request against the queried window, so re-running the fetch produces different index values every time. The committed CSV is the analysis dataset of record; the pipeline is not bit-reproducible on the Trends side.
- **TÜİK** (Turkish Statistical Institute) ships monthly arrivals as a single Excel workbook with *stacked* blocks: one block per year span, each with its own sub-headers, year cells stored as floats (`2016.0`), and Turkish month labels. Parsing this needed a tiny Unicode rabbit hole: `"İ".lower()` in Python yields `i + combining dot above`, not `"i"`, and `"ı"` has no NFD decomposition, so any normal `unicodedata` pipeline silently mis-matches half the month names. I built a translation table that handles both quirks, then walked the sheet block by block to extract the "Toplam" column per year.

None of this is in any textbook. All of it is what the job actually is.

## What I found

**The one-month lead of Google searches over arrivals is a seasonality artefact.** On the raw series, German search intent peaks against arrivals at lag +1 with rho = +0.509 and British search intent at lag +1 with rho = +0.691 (N = 118, both statistically significant). That was the original headline, and it looked clean. But the raw series both carry the same summer cycle, and a raw CCF is largely picking that shared cycle up. When I repeat the analysis on 12-month log differences (both sides deseasonalised), the DE Trends CCF peaks at lag -6 with rho = +0.622 while GB peaks at lag 0 with rho = +0.459 (N = 106). The one-month lead does not survive. That is a lesson worth sitting with: a strong-looking correlation that lives entirely on shared seasonality is not information, and any dashboard built on it would be reading the calendar, not the market.

[SCREENSHOT: Raw-CCF for DE, showing peak at lag +1 with rho = +0.51, `lag_DE_ccf.png`]
[SCREENSHOT: YoY-CCF panels for DE, GB, RU showing the raw peak does not survive deseasonalisation, `lag_yoy_ccf.png`]

**Russia broke the model in an instructive way, and the break lines up exactly with the war.** Rather than trying to summarise the full sample, I split the RU CCF at February 2022. Pre-war (N = 73) it peaks at lag +1 with rho = +0.721: search intent leads arrivals by a month, exactly the pattern you would expect from a normal holiday-booking flow. Post-war (N = 45) it peaks at lag -5 with rho = -0.727: arrivals rise while search falls, and search now lags arrivals instead of leading. The sign flip between the two sub-periods is the break itself. The most plausible reading is that post-February 2022 Russian flows into Turkey stopped being vacation traffic and started being relocations: people moving rather than holidaying, often via Turkey as one of the few remaining open routes. The data did not tell me this story directly, but a clean, sharp sign reversal in a previously well-behaved series almost always means the underlying behaviour changed, not the statistics.

[SCREENSHOT: RU pre and post February 2022 split-CCF, showing the sign reversal at the war, `lag_RU_split_ccf.png`]

## The exchange rate story

EUR and GBP move together. Their YoY real rates against TRY correlate at 0.864, and in the joint regression their VIFs sit at 14.97 (EUR) and 15.80 (GBP), well past the textbook threshold for severe multicollinearity. When OLS sees two regressors that nearly co-vary, it has no stable way to split the joint elasticity between them, so tiny perturbations in the sample reshuffle the coefficients. In this respecified model (Newey-West HAC with maxlags = 12 because Delta-12 mechanically induces an MA(11) error structure from overlapping differences, Trends entered in YoY-log space rather than raw index space, twelve-month pulse dummies for the shocks instead of step dummies), EUR/TRY comes out with beta = +2.05, p = 0.083, 95 percent CI [-0.27, +4.37]. It is positive, but the interval straddles zero.

The fix for the collinearity, dropping one or collapsing both, is the boring one. When I isolated either currency, or when I used the average of the two real rates as a single regressor, the sign is stably positive and the condition number drops from around 45 on the joint spec to around 10. The point estimates land around +0.7 to +0.8 across all three single-FX specifications (EUR-only: beta = +0.831, p = 0.094; GBP-only: beta = +0.697, p = 0.184; avg(EUR,GBP): beta = +0.787, p = 0.132). None of them clear the 5 percent significance threshold, and the ex-COVID robustness check on the preferred single-FX spec (EUR-only) collapses the coefficient to essentially zero (beta = +0.054, p = 0.637 on N = 75). The elasticity is not a robust structural feature of the data; it is either not there at monthly aggregate frequency or it is being drowned out by variation that a simple static spec cannot pick up.

I want to be honest about how loud that signal is. In the earlier version of this analysis I reported roughly +2 percent arrivals per +1 percent real TRY depreciation. That number was fragile in three ways: it used a HAC bandwidth (maxlags = 4) that was too short for the MA(11) error structure of the YoY transform, it used raw seasonal Trends regressors that pulled the FX beta around, and it lumped the COVID collapse and rebound into a single step dummy that a YoY-differenced dependent variable cannot identify (a permanent level shift shows up in a YoY series only for the first twelve months). Fixing all three moves the FX story from "borderline significant with a nice sign" to "point estimate is directionally right, but the confidence interval clears zero and the sign flips when I drop the pandemic window." That is a less flattering result. It is also the honest one.

[SCREENSHOT: Real vs. nominal EUR/TRY over time, showing the divergence after 2021, `02_eur_try_nominal_vs_real.png`]
[SCREENSHOT: Real EUR/TRY vs. arrivals scatter, with COVID highlighted, `03_real_eur_vs_arrivals_scatter.png`]

## The surviving signal: UK YoY search intent

The one regressor that clears 5 percent in the preferred YoY specification is not the exchange rate at all. It is the year-on-year log change in UK Google searches for "Turkey holiday" at lag 1: beta = +0.848, p = 0.022, 95 percent CI [+0.125, +1.571]. A +10 percent YoY move in UK search intent this month is associated with about +8 percent higher YoY arrivals next month, deseasonalised. This is a real signal precisely because the YoY-differenced dependent variable strips out the summer cycle: what remains is deviation from the typical monthly path, and deviation-in-search predicts deviation-in-arrivals for the UK market at a one-month lag. That is the closest thing to an actionable operator-level result the data will support. The German Trends coefficient at the same lag has a wrong sign in the full sample (beta = -0.461, p = 0.054), but flips positive and significant (beta = +0.28, p = 0.004) when the COVID window is dropped, which is very likely telling me that the DE full-sample sign is a pandemic confounder rather than a real reversal.

[SCREENSHOT: German search interest for "Türkei Urlaub" vs. monthly arrivals, `04_trends_de_vs_arrivals.png`]
[SCREENSHOT: Correlation heatmap of arrivals, real FX rates, and Trends series, `05_correlation_heatmap.png`]

## What this means in practice

**For a tourism operator,** the most actionable result is UK-Trends YoY at a one-month lead. It is a free, forward-looking, deseasonalised leading indicator you can put on a dashboard tomorrow: when the YoY change in UK "Turkey holiday" searches moves in month t, expect arrivals YoY to move in the same direction in month t+1 with an elasticity around 0.85. The old "raw-CCF peak at lag +1" claim was seasonality; this one is not.

**For a policymaker,** the exchange-rate elasticity is a lever the data does not actually support pulling. A point estimate of around +0.8 percent arrivals per +1 percent real depreciation would already have been small next to what a weak real exchange rate does to the rest of an import-dependent economy. A confidence interval that includes zero, and a sign that flips when the COVID window is dropped, means the aggregate monthly evidence is not strong enough to design tourism-tuned FX policy around.

**For an investor,** the Russia inversion is still the most interesting datapoint. When Russian search interest for Turkish holidays and Russian arrivals to Turkey decoupled at February 2022, searches falling while arrivals rose, that was not noise, it was a regime change. Tourism flows from sanctioned economies stopped reflecting vacation demand and started reflecting capital and people in motion. That has direct read-through to Turkish real estate, retail banking in the southern coast, and any business whose revenue mix depends on which kind of Russian shows up. The pre-2022 rho of +0.72 at lag +1 and the post-2022 rho of -0.73 at lag -5 are as clean a structural break as time-series data ever hand you.

## Limitations

I would rather be the one to name these than let a reader find them.

- **Aggregate arrivals mask country-level heterogeneity.** I am regressing total foreign arrivals on a small set of macros, which averages behaviour across 240-odd source countries. A German visitor and a Saudi visitor respond to very different things; collapsing them into one series throws away most of that signal.
- **The YoY transform costs me a year.** After Delta-12 differencing, the usable sample is 99 monthly observations (2017-01 through 2025-03). That is enough to detect a large effect and not enough to slice a small effect cleanly by sub-period; the ex-COVID cut takes it down to 75.
- **Residual autocorrelation is real.** Durbin-Watson sits between 0.9 and 1.2 across specifications; I am using Newey-West HAC standard errors with maxlags = 12 (the minimum bandwidth an overlapping 12-month difference needs, since it induces an MA(11) error structure) so the inference is honest, but a static OLS is leaving dynamic structure on the table.
- **Stationarity is not a clean call for the FX YoY series.** ADF and KPSS disagree on `yoy_real_EUR_TRY`: ADF does not reject a unit root at p = 0.25, KPSS does not reject stationarity at p = 0.10. The two-test verdict, taking into account ADF's known low power on samples of about 100 observations, is that the series is stationary, but the FX-coefficient inference should be read as fragile.
- **Russian CPI ends March 2022.** OECD stopped publishing after sanctions; FRED has no successor monthly series. Anything involving real RUB/TRY structurally ends there.
- **UK CPI ends March 2025.** The OECD MEI series for GBR was discontinued and FRED has no currently-updated ONS-sourced monthly UK-CPI *index* series that I could substitute in without breaking the real-FX construction. The Germany series was migrated to Eurostat HICP, which is monthly and currently updated.
- **Google Trends is not bit-reproducible.** Trends indices are re-scaled per request, so re-fetching the same query window produces different index values. The committed `master_monthly.csv` is the analysis dataset of record.

## What I would do next

Three concrete moves, in order of payoff.

**Country-level panel.** TÜİK gives me arrivals broken down by country of origin in `tuik_country_year.csv`. Going from a single time series to a 25-country panel changes the question from "what moves total arrivals?" to "which countries respond to which shocks?", and gives me roughly an order of magnitude more variation to fit on. The Russia inversion alone would be far easier to identify cleanly with a panel, and the FX channel would get a shot at showing up market by market where the aggregate washes it out.

**A proper REER basket.** Instead of regressing on individual real bilateral rates and then arguing about which one to keep, build a tourism-weighted real effective exchange rate using each origin country's share of arrivals as the weight. That sidesteps the EUR/GBP collinearity problem by construction rather than by dropping a regressor.

**A dynamic specification.** AR(1) on the dependent variable, distributed lags on FX, and re-run the diagnostics. A Durbin-Watson between 0.9 and 1.2 is telling me the static OLS is leaving information on the table. I would start with an ARDL and let the lag structure earn its place.

## Technical stack

Python end-to-end: `pandas` for the panel build, `statsmodels` for the OLS, Newey-West HAC, ADF, KPSS, and VIF diagnostics, `evds` / `fredapi` / `pytrends` for the API pulls, `openpyxl` for the TÜİK Excel parse, and `matplotlib` + `seaborn` for the figures. Everything orchestrated through Jupyter, with the notebooks programmatically generated from `.py` builders so the analysis is reproducible end-to-end: `nbconvert --execute` runs the whole pipeline from a clean checkout, and `python -m pytest` runs the parser unit tests.
