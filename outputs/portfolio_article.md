# Turkey Tourism Demand: What Actually Moves the Needle

Tourism is one of Turkey's largest sources of foreign currency, and every conversation about it eventually circles back to the same intuition: "the lira is cheap, so more people will come." I wanted to test that intuition properly. Pulling 15 years of monthly data from four very different sources, I set out to answer one question — what really drives foreign arrivals to Turkey: the exchange rate, intent (Google searches), or something else entirely? The headline answer turned out to be more interesting than I expected, and the path to it was a small masterclass in why real-world data work looks nothing like a Kaggle CSV.

[SCREENSHOT: Monthly foreign arrivals to Turkey 2016–2025, with COVID and Russia–Ukraine war annotated as shocks — `01_arrivals_with_shocks.png`]

## The data challenge

The analysis touches four sources, and each one fought back in its own way:

- **EVDS** (Central Bank of Türkiye) for TRY exchange rates against EUR, GBP, USD, RUB and the Turkish CPI. The EVDS Python client silently rewrites column names — every `.` in the series code becomes `_` on the way back, so my naïve rename map quietly produced empty columns until I figured out what was happening.
- **FRED** for OECD-sourced monthly CPI for Germany, the UK, the US, and Russia. The Russian series ends in March 2022 because the OECD stopped publishing after sanctions. There is no clean successor at monthly frequency, so anything involving *real* RUB/TRY is structurally capped at that date. I treated that as a constraint rather than something to paper over.
- **Google Trends** for search-intent indices on Turkish-tourism keywords across DE, GB, SA, AE, and RU. Trends rate-limits hard — and pytrends's own retry logic uses a urllib3 kwarg that was removed in 1.26, so it crashes instead of backing off. I disabled its retries, wrote my own exponential backoff with jitter on 429s, and accepted that the fetch step takes about five minutes per run.
- **TÜİK** (Turkish Statistical Institute) ships monthly arrivals as a single Excel workbook with *stacked* blocks — one block per year span, each with its own sub-headers, year cells stored as floats (`2016.0`), and Turkish month labels. Parsing this needed a tiny Unicode rabbit hole: `"İ".lower()` in Python yields `i + combining dot above`, not `"i"`, and `"ı"` has no NFD decomposition, so any normal `unicodedata` pipeline silently mis-matches half the month names. I built a translation table that handles both quirks, then walked the sheet block by block to extract the "Toplam" column per year.

None of this is in any textbook. All of it is what the job actually is.

## What I found

**Google searches lead arrivals by about one month.** The cross-correlation function for German search interest in Turkish holidays peaks at lag +1 with ρ = 0.51; for the UK it peaks at lag +1 with ρ = 0.69. In plain terms: when search interest jumps in May, arrivals jump in June. That fits how real people book holidays — a few weeks of looking around, then the booking, then the trip. For a tourism operator, this is the most actionable result in the project. Trends is free, fast, and forward-looking.

[SCREENSHOT: Cross-correlation function for Germany — search-intent series vs. arrivals, peak at lag +1, ρ = 0.51 — `lag_DE_ccf.png`]
[SCREENSHOT: Same for UK, peak at lag +1, ρ = 0.69 — `lag_GB_ccf.png`]

**Russia broke the model in an instructive way.** The CCF for Russian search interest peaks at lag −4 with ρ = −0.47 — Russian search interest in Turkish holidays *fell* at exactly the time Russian arrivals *rose*. The vacation-driven relationship inverted. The most plausible reading is that post-February 2022, Russian flows into Turkey stopped being vacation traffic and started being relocations: people moving rather than holidaying, often via Turkey as one of the few remaining open routes. The data didn't tell me this story directly — but a clean inversion of sign in a previously well-behaved series almost always means the underlying behaviour changed, not the statistics.

[SCREENSHOT: Cross-correlation function for Russia, peak at lag −4, ρ = −0.47, with the war marked on the x-axis — `lag_RU_ccf.png`]

## The exchange rate story

EUR and GBP move together. Their YoY real rates against TRY correlate at 0.86, and in a joint regression their VIFs both sit above 12 — well past the textbook threshold for severe multicollinearity. When OLS sees two regressors that nearly co-vary, it has no stable way to split the joint elasticity between them: tiny perturbations in the sample reshuffle the coefficients. In my case, that produced the classic symptom — EUR/TRY came out **negative and insignificant** (β ≈ −5.16, p ≈ 0.13), GBP/TRY came out **positive and significant** (β ≈ +6.95, p ≈ 0.04), and the condition number on the design matrix exploded to ~2,560. The model wasn't telling me a story about Germany versus Britain. It was telling me it couldn't tell those two stories apart.

The fix is the boring one: drop one, or collapse them. When I isolated either currency on its own — or when I used the average of the two real rates as a single regressor — the sign flipped back to positive and the coefficient stabilized. **The real elasticity of arrivals to a real TRY depreciation lands at roughly +2% per +1%.** In English: a 1% real-terms cheapening of the lira against the European currencies is associated with about 2% more YoY foreign arrivals.

I want to be honest about how loud that signal is. The point estimate is in the textbook direction and economically meaningful, but the p-values across the surviving specifications sit between 0.05 and 0.13. That's "I believe the effect is real" territory, not "this is bankable." At monthly aggregate level, with the post-COVID sample we have, the FX channel exists — but it isn't overwhelming, and anyone who tells you they can predict next quarter's arrivals from the lira alone is selling you something.

[SCREENSHOT: Real vs. nominal EUR/TRY over time, showing the divergence after 2021 — `02_eur_try_nominal_vs_real.png`]
[SCREENSHOT: Real EUR/TRY vs. YoY arrivals scatter, with the fitted line — `03_real_eur_vs_arrivals_scatter.png`]

## What this means in practice

**For a tourism operator,** the most actionable result isn't the exchange rate — it's Trends. German and British search intent leads arrivals by about a month with correlations of 0.51 and 0.69 respectively. That's a free, daily-refreshed leading indicator you can put on a dashboard tomorrow. If search interest for "Türkei Urlaub" jumps in May, your June occupancy forecast should move *now*, not after the booking data confirms it.

**For a policymaker,** the exchange-rate elasticity matters but it isn't a lever to pull casually. A 1% real depreciation buying ~2% more arrivals sounds attractive, until you remember every other thing a weak real exchange rate does to an import-dependent economy. Tourism revenue is one line; food and energy import bills are another. The fact that p-values sit at the edge of significance is the relevant honesty here: the effect is real, but it isn't large enough to justify FX policy designed around it.

**For an investor,** the Russia inversion is the most interesting datapoint. When Russian search interest for Turkish holidays and Russian arrivals to Turkey decoupled in 2022 — searches falling while arrivals rose — that wasn't noise, it was a regime change. Tourism flows from sanctioned economies stopped reflecting vacation demand and started reflecting capital and people in motion. That has direct read-through to Turkish real estate, retail banking in the southern coast, and any business whose revenue mix depends on which kind of Russian shows up.

[SCREENSHOT: German search interest for "Türkei Urlaub" vs. monthly arrivals, shifted by one month to show the lead relationship — `04_trends_de_vs_arrivals.png`]
[SCREENSHOT: Correlation heatmap of arrivals, real FX rates, and Trends series — `05_correlation_heatmap.png`]

## Limitations

I'd rather be the one to name these than let a reader find them.

- **Aggregate arrivals mask country-level heterogeneity.** I'm regressing total foreign arrivals on a small set of macros, which means I'm averaging behaviour across 240-odd source countries. A German visitor and a Saudi visitor respond to very different things; collapsing them into one series throws away most of that signal.
- **The YoY transform costs me a year.** After differencing, the usable sample is 99 monthly observations. That's enough to detect a meaningful effect but not enough to slice it cleanly by sub-period, and it's why the p-values are wider than I'd like.
- **Residual autocorrelation is real.** Durbin–Watson sits at 0.53 across every spec I tried. I'm using Newey–West HAC standard errors so the inference is honest, but a DW that low is the residuals telling me there's dynamic structure (an AR term, a lagged dependent, distributed FX lags) the static spec isn't capturing.
- **Russian CPI ends March 2022.** OECD stopped publishing after sanctions; FRED has no successor monthly series. Anything involving *real* RUB/TRY structurally ends there. I treated it as a constraint instead of inventing a proxy.

## What I'd do next

Three concrete moves, in order of payoff.

**Country-level panel.** TÜİK gives me arrivals broken down by country of origin in `tuik_country_year.csv`. Going from a single time series to a 25-country panel changes the question from "what moves total arrivals?" to "which countries respond to which shocks?" — and gives me roughly an order of magnitude more variation to fit on. The Russia inversion alone would be far easier to identify cleanly with a panel.

**A proper REER basket.** Instead of regressing on individual real bilateral rates and then arguing about which one to keep, build a tourism-weighted real effective exchange rate using each origin country's share of arrivals as the weight. That sidesteps the EUR/GBP collinearity problem by construction rather than by dropping a regressor.

**A dynamic specification.** AR(1) on the dependent variable, distributed lags on FX, and re-run the diagnostics. The DW of 0.53 is shouting that the static OLS is leaving information on the table. I'd start with an ARDL and let the lag structure earn its place.

## Technical stack

Python end-to-end: `pandas` for the panel build, `statsmodels` for the OLS, Newey–West HAC, ADF, and VIF diagnostics, `evds` / `fredapi` / `pytrends` for the API pulls, `openpyxl` for the TÜİK Excel parse, and `matplotlib` + `seaborn` for the figures. Everything orchestrated through Jupyter, with the notebooks programmatically generated from `.py` builders so the analysis is reproducible end-to-end — `nbconvert --execute` runs the whole pipeline from a clean checkout.
