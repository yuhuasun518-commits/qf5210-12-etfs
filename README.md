# QF5210 · 12-ETF Alpha Trading

Streamlit performance dashboard for four momentum portfolios and two buy-and-hold benchmarks. Uses the supplied archived backtest through **23 October 2026**. The original source, data, report, and results are preserved.

## Run the dashboard

Use Python 3.11 or 3.12. On Windows, double-click `start_dashboard.bat` to install dashboard dependencies and launch it in local mode. The terminal prints the local URL (usually `http://localhost:8501`). No API key or fresh market-data download is required.

For a manual Windows PowerShell launch, run from this folder:

```bash
python -m pip install -r requirements.txt
$env:QF5210_LOCAL_MODE="1"
python -m streamlit run streamlit_app.py
```

On macOS/Linux, replace the last two commands with `QF5210_LOCAL_MODE=1 python -m streamlit run streamlit_app.py`. Do not set this local-only flag on a hosted deployment.

## Dashboard views

- **Performance:** total return, CAGR, volatility, Sharpe, Sortino, drawdown, ending wealth, order counts, equity curves, monthly returns, and period comparisons.
- **Risk:** drawdowns, return-versus-volatility comparison, daily return distribution, and rolling 126-session Sharpe where enough observations exist.
- **Allocations:** actual end-of-day holdings, final weights, average weights, and decision targets.
- **Trading activity:** order counts, rebalance counts, actual USD costs, execution logs, and CSV downloads.
- **Methodology & checks:** metric definitions, source hashes, reconciliation to all 18 archived portfolio/period rows, archived causality checks, data warning, and research limitations.
- **Project files:** download the complete project ZIP, Word report, all-period results, or selected Python source; read source code directly in the app.

Select training, validation, or testing, choose a focus portfolio, and choose the portfolios to compare. Clearing the comparison selector keeps the focus portfolio visible. The default period is validation; the default focus is M1. No winner is automatically selected using testing results.

## Models

| Code | Model |
|---|---|
| M1 | Top-four 12–1 momentum, equal weights |
| M2 | Same momentum selection, inverse 21-return historical volatility |
| M3 | Same momentum selection, inverse GARCH(1,1) forecast volatility |
| M4 | M3, with allocations failing individual SMA200 moved to BIL |
| BH_12 | Initially equal-weighted 12-ETF buy-and-hold |
| BH_SPY | SPY buy-and-hold |

Universe: EWS, SPY, AGG, GLD, BIL, QQQ, IWM, EFA, EEM, XLK, XLE, DBC.

| Period | First return | Last return | Sessions |
|---|---|---|---:|
| Training | 2010-01-05 | 2022-12-30 | 3,271 |
| Validation | 2023-01-03 | 2026-09-30 | 939 |
| Testing | 2026-10-01 | 2026-10-23 | 17 |

## Measurement and limitations

Net daily returns include 5 bps transaction costs on buys and sells. Each reporting period starts at USD 100,000 for display, but holdings continue across boundaries. CAGR and volatility use 252 sessions per year. Sharpe uses daily excess returns over BIL and sample standard deviation. Maximum drawdown includes the initial capital, including a loss on the first day.

Sortino is an added dashboard metric: `sqrt(252) * mean(r - r_BIL) / sqrt(mean(min(r - r_BIL, 0)^2))`. Its denominator uses all daily observations. Undefined ratios display as N/A. Positive-session frequency is not a trade win rate. The source log contains portfolio execution events, not individual round-trip P&L.

ETF order counts exclude initial purchases. Initial purchase costs remain included in performance. Fees shown in the Trading activity tab are actual USD from the continuous portfolio, not rebased period fees.

**Testing has only 17 sessions; annualised testing statistics are unstable.** The supplied manifest acknowledges prior inspection of October results. Fixed-universe selection, adjusted-price assumptions, fractional holdings, no tax, no leverage, full next-close fills, and no final liquidation remain limitations. The data collector retained a flagged XLE return on 9 March 2020. The dashboard displays these source limitations without changing the data.

## Files and reproducibility

The application entry point is `streamlit_app.py`; calculations and validation are in `dashboard_data.py`. Original folders remain `Data Collection/` and `Strategy Code/`. The original `QF5210_Report.docx` is unchanged.

The pinned data snapshot is `Data Collection/output_12etf/run_20261027T132745_024986Z/data/`. Results come from `Strategy Code/output_four_models/`. The dashboard does not import or execute the strategy script. It checks the input SHA-256 hashes, validates daily NAV/returns/weights, and reconciles 144 numeric fields against `performance_metrics.csv` before displaying results. Results are cached for up to 60 seconds.

To intentionally rerun the original backtest, install its separate dependencies and run it. It overwrites the archived output directory, so first retain a copy if the old results must be preserved:

```bash
python -m pip install -r "Strategy Code/requirements.txt"
python "Strategy Code/QF5210_Four_Models_Spyder.py"
```

To test the dashboard:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

Create an app in [Streamlit Community Cloud](https://share.streamlit.io/). Keep the GitHub repository private. Select branch `main` and main file path `streamlit_app.py`. Choose Python 3.11 or 3.12. Dependencies are in the root `requirements.txt`.

Before deployment, generate a random access token (for example, `secrets.token_urlsafe(32)`) and save it as the root `share_token` value in Streamlit's **Advanced settings → Secrets**. Never commit the real token. Hosted content stays blocked if the token is missing, shorter than 32 characters, or does not match the visitor's link. Do not set `QF5210_LOCAL_MODE` on the host.

To allow recipients without Streamlit invitations, the Streamlit app endpoint must be publicly reachable. The app then checks the access token before showing any project data or downloads. Share the deployed app URL with `?access=YOUR_RANDOM_TOKEN` appended. Anyone receiving the complete link can open the dashboard and download all project files. Keep the complete link private to your intended audience; they can forward it. Changing the token in Streamlit Secrets invalidates the old link on the next app run. The bare app URL only shows an access message. This is application-level link access, not a native Streamlit "unlisted" setting; the GitHub repository remains private.

The download ZIP uses `download_manifest.json` as an explicit file allowlist. Runtime secrets, environment files, git history, and symlinks are excluded. If you deliberately add project deliverables later, update that manifest.

See [Streamlit's deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). A local dashboard and GitHub repository do not by themselves create a hosted Streamlit app.

## Source preservation

The uploaded ZIP is the source of the original files. All added files are dashboard code, configuration, requirements, tests, documentation, and a Windows launcher. No credentials are required or committed. `.gitignore` excludes local environments, caches, `.env` files, and Streamlit secrets.
