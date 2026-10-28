# -*- coding: utf-8 -*-
"""QF5210 - Spyder-ready collector: adjusted OPEN/CLOSE prices and simple returns.
Version: 2026-10-27 adjusted-open update.

SETUP: In Spyder's IPython console run the following once, then restart kernel:
    %pip install --upgrade yfinance pandas numpy pandas_market_calendars
Open this file and press F5. No command-line arguments are needed.

Outputs: output_12etf/run_<timestamp>/ beside this script (change OUTPUT_ROOT).
Prices: 2010-01-01 through 2026-10-23 inclusive.
Training: 2010-01-01 to 2022-12-31.
Validation: 2023-01-01 to 2026-09-30 (September has only 30 days).
Testing: 2026-10-01 to 2026-10-23.

AdjustedOpen[t] = Open[t] * AdjClose[t] / Close[t].
Open and close prices are downloaded together, on the same adjustment basis.
Additional files: ETF12_Adjusted_Open_{Full,Training,Validation,Testing}.csv.
Adjusted prices are historical analysis series, not actual cash execution quotes.
For an opening-execution backtest, model the relevant overnight/intraday holding
intervals; the existing simple-return files remain CLOSE-TO-CLOSE returns.

Returns: AdjClose[t] / AdjClose[t-1] - 1. Decimal units: 0.01 = 1%.
No forward/backward filling, no zero filling, and no outlier clipping.
Only the structural first return row is omitted: no pre-2010 price is fetched.
With complete data, prices start 2010-01-04 and returns start 2010-01-05.
Compute returns BEFORE splitting so validation/testing keep their first return.

All tickers are US-listed and USD-quoted; EWS supplies Singapore exposure.
Yahoo adjusted prices incorporate split/distribution adjustments. Do not add
dividends again to returns computed from adjusted prices. They are retrospective
and can be revised; archive the dated snapshots. The fixed ETF universe selected
today does not remove retrospective universe-selection bias. Fit models/scalers
on training data only. A return dated t is known only after that session closes.

Documentation:
https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
https://pandas-market-calendars.readthedocs.io/en/latest/usage.html
"""

# %% Imports and configuration
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time

try:
    import numpy as np
    import pandas as pd
    import pandas_market_calendars as mcal
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "In Spyder's IPython console run:\n"
        "%pip install --upgrade yfinance pandas numpy pandas_market_calendars\n"
        "Then restart the kernel and run this script again."
    ) from exc

TICKERS = ["EWS", "SPY", "AGG", "GLD", "BIL", "QQQ",
           "IWM", "EFA", "EEM", "XLK", "XLE", "DBC"]
START_DATE = "2010-01-01"
END_DATE = "2026-10-23"
PERIODS = {
    "Training": ("2010-01-01", "2022-12-31"),
    "Validation": ("2023-01-01", "2026-09-30"),
    "Testing": ("2026-10-01", "2026-10-23"),
}
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_ROOT = SCRIPT_DIR / "output_12etf"
# Alternative example: OUTPUT_ROOT = Path(r"C:\Users\yuhua\Desktop\ETF_Data")
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 30
LARGE_RETURN_THRESHOLD = 0.20  # Absolute daily return >20%: review, do not delete.
YFINANCE_REPAIR = False       # Preserve vendor values, without heuristic repairs.
ISSUE_COLUMNS = ["Severity", "Ticker", "Date", "Issue", "Detail"]


# %% Download and validation helpers
def add_issue(issues, severity, ticker, date, issue, detail):
    date_text = "" if date is None else pd.Timestamp(date).strftime("%Y-%m-%d")
    issues.append(dict(zip(ISSUE_COLUMNS,
                           [severity, ticker, date_text, issue, str(detail)])))


def expected_sessions():
    """Common US equity session calendar, including ad-hoc closures."""
    days = mcal.get_calendar("NYSE").schedule(
        start_date=START_DATE, end_date=END_DATE).index
    return pd.DatetimeIndex(days).tz_localize(None).rename("Date")


def normalise_download(frame, ticker):
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if ticker in frame.columns.get_level_values(level):
                frame = frame.xs(ticker, level=level, axis=1)
                break
    if isinstance(frame.columns, pd.MultiIndex) or "Adj Close" not in frame.columns:
        raise ValueError("No unambiguous 'Adj Close'; raw Close is not a substitute.")
    required = ["Open", "Close", "Adj Close"]
    if not set(required).issubset(frame.columns) or not frame.columns.is_unique:
        raise ValueError("Response must contain unique Open, Close and Adj Close fields.")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if dates.isna().any():
        raise ValueError("Response contains invalid dates.")
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    frame.index = dates.normalize().rename("Date")
    return frame


def download_ticker(ticker, sessions, raw_dir, issues):
    """Retry errors/partial data; retain the most complete response for audit."""
    best, best_score = None, -1
    # Yahoo end is EXCLUSIVE. Request October 24 to include October 23.
    end_exclusive = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            frame = yf.download(
                ticker, start=START_DATE, end=end_exclusive, interval="1d",
                auto_adjust=False, back_adjust=False, repair=YFINANCE_REPAIR,
                actions=True, keepna=True, threads=False, progress=False,
                ignore_tz=True, rounding=False, timeout=TIMEOUT_SECONDS,
                multi_level_index=False,
            )
            if frame is None or frame.empty:
                raise ValueError("Yahoo returned no rows; possibly blocked or rate limited.")
            frame = normalise_download(frame, ticker)
            candidate = frame[["Open", "Close", "Adj Close"]].apply(
                pd.to_numeric, errors="coerce").astype(float)
            valid = candidate.index[(np.isfinite(candidate) & (candidate > 0)).all(axis=1)]
            score = len(sessions.intersection(valid.unique()))
            if score > best_score:
                best, best_score = frame, score
            if score == len(sessions):
                break
            print(f"  Attempt {attempt}: {score}/{len(sessions)} sessions have valid prices.")
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {exc}", flush=True)
            add_issue(issues, "INFO", ticker, None, "download_attempt_failed", exc)
        if attempt < MAX_ATTEMPTS:
            time.sleep(min(3 * attempt, 15))
    if best is None:
        add_issue(issues, "ERROR", ticker, None, "download_failed", "All attempts failed.")
        return pd.DataFrame(columns=["Open", "Close", "Adj Close"])
    best.to_csv(raw_dir / f"{ticker}_yahoo_history.csv", index_label="Date",
                date_format="%Y-%m-%d", float_format="%.15g")
    return best


def with_adjusted_open(frame, ticker, issues):
    """Apply Yahoo's same-day adjustment factor to the opening price."""
    result = frame.copy()
    values = frame.reindex(columns=["Open", "Close", "Adj Close"]).apply(
        pd.to_numeric, errors="coerce").astype(float)
    valid = np.isfinite(values) & (values > 0)
    for column in values.columns:
        for day in values.index[~valid[column]]:
            add_issue(issues, "ERROR", ticker, day, "invalid_adjustment_input",
                      f"{column} is missing, non-numeric, non-finite or non-positive.")
    values = values.where(valid)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result["Adjustment Factor"] = values["Adj Close"] / values["Close"]
        result["Adj Open"] = values["Open"] * result["Adjustment Factor"]
    return result


def clean_price_series(frame, ticker, sessions, issues, field="Adj Close"):
    """Report duplicates and invalid prices; never impute a missing price."""
    if frame.empty:
        return pd.Series(np.nan, index=sessions, name=ticker)
    if not frame.index.is_monotonic_increasing:
        add_issue(issues, "INFO", ticker, None, "dates_sorted", "Sorted downloaded dates.")
    prices = pd.to_numeric(frame[field], errors="coerce").astype(float)
    for day in prices.index[prices.index.duplicated(keep=False)].unique():
        values = prices.loc[[day]]
        if values.nunique(dropna=False) == 1:
            add_issue(issues, "WARNING", ticker, day, "identical_duplicate",
                      f"{field}: kept one identical adjusted price; raw rows retained.")
        else:
            add_issue(issues, "ERROR", ticker, day, "conflicting_duplicate",
                      f"{field}: conflicting prices; set to missing for investigation.")
            prices.loc[day] = np.nan
    prices = prices.loc[~prices.index.duplicated(keep="first")].sort_index()
    for day in prices.index.difference(sessions):
        add_issue(issues, "WARNING", ticker, day, "unexpected_date",
                  "Not an expected session in the requested range; excluded from model files.")
    bad = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
    for day, value in prices.loc[bad].items():
        add_issue(issues, "ERROR", ticker, day, "invalid_price", f"{field}={value}")
    prices.loc[bad] = np.nan
    return prices.reindex(sessions).rename(ticker)


def make_returns_and_splits(prices):
    """Preserve split-boundary returns and prevent returns spanning missing days."""
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be sorted and unique.")
    returns = prices.pct_change(fill_method=None).iloc[1:].copy()
    splits = {name: returns.loc[start:end].copy()
              for name, (start, end) in PERIODS.items()}
    combined = pd.concat(list(splits.values()))
    if not combined.index.is_unique or not combined.equals(returns):
        raise ValueError("Periods must cover the returns once, with no gaps or overlap.")
    return returns, splits


def quality_reports(prices, returns, splits, issues):
    for ticker in TICKERS:
        for day in prices.index[prices[ticker].isna()]:
            add_issue(issues, "ERROR", ticker, day, "missing_price",
                      "Missing/non-numeric/invalid adjusted price on expected session; not filled.")
        for day in returns.index[returns[ticker].isna()]:
            add_issue(issues, "ERROR", ticker, day, "missing_return",
                      "Current or immediately preceding session price missing; not filled.")
        for day, value in returns[ticker].loc[
                returns[ticker].abs() > LARGE_RETURN_THRESHOLD].items():
            add_issue(issues, "WARNING", ticker, day, "large_return",
                      f"Decimal return={value:.10g}; review raw prices/actions. Retained.")
    rows = []
    for name, frame in {"Full": returns, **splits}.items():
        for ticker in TICKERS:
            values = frame[ticker]
            rows.append({
                "Period": name, "Ticker": ticker, "Rows": len(frame),
                "First_date": frame.index.min().strftime("%Y-%m-%d") if len(frame) else "",
                "Last_date": frame.index.max().strftime("%Y-%m-%d") if len(frame) else "",
                "Numeric_returns": int(values.notna().sum()),
                "Missing_returns": int(values.isna().sum()),
                "Min_return": values.min(), "Max_return": values.max(),
                "Large_return_flags": int((values.abs() > LARGE_RETURN_THRESHOLD).sum()),
                "Dates_sorted": frame.index.is_monotonic_increasing,
                "Dates_unique": frame.index.is_unique,
            })
    return pd.DataFrame(rows)


# %% Main workflow
def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = OUTPUT_ROOT / f"run_{stamp}"
    raw_dir, report_dir = run_dir / "raw", run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir()
    sessions = expected_sessions()
    issues, series_list, open_series_list = [], [], []
    print("QF5210 - 12 ETF adjusted-price data collector")
    print(f"Requested dates: {START_DATE} to {END_DATE}, inclusive")
    print("Validation ends 2026-09-30; September 31 does not exist.")
    print(f"Expected price sessions: {len(sessions)} | Output: {run_dir}\n")
    for number, ticker in enumerate(TICKERS, 1):
        print(f"[{number:02d}/{len(TICKERS)}] Downloading {ticker} ...", flush=True)
        raw = download_ticker(ticker, sessions, raw_dir, issues)
        adjusted = with_adjusted_open(raw, ticker, issues)
        values = clean_price_series(adjusted, ticker, sessions, issues)
        opens = clean_price_series(adjusted, ticker, sessions, issues, field="Adj Open")
        series_list.append(values)
        open_series_list.append(opens)
        print(f"  Valid adjusted closes: {values.notna().sum()}/{len(sessions)}; "
              f"adjusted opens: {opens.notna().sum()}/{len(sessions)}", flush=True)
        time.sleep(0.5)
    prices = pd.concat(series_list, axis=1).reindex(columns=TICKERS)
    prices.index.name = "Date"
    open_prices = pd.concat(open_series_list, axis=1).reindex(columns=TICKERS)
    open_prices.index.name = "Date"
    open_splits = {name: open_prices.loc[start:end].copy()
                   for name, (start, end) in PERIODS.items()}
    for ticker in TICKERS:
        for day in open_prices.index[open_prices[ticker].isna()]:
            add_issue(issues, "ERROR", ticker, day, "missing_adjusted_open",
                      "Adjusted opening price missing on an expected session; not filled.")
    returns, splits = make_returns_and_splits(prices)
    quality = quality_reports(prices, returns, splits, issues)
    errors = sum(item["Severity"] == "ERROR" for item in issues)
    warnings = sum(item["Severity"] == "WARNING" for item in issues)
    status = ("FAIL_INCOMPLETE_OR_INVALID" if errors else
              "REVIEW_WARNINGS" if warnings else "PASS_AUTOMATED_CHECKS")
    data_dir = run_dir / ("incomplete_data_DO_NOT_MODEL" if errors else "data")
    data_dir.mkdir()
    outputs = {"ETF12_Adjusted_Close_Full.csv": prices,
               "ETF12_Adjusted_Open_Full.csv": open_prices,
               "ETF12_Simple_Returns_Full.csv": returns}
    outputs.update({f"ETF12_Adjusted_Open_{name}.csv": frame
                    for name, frame in open_splits.items()})
    outputs.update({f"ETF12_Simple_Returns_{name}.csv": frame
                    for name, frame in splits.items()})
    for filename, frame in outputs.items():
        frame.to_csv(data_dir / filename, index_label="Date", date_format="%Y-%m-%d",
                     float_format="%.15g", na_rep="")
    quality.to_csv(report_dir / "data_quality_summary.csv", index=False, float_format="%.15g")
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(report_dir / "data_issues.csv", index=False)
    price_quality = []
    for field, panel in [("Adj Close", prices), ("Adj Open", open_prices)]:
        for name, frame in {"Full": panel, **{
                name: panel.loc[start:end] for name, (start, end) in PERIODS.items()}}.items():
            for ticker in TICKERS:
                price_quality.append({
                    "Price_field": field, "Period": name, "Ticker": ticker,
                    "Rows": len(frame), "Valid_prices": int(frame[ticker].notna().sum()),
                    "Missing_prices": int(frame[ticker].isna().sum()),
                    "First_date": str(frame.index.min().date()),
                    "Last_date": str(frame.index.max().date()),
                })
    pd.DataFrame(price_quality).to_csv(report_dir / "price_quality_summary.csv", index=False)
    metadata = {
        "status": status, "downloaded_at_utc": stamp, "source": "Yahoo Finance via yfinance",
        "tickers": TICKERS, "currency": "USD", "price_fields": ["Adj Close", "Adj Open"],
        "adjusted_open_formula": "Open[t] * AdjClose[t] / Close[t]",
        "return_price_field": "Adj Close",
        "requested_start_inclusive": START_DATE, "requested_end_inclusive": END_DATE,
        "periods_inclusive": PERIODS, "calendar": "NYSE", "auto_adjust": False,
        "repair": YFINANCE_REPAIR, "return_formula": "AdjClose[t]/AdjClose[t-1]-1",
        "return_units": "decimal", "missing_policy": "Keep missing values; never fill",
        "initial_return_row_omitted": str(sessions[0].date()),
        "large_return_threshold": LARGE_RETURN_THRESHOLD,
        "errors": errors, "warnings": warnings, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["yfinance", "pandas", "numpy", "pandas_market_calendars"]},
        "sha256": {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(run_dir.rglob("*.csv"))},
    }
    (report_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = [
        f"STATUS: {status}", f"Errors: {errors}; review warnings: {warnings}",
        f"Data directory: {data_dir}",
        "All bounds are inclusive. Validation corrected to September 30.",
        "Adjusted Open = Open * (Adj Close / Close), using a single download snapshot.",
        "Adjusted Open price files are split by date, including each period's first session.",
        "Simple-return files remain CLOSE-TO-CLOSE; decimal units (0.01 = 1%).",
        "Adjusted prices are not actual historical cash execution quotes.",
        "Only the initial structural empty return row is omitted.",
        "Returns computed before splitting; missing values/rows are retained.",
        "PASS: automated checks passed; not independent verification of vendor prices.",
        "REVIEW_WARNINGS: review data_issues.csv, including large returns.",
        "FAIL: incomplete/invalid data; resolve errors before modelling.",
        "Raw Yahoo history and corporate actions are in raw/ for investigation.",
        "Adjusted history may be revised; archive this dated snapshot.",
        "The fixed ETF universe does not remove retrospective selection bias.", "",
    ]
    for name, frame in splits.items():
        lines.append(f"{name}: {len(frame)} rows | {frame.index.min().date()} to "
                     f"{frame.index.max().date()} | missing cells={int(frame.isna().sum().sum())}")
    (report_dir / "DATA_STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    if errors:
        raise RuntimeError(f"Data checks failed. Read {report_dir / 'DATA_STATUS.txt'}")
    return prices, returns, splits, run_dir


if __name__ == "__main__":
    # Available in Spyder's Variable Explorer after a successful run.
    adjusted_close, simple_returns, return_splits, output_directory = main()
    adjusted_open = pd.read_csv(
        output_directory / "data" / "ETF12_Adjusted_Open_Full.csv",
        index_col="Date", parse_dates=True)