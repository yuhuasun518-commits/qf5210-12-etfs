"""Read the archived backtest and calculate transparent, reconciled metrics."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Strategy Code" / "output_four_models"
DATA = ROOT / "Data Collection" / "output_12etf" / "run_20261027T132745_024986Z" / "data"
TICKERS = ["EWS", "SPY", "AGG", "GLD", "BIL", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLE", "DBC"]
MODELS = ["M1", "M2", "M3", "M4", "BH_12", "BH_SPY"]
LABELS = {"M1": "M1 · Equal weight", "M2": "M2 · Historical volatility",
          "M3": "M3 · GARCH", "M4": "M4 · GARCH + trend",
          "BH_12": "12-ETF buy & hold", "BH_SPY": "SPY buy & hold"}
COLORS = {"M1": "#2563eb", "M2": "#0d9488", "M3": "#8b5cf6", "M4": "#d97706",
          "BH_12": "#94a3b8", "BH_SPY": "#334155"}
METRIC_COLUMNS = ["Starting value", "Ending value", "Annual return", "Annual volatility",
                  "Sharpe ratio", "Max drawdown", "Number of trades"]


def read_daily(path):
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    if frame.empty or frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be present, unique, and increasing: {path.name}")
    required = ["net_return", "NAV", "fee_USD", *TICKERS]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing daily output columns: {path.name}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or nonfinite data: {path.name}")
    if (frame.net_return <= -1).any() or (frame.NAV <= 0).any() or (frame.fee_USD < 0).any():
        raise ValueError(f"Invalid returns, NAV, or costs: {path.name}")
    if (frame[TICKERS] < -1e-8).any().any() or not np.allclose(frame[TICKERS].sum(axis=1), 1, atol=1e-8):
        raise ValueError(f"Invalid long-only portfolio weights: {path.name}")
    return frame


def load_bundle():
    manifest = json.loads((OUTPUT / "run_manifest.json").read_text(encoding="utf-8"))
    hash_rows = []
    for filename, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
        hash_rows.append({"File": filename, "Match": actual == expected, "SHA-256": actual})
        if actual != expected:
            raise ValueError(f"The input snapshot differs from the backtest manifest: {filename}")
    paths = {m: read_daily(OUTPUT / f"daily_{m}.csv") for m in MODELS}
    reference_index = paths[MODELS[0]].index
    for model, frame in paths.items():
        if not frame.index.equals(reference_index):
            raise ValueError(f"Misaligned portfolio dates: {model}")
        expected_nav = manifest["parameters"]["initial_capital"] * (1 + frame.net_return).cumprod()
        if not np.allclose(frame.NAV, expected_nav, rtol=1e-8, atol=1e-5):
            raise ValueError(f"NAV and daily returns disagree: {model}")
    returns = pd.read_csv(DATA / "ETF12_Simple_Returns_Full.csv", index_col=0, parse_dates=True)
    if not returns.index.is_unique or returns.index.hasnans or not returns.index.is_monotonic_increasing:
        raise ValueError("Invalid ETF return dates.")
    bil = returns["BIL"].reindex(reference_index)
    if not np.isfinite(bil.to_numpy()).all():
        raise ValueError("BIL returns are missing on one or more portfolio dates.")
    trades = pd.read_csv(OUTPUT / "trades.csv", parse_dates=["signal_date", "execution_date"])
    signals = pd.read_csv(OUTPUT / "signals_and_target_weights.csv", parse_dates=["signal_date"])
    reported = pd.read_csv(OUTPUT / "performance_metrics.csv")
    rows = []
    for period, (start, end) in manifest["periods"].items():
        for model in MODELS:
            sample = paths[model].loc[start:end]
            if sample.empty or sample.index[0] != pd.Timestamp(start) or sample.index[-1] != pd.Timestamp(end):
                raise ValueError(f"Incomplete reporting period: {period} {model}")
            row = calculate_metrics(sample, bil, trades, model, **metric_parameters(manifest))
            rows.append({"Period": period, "Model": model, **row})
    metrics = pd.DataFrame(rows)
    checks = reconcile_metrics(metrics, reported)
    if not checks["Match"].all():
        raise ValueError("Dashboard metrics do not reconcile to performance_metrics.csv.")
    return {"manifest": manifest, "paths": paths, "bil": bil, "trades": trades,
            "signals": signals, "reported": reported, "metrics": metrics,
            "reconciliation": checks, "hashes": pd.DataFrame(hash_rows),
            "causality": pd.read_csv(OUTPUT / "causality_checks.csv"),
            "issues": pd.read_csv(DATA.parent / "reports" / "data_issues.csv")}


def metric_parameters(manifest):
    p = manifest["parameters"]
    return {"capital": p["initial_capital"], "sessions": p["annual_sessions"]}


def wealth_and_drawdown(returns, capital=100_000):
    """Include starting wealth when measuring drawdown, including a first-day loss."""
    values = np.r_[1.0, np.cumprod(1 + np.asarray(returns, dtype=float))]
    drawdown = values / np.maximum.accumulate(values) - 1
    return values * capital, drawdown


def calculate_metrics(sample, bil, trades, model, capital=100_000, sessions=252):
    r = sample.net_return.to_numpy(dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Metrics require at least two finite returns greater than -100%.")
    risk_free = bil.reindex(sample.index).to_numpy(dtype=float)
    if not np.isfinite(risk_free).all():
        raise ValueError("BIL observations must align with portfolio returns.")
    excess = r - risk_free
    wealth, dd = wealth_and_drawdown(r, capital)
    std = excess.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    executed = trades.loc[trades.model.eq(model) & trades.event.eq("rebalance")
                          & trades.execution_date.between(sample.index[0], sample.index[-1])]
    return {"Start date": str(sample.index[0].date()), "End date": str(sample.index[-1].date()),
            "Observations": len(r), "Starting value": capital, "Ending value": wealth[-1],
            "Total return": wealth[-1] / capital - 1,
            "Annual return": (wealth[-1] / capital) ** (sessions / len(r)) - 1,
            "Annual volatility": np.sqrt(sessions) * r.std(ddof=1),
            "Sharpe ratio": np.sqrt(sessions) * excess.mean() / std if std > 1e-15 else np.nan,
            "Sortino ratio": np.sqrt(sessions) * excess.mean() / downside if downside > 1e-15 else np.nan,
            "Max drawdown": dd.min(), "Positive days": np.mean(r > 0),
            "Best day": r.max(), "Worst day": r.min(),
            "Number of trades": int(executed.orders.sum()),
            "Portfolio rebalances": int(executed.orders.gt(0).sum()),
            "ETF buy orders": int(executed.buy_orders.sum()), "ETF sell orders": int(executed.sell_orders.sum()),
            "Actual fees (USD)": sample.fee_USD.sum()}


def reconcile_metrics(calculated, reported):
    key = ["Period", "Model"]
    left, right = calculated.set_index(key), reported.set_index(key)
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError("Reporting periods/models do not match the archived summary.")
    rows = []
    for idx in left.index:
        for column in [*METRIC_COLUMNS, "Observations"]:
            a, b = left.loc[idx, column], right.loc[idx, column]
            rows.append({"Period": idx[0], "Model": idx[1], "Metric": column,
                         "Dashboard": a, "Archived": b, "Difference": a-b,
                         "Match": bool(np.isclose(a, b, rtol=1e-8, atol=1e-8))})
    return pd.DataFrame(rows)


def period_curves(bundle, period, models):
    start, end = bundle["manifest"]["periods"][period]
    capital = bundle["manifest"]["parameters"]["initial_capital"]
    all_dates = bundle["paths"][MODELS[0]].index
    earlier = all_dates[all_dates < pd.Timestamp(start)]
    # 2010-01-04 is the archived initial purchase date; later periods use the previous observation.
    baseline = earlier[-1] if len(earlier) else bundle["trades"].execution_date.min()
    wealth, drawdown = {}, {}
    for model in models:
        sample = bundle["paths"][model].loc[start:end]
        index = pd.DatetimeIndex([baseline, *sample.index])
        w, dd = wealth_and_drawdown(sample.net_return, capital)
        wealth[model] = pd.Series(w, index=index)
        drawdown[model] = pd.Series(dd, index=index)
    return pd.DataFrame(wealth), pd.DataFrame(drawdown)


def monthly_returns(returns):
    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1
    frame = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.to_numpy()})
    return frame.pivot(index="Year", columns="Month", values="Return").reindex(columns=range(1, 13))
