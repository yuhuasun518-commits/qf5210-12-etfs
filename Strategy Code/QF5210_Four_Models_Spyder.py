# -*- coding: utf-8 -*-
"""QF5210: four ETF models, chronological next-close backtest.

SPYDER: open this file and press F5 (Run file). Results also appear as the
`performance_metrics` DataFrame in Variable Explorer. Install dependencies in
the SAME Python environment as Spyder: pip install numpy pandas scipy arch

PORTABILITY: keep Strategy Code beside Data Collection, or put a `data` folder
beside this script. Otherwise set DATA_DIR below to the input CSV directory.
No internet, Excel, cached forecasts, or previous backtest outputs are needed.

M1: top-four 12-1 momentum, equal weights.
M2: same selection, inverse 21-return historical volatility.
M3: same selection, inverse GARCH(1,1) forecast volatility.
M4: M3 weights; allocations failing each ETF's SMA200 move to BIL.
Benchmarks: initially equal-weighted 12-ETF buy-and-hold and SPY buy-and-hold.
They invest at the initial close, pay initial transaction costs, and then drift
without rebalancing. All three period tables contain these six portfolios.

All four use a 3-percentage-point absolute weight band, including BIL.
Once any weight gap exceeds 0.03, execute the entire target next close.
Number of trades = individual executed ETF buy/sell orders, including BIL,
excluding initial purchases to match the revised report. Their cost stays
in performance. Portfolio rebalance counts are saved separately.

Signal information ends at close t. Orders decided then execute at close t+1.
New holdings first earn the t+1 to t+2 close return. Parameters are fixed.
Rolling refits use observations already available on each decision date,
including earlier validation/test observations as time advances.

All periods are portions of ONE continuous portfolio, rebased to USD100,000
for reporting. Training includes the initial BIL purchase cost and warm-up.

GARCH API: https://arch.readthedocs.io/en/stable/univariate/forecasting.html
"""

# %% Imports and user settings
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import importlib.metadata
import json
import platform
import warnings

try:
    import numpy as np
    import pandas as pd
    from arch import arch_model
except ImportError as exc:
    raise ImportError(
        "Install numpy pandas scipy arch in Spyder's Python environment. "
        "Example: python -m pip install numpy pandas scipy arch"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = SCRIPT_DIR.parent / "Data Collection" / "output_12etf" / "run_20261027T132745_024986Z" / "data"
OUTPUT_DIR = SCRIPT_DIR / "output_four_models"
DATA_RUN = "run_20261027T132745_024986Z"  # Fixed snapshot; never silently pick latest.
RUN_CONSISTENCY_CHECKS = True  # Compare calculations at reporting boundaries.

TICKERS = ("EWS", "SPY", "AGG", "GLD", "BIL", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLE", "DBC")
BIL = TICKERS.index("BIL")
RISK = np.array([i for i in range(len(TICKERS)) if i != BIL])
MODELS = ("M1", "M2", "M3", "M4")
BENCHMARKS = ("BH_12", "BH_SPY")
PORTFOLIOS = MODELS + BENCHMARKS
ROW_LABELS = {**{m: m for m in MODELS}, "BH_12": "12-ETF buy-and-hold",
              "BH_SPY": "SPY buy-and-hold"}
MODEL_NAMES = {
    "M1": "Momentum + equal weights",
    "M2": "Momentum + historical inverse volatility",
    "M3": "Momentum + GARCH inverse volatility",
    "M4": "Momentum + GARCH + individual SMA200",
    "BH_12": "Initially equal-weighted 12-ETF buy-and-hold",
    "BH_SPY": "SPY buy-and-hold",
}
PERIODS = {
    "Training": ("2010-01-05", "2022-12-30"),
    "Validation": ("2023-01-03", "2026-09-30"),
    "Testing": ("2026-10-01", "2026-10-23"),
}
FIRST_PRICE_DATE = "2010-01-04"
METRIC_COLUMNS = ["Starting value", "Ending value", "Annual return",
                  "Annual volatility", "Sharpe ratio", "Max drawdown", "Number of trades"]
ORDER_TOLERANCE = 1e-10  # Notional as fraction of pre-trade NAV; ignores roundoff.


@dataclass(frozen=True)
class Parameters:
    initial_capital: float = 100_000.0
    annual_sessions: int = 252
    momentum_boundary: int = 252
    momentum_skip: int = 21
    top_n: int = 4
    warmup: int = 252
    decision_interval: int = 21
    historical_window: int = 21
    garch_minimum: int = 252
    garch_window: int = 504
    garch_horizon: int = 22
    trend_window: int = 200
    target_cap: float = 0.50
    trade_band: float = 0.03
    cost_rate: float = 0.0005


PARAMS = Parameters()


# %% Read and validate local data (no filling or interpolation)
def find_data_dir(override=None):
    if override is not None:
        candidates = [Path(override).expanduser()]
    else:
        candidates = [
            SCRIPT_DIR / "data",
            SCRIPT_DIR.parent / "Data Collection" / "output_12etf" / DATA_RUN / "data",
            SCRIPT_DIR.parent / "data",
        ]
    for directory in candidates:
        if (directory / "ETF12_Adjusted_Close_Full.csv").is_file():
            return directory.resolve()
    raise FileNotFoundError(
        "Cannot find ETF12_Adjusted_Close_Full.csv. Keep Strategy Code beside "
        "Data Collection, place the CSVs in a data folder beside the script, "
        "or set DATA_DIR explicitly. Checked: " + "; ".join(map(str, candidates))
    )


def read_panel(path):
    panel = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.index.hasnans:
        raise ValueError(f"Invalid dates: {path.name}")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be unique and increasing: {path.name}")
    if set(panel.columns) != set(TICKERS) or len(panel.columns) != len(TICKERS):
        raise ValueError(f"Expected exactly the 12 ETF columns: {path.name}")
    panel = panel.loc[:, list(TICKERS)].astype(float)
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError(f"Missing or nonfinite values; no automatic filling: {path.name}")
    return panel


def load_inputs(directory):
    price_file = directory / "ETF12_Adjusted_Close_Full.csv"
    prices = read_panel(price_file)
    if (prices.to_numpy() <= 0).any():
        raise ValueError("All adjusted prices must be positive.")
    final_date = max(end for _, end in PERIODS.values())
    # Pin the report interval; any later prices in an updated file are excluded.
    prices = prices.loc[FIRST_PRICE_DATE:final_date].copy()
    if len(prices) < 2 or str(prices.index[0].date()) != FIRST_PRICE_DATE:
        raise ValueError(f"The initial valuation price must be {FIRST_PRICE_DATE}.")
    calculated = prices.pct_change(fill_method=None).iloc[1:]
    source_files = [price_file]
    return_file = directory / "ETF12_Simple_Returns_Full.csv"
    if return_file.is_file():
        returns = read_panel(return_file).loc[calculated.index[0]:final_date]
        if not returns.index.equals(calculated.index):
            raise ValueError("Close and returns date indices differ.")
        if not np.allclose(returns, calculated, atol=1e-12, rtol=1e-9):
            raise ValueError("Supplied simple returns disagree with adjusted prices.")
        # Use the validated source return precision to reproduce the report.
        source_files.append(return_file)
    else:
        returns = calculated
    for period, (start, end) in PERIODS.items():
        sample = returns.loc[start:end]
        if len(sample) < 2 or sample.index[0] != pd.Timestamp(start) or sample.index[-1] != pd.Timestamp(end):
            raise ValueError(f"Incomplete {period} interval: expected {start} to {end}.")
        split_file = directory / f"ETF12_Simple_Returns_{period}.csv"
        if split_file.is_file():
            split = read_panel(split_file).loc[start:end]
            if not split.index.equals(sample.index) or not np.allclose(split, sample, atol=1e-12, rtol=1e-9):
                raise ValueError(f"{period} split disagrees with the full returns file.")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    return prices, returns, hashes


# %% Portfolio construction: only history ending on the current signal date
def capped_inverse_weights(volatility, selected, p):
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Inverse volatility requires finite positive estimates.")
    active = np.zeros(len(TICKERS))
    active[selected] = 1.0 / volatility
    weights = np.zeros(len(TICKERS))
    remaining = 1.0
    for _ in TICKERS:
        if active.sum() == 0:
            break
        trial = remaining * active / active.sum()
        hit = (trial > p.target_cap + 1e-14) & (active > 0)
        if not hit.any():
            weights += trial
            break
        weights[hit] = p.target_cap
        active[hit] = 0.0
        remaining = 1.0 - weights.sum()
    weights[BIL] += max(0.0, 1.0 - weights.sum())
    return weights


def forecast_volatility(history, historical_vol, p):
    """Fit a fresh zero-mean normal GARCH(1,1), using past returns only."""
    values = np.asarray(history[-p.garch_window:], dtype=float)
    if len(values) < p.garch_minimum:
        raise ValueError("Insufficient past returns for GARCH.")
    detail = {"observations": len(values), "fallback": False, "fallback_reason": "",
              "omega_percent_squared": np.nan, "alpha": np.nan, "beta": np.nan,
              "fit_warnings": ""}
    try:
        with warnings.catch_warnings(record=True) as caught:
            fit = arch_model(values * 100.0, mean="Zero", vol="GARCH",
                             p=1, o=0, q=1, power=2.0, dist="normal",
                             rescale=False).fit(disp="off", show_warning=False)
        detail["fit_warnings"] = " | ".join(str(item.message) for item in caught)
        if fit.convergence_flag != 0:
            raise RuntimeError(f"Optimizer status {fit.convergence_flag}")
        variance = fit.forecast(horizon=p.garch_horizon, method="analytic",
                                reindex=False).variance.to_numpy()[-1] / 10_000.0
        # Entry is at next close: sizing covers forecast steps 2 through 22.
        result = float(np.sqrt(variance[1:p.garch_horizon].mean()))
        if not np.isfinite(result) or result <= 0:
            raise ValueError("Invalid GARCH forecast")
        detail.update(omega_percent_squared=float(fit.params["omega"]),
                      alpha=float(fit.params["alpha[1]"]), beta=float(fit.params["beta[1]"]))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result = float(historical_vol)
        detail.update(fallback=True, fallback_reason=str(exc))
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Both GARCH and historical fallback are invalid.")
    detail["daily_sizing_volatility"] = result
    return result, detail


def make_targets(price_history, return_history, p):
    """Construct model weights from the supplied price and return histories."""
    if len(return_history) < p.warmup:
        raise ValueError("Signal requested before warm-up is complete.")
    prices = price_history.to_numpy()
    ret = return_history.to_numpy()
    score = prices[-1 - p.momentum_skip] / prices[-1 - p.momentum_boundary] - 1.0
    selected = RISK[np.argsort(-score[RISK], kind="stable")[:p.top_n]]
    hist_vol = ret[-p.historical_window:].std(axis=0, ddof=1)
    sma = prices[-p.trend_window:].mean(axis=0)
    targets = np.zeros((len(MODELS), len(TICKERS)))
    targets[0, selected] = 1.0 / p.top_n
    targets[1] = capped_inverse_weights(hist_vol[selected], selected, p)
    predicted = []
    fits = []
    for j in selected:
        vol, detail = forecast_volatility(ret[:, j], hist_vol[j], p)
        predicted.append(vol)
        detail.update(signal_date=return_history.index[-1], ETF=TICKERS[j],
                      estimation_start=return_history.index[max(0, len(ret)-p.garch_window)],
                      estimation_end=return_history.index[-1])
        fits.append(detail)
    targets[2] = capped_inverse_weights(np.array(predicted), selected, p)
    targets[3] = targets[2].copy()
    failed = selected[prices[-1, selected] <= sma[selected]]
    targets[3, BIL] += targets[3, failed].sum()
    targets[3, failed] = 0.0
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-12) or (targets < -1e-12).any():
        raise RuntimeError("Invalid target allocations.")
    if (targets[:, RISK] > p.target_cap + 1e-12).any():
        raise RuntimeError("Target cap exceeded.")
    selected_names = ",".join(TICKERS[j] for j in selected)
    return targets, selected_names, fits


def execute_rebalance(current_weights, target, rate):
    """Solve q + rate*sum(abs(q*target-current)) = 1; q is retained NAV."""
    q = 1.0
    for _ in range(15):
        q = 1.0 - rate * np.abs(q * target - current_weights).sum()
    amounts = q * target - current_weights
    if abs(q + rate * np.abs(amounts).sum() - 1.0) > 1e-12:
        raise RuntimeError("Self-financing cost calculation failed.")
    return q, amounts


# %% Event-by-event simulation; current returns precede pending trades
def backtest(prices, returns, p=PARAMS, verbose=True):
    if not prices.index[1:].equals(returns.index):
        raise ValueError("Backtest price/return alignment failed.")
    if not (0 <= p.cost_rate < 0.01) or p.top_n * p.target_cap < 1:
        raise ValueError("Unsupported cost rate or infeasible position cap.")
    weights = np.zeros((4, len(TICKERS)))
    weights[:, BIL] = 1.0
    nav = np.full(4, p.initial_capital)
    pending = [None] * 4
    paths = {model: [] for model in MODELS}
    trades, signals, fits = [], [], []
    return_values = returns.to_numpy()
    decision_number = 0

    for t, date in enumerate(returns.index):
        previous_nav = nav.copy()
        # Initial BIL purchase at the initial price date, before the first return.
        fees = np.zeros(4)
        if t == 0:
            nav /= 1.0 + p.cost_rate
            fees = previous_nav - nav
            for m, model in enumerate(MODELS):
                trades.append(dict(model=model, signal_date=pd.NaT,
                    execution_date=prices.index[0], event="initial_purchase",
                    pre_NAV=previous_nav[m], post_NAV=nav[m], fee_USD=fees[m],
                    turnover=nav[m]/previous_nav[m], orders=1, buy_orders=1, sell_orders=0))
        factors = 1.0 + return_values[t]
        for m, model in enumerate(MODELS):
            gross = float(weights[m] @ factors)
            nav[m] *= gross
            weights[m] = weights[m] * factors / gross
            if pending[m] is not None:
                signal_date, target = pending[m]
                if not signal_date < date:
                    raise RuntimeError("Execution must follow the signal date.")
                before = nav[m]
                q, amounts = execute_rebalance(weights[m], target, p.cost_rate)
                nav[m] *= q
                fee = before - nav[m]
                fees[m] += fee
                weights[m] = target
                trades.append(dict(model=model, signal_date=signal_date,
                    execution_date=date, event="rebalance", pre_NAV=before,
                    post_NAV=nav[m], fee_USD=fee,
                    turnover=float(np.abs(amounts).sum()),
                    orders=int((np.abs(amounts) > ORDER_TOLERANCE).sum()),
                    buy_orders=int((amounts > ORDER_TOLERANCE).sum()),
                    sell_orders=int((amounts < -ORDER_TOLERANCE).sum())))
                pending[m] = None
            paths[model].append([nav[m]/previous_nav[m]-1.0, nav[m], fees[m], *weights[m]])

        if t >= p.warmup - 1 and (t - (p.warmup - 1)) % p.decision_interval == 0:
            # t is a return row; the corresponding price row is t+1.
            # Construct the target using the observations through this date.
            targets, selected, current_fits = make_targets(
                prices.iloc[:t+2], returns.iloc[:t+1], p)
            fits.extend(current_fits)
            for m, model in enumerate(MODELS):
                difference = float(np.max(np.abs(targets[m] - weights[m])))
                trigger = difference > p.trade_band
                signals.append(dict(model=model, signal_date=date,
                    selected=selected, max_weight_difference=difference,
                    trade_triggered=trigger, **dict(zip(TICKERS, targets[m]))))
                if trigger:
                    pending[m] = (date, targets[m].copy())
            decision_number += 1
            if verbose and decision_number % 40 == 0:
                print(f"Computed {decision_number} decisions through {date.date()}", flush=True)

    columns = ["net_return", "NAV", "fee_USD", *TICKERS]
    result = {m: pd.DataFrame(rows, index=returns.index, columns=columns)
              for m, rows in paths.items()}
    return result, pd.DataFrame(trades), pd.DataFrame(signals), pd.DataFrame(fits)


# %% Passive benchmarks: initial purchases, then hold without rebalancing
def buy_and_hold(prices, returns, p=PARAMS):
    if not prices.index[1:].equals(returns.index):
        raise ValueError("Benchmark price/return alignment failed.")
    initial_weights = np.zeros((2, len(TICKERS)))
    initial_weights[0] = 1.0 / len(TICKERS)
    initial_weights[1, TICKERS.index("SPY")] = 1.0
    paths, trades = {}, []
    for model, initial in zip(BENCHMARKS, initial_weights):
        weights = initial.copy()
        nav = p.initial_capital
        rows = []
        for t, values in enumerate(returns.to_numpy()):
            previous = nav
            fee = 0.0
            if t == 0:
                nav /= 1.0 + p.cost_rate
                fee = previous - nav
                orders = int((initial > 0).sum())
                trades.append(dict(model=model, signal_date=pd.NaT,
                    execution_date=prices.index[0], event="initial_purchase",
                    pre_NAV=previous, post_NAV=nav, fee_USD=fee,
                    turnover=nav/previous, orders=orders, buy_orders=orders, sell_orders=0))
            factors = 1.0 + values
            gross = float(weights @ factors)
            nav *= gross
            weights = weights * factors / gross
            rows.append([nav/previous-1.0, nav, fee, *weights])
        paths[model] = pd.DataFrame(rows, index=returns.index,
            columns=["net_return", "NAV", "fee_USD", *TICKERS])
    return paths, pd.DataFrame(trades)


# %% Metrics: each period is rebased; actual holdings are never restarted
def summarize_trading(trades):
    """Count orders by EXECUTION date, not signal date; exclude initial purchase."""
    rows = []
    for period, (start, end) in PERIODS.items():
        for model in PORTFOLIOS:
            executed = trades.loc[
                trades.model.eq(model) & trades.event.eq("rebalance")
                & trades.execution_date.between(pd.Timestamp(start), pd.Timestamp(end))]
            buys = int(executed.buy_orders.sum())
            sells = int(executed.sell_orders.sum())
            orders = int(executed.orders.sum())
            if buys + sells != orders:
                raise RuntimeError("Order count does not equal buys plus sells.")
            rows.append({"Period": period, "Model": model,
                         "Number of trades": orders, "ETF buy orders": buys,
                         "ETF sell orders": sells,
                         "Portfolio rebalances": int(executed.orders.gt(0).sum())})
    return pd.DataFrame(rows)


def calculate_metrics(paths, returns, trades, p=PARAMS):
    trading = summarize_trading(trades).set_index(["Period", "Model"])
    rows = []
    for period, (start, end) in PERIODS.items():
        for model in PORTFOLIOS:
            sample = paths[model].loc[start:end]
            r = sample["net_return"].to_numpy()
            if len(r) < 2:
                raise ValueError(f"Not enough observations: {period} {model}")
            excess = r - returns.loc[sample.index, "BIL"].to_numpy()
            wealth = np.r_[1.0, np.cumprod(1.0 + r)]
            ex_sd = excess.std(ddof=1)
            rows.append({"Period": period, "Model": model,
                "Start date": str(sample.index[0].date()),
                "End date": str(sample.index[-1].date()), "Observations": len(r),
                "Starting value": p.initial_capital,
                "Ending value": p.initial_capital * wealth[-1],
                "Annual return": wealth[-1] ** (p.annual_sessions/len(r)) - 1.0,
                "Annual volatility": np.sqrt(p.annual_sessions)*r.std(ddof=1),
                "Sharpe ratio": np.sqrt(p.annual_sessions)*excess.mean()/ex_sd if ex_sd > 0 else np.nan,
                "Max drawdown": float(np.min(wealth/np.maximum.accumulate(wealth)-1.0)),
                "Number of trades": int(trading.loc[(period, model), "Number of trades"])})
    return pd.DataFrame(rows)


def verify_period_consistency(prices, returns, full_paths, p=PARAMS):
    """Compare full-run and period-end calculations at both boundaries."""
    checks = []
    for cutoff in (PERIODS["Training"][1], PERIODS["Validation"][1]):
        print(f"Checking calculation consistency through {cutoff} ...", flush=True)
        shortened, _, _, _ = backtest(prices.loc[:cutoff], returns.loc[:cutoff], p=p, verbose=False)
        passive, _ = buy_and_hold(prices.loc[:cutoff], returns.loc[:cutoff], p=p)
        shortened.update(passive)
        for model in PORTFOLIOS:
            expected = full_paths[model].loc[:cutoff]
            actual = shortened[model]
            delta_return = float(np.max(np.abs(actual.net_return - expected.net_return)))
            delta_weights = float(np.max(np.abs(actual.loc[:, list(TICKERS)].to_numpy()
                                               - expected.loc[:, list(TICKERS)].to_numpy())))
            passed = delta_return <= 1e-12 and delta_weights <= 1e-12
            checks.append(dict(cutoff=cutoff, model=model, passed=passed,
                               max_return_difference=delta_return,
                               max_weight_difference=delta_weights))
            if not passed:
                raise RuntimeError(f"Calculation consistency check failed: {cutoff} {model}")
    return pd.DataFrame(checks)


def format_table(frame):
    result = frame.copy()
    for column in ("Starting value", "Ending value"):
        result[column] = result[column].map(lambda x: f"{x:,.2f}")
    for column in ("Annual return", "Annual volatility", "Max drawdown"):
        result[column] = result[column].map(lambda x: f"{100*x:.2f}%")
    result["Sharpe ratio"] = result["Sharpe ratio"].map(lambda x: f"{x:.3f}")
    result["Number of trades"] = result["Number of trades"].astype(int)
    result["Model"] = result["Model"].map(ROW_LABELS)
    return result


def compare_report(metrics):
    """Optional read-only comparison AFTER simulation; never used for signals."""
    reference_path = SCRIPT_DIR / "report_reference_metrics.csv"
    if not reference_path.is_file():
        return pd.DataFrame()
    reference = pd.read_csv(reference_path).set_index(["Period", "Model"])
    strategy_rows = reference.index.get_level_values("Model").isin(MODELS)
    if "Rebalancing band" not in reference.columns or not np.allclose(reference.loc[strategy_rows, "Rebalancing band"], PARAMS.trade_band):
        raise ValueError("Report reference must use the same rebalancing band as the run.")
    actual = metrics.set_index(["Period", "Model"])
    if not reference.index.is_unique or set(reference.index) != set(actual.index):
        raise ValueError("Report reference must contain all four models and both benchmarks in every period.")
    rows = []
    for key, row in actual.iterrows():
        if key not in reference.index:
            continue
        for column in METRIC_COLUMNS:
            reported = float(reference.loc[key, column])
            observed = float(row[column])
            tolerance = (0.0 if column == "Number of trades" else
                         0.005 if column in ("Starting value", "Ending value") else 5e-8)
            rows.append({"Period": key[0], "Model": key[1], "Metric": column,
                         "Report": reported, "Recomputed": observed,
                         "Difference": observed-reported,
                         "Matches": abs(observed-reported) <= tolerance})
    return pd.DataFrame(rows)


# %% Run in Spyder: F5. Returned objects remain in Variable Explorer.
def main():
    directory = find_data_dir(DATA_DIR)
    prices, returns, hashes = load_inputs(directory)
    print(f"Input directory: {directory}")
    print(f"Four fixed models; {100*PARAMS.trade_band:g}-percentage-point band; next-close execution; costs included.")
    print("Fitting GARCH from raw past returns. This can take a few minutes.", flush=True)
    paths, trades, signals, fits = backtest(prices, returns)
    passive_paths, passive_trades = buy_and_hold(prices, returns)
    paths.update(passive_paths)
    trades = pd.concat([trades, passive_trades], ignore_index=True)
    metrics = calculate_metrics(paths, returns, trades)
    trading = summarize_trading(trades)
    checks = verify_period_consistency(prices, returns, paths) if RUN_CONSISTENCY_CHECKS else pd.DataFrame()
    comparison = compare_report(metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "performance_metrics.csv", index=False, float_format="%.12g")
    readable = format_table(metrics)
    readable.to_csv(OUTPUT_DIR / "performance_metrics_formatted.csv", index=False)
    lines = []
    period_tables = {}
    for period in PERIODS:
        numeric_part = metrics.loc[metrics.Period == period, ["Model", *METRIC_COLUMNS]].set_index("Model")
        numeric_part = numeric_part.rename(index=ROW_LABELS)
        period_tables[period] = numeric_part
        numeric_part.to_csv(OUTPUT_DIR / f"{period.lower()}_performance.csv", float_format="%.12g")
        part = readable.loc[readable.Period == period, ["Model", *METRIC_COLUMNS]]
        part.to_csv(OUTPUT_DIR / f"{period.lower()}_performance_formatted.csv", index=False)
        title = f"{period}: {PERIODS[period][0]} to {PERIODS[period][1]}"
        table = title + "\n" + part.to_string(index=False)
        print("\n" + table)
        lines.append(table)
    notes = (
        "All values are USD. Each period is rebased to 100,000; holdings continue.\n"
        "Annual return is geometric CAGR using 252 sessions per year.\n"
        "Sharpe uses daily BIL excess returns and their sample standard deviation.\n"
        "Volatility uses sample standard deviation of net portfolio returns.\n"
        "Drawdown includes the reporting period's initial value.\n"
        "Number of trades = individual executed ETF buys/sells, including BIL, by execution date.\n"
        "Displayed counts exclude initial purchases for all portfolios; their costs remain included.\n"
        "One portfolio rebalance may contain several ETF trades; see trading_activity.csv.\n"
        "The fixed 3-percentage-point band applies to all ETFs in M1-M4 and all periods.\n"
        "Benchmarks: initially equal-weighted 12 ETFs and SPY only; buy once, then no rebalancing.\n"
        "Benchmark weights and holdings continue across periods; reporting rebasing does not reset them.\n"
        "Benchmark trade counts are zero after excluding their initial 12 and 1 purchases.\n"
        "Testing has only 17 observations with the supplied snapshot; annualised metrics are unstable.\n"
        "Assumptions: fixed ETF universe; adjusted distributions; fractional long-only holdings;\n"
        "no leverage or tax; full fills at next close; 5 bps on buys and sells; no final liquidation.\n"
    )
    (OUTPUT_DIR / "performance_summary.txt").write_text("\n\n".join(lines) + "\n\n" + notes, encoding="utf-8")
    for model, path in paths.items():
        path.to_csv(OUTPUT_DIR / f"daily_{model}.csv", index_label="Date")
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    trading.to_csv(OUTPUT_DIR / "trading_activity.csv", index=False)
    signals.to_csv(OUTPUT_DIR / "signals_and_target_weights.csv", index=False)
    fits.to_csv(OUTPUT_DIR / "garch_fit_log.csv", index=False)
    if not checks.empty:
        checks.to_csv(OUTPUT_DIR / "period_consistency_checks.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(OUTPUT_DIR / "report_comparison.csv", index=False)
        print(f"\nReport matches: {comparison.Matches.sum()} / {len(comparison)} metric values.")
    manifest = {"parameters": asdict(PARAMS), "periods": PERIODS,
                "models": MODEL_NAMES, "source_sha256": hashes,
                "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "arch")},
                "consistency_checks_run": RUN_CONSISTENCY_CHECKS,
                "garch_fits": len(fits), "garch_fallbacks": int(fits.fallback.sum()),
                "trade_count_definition": "Executed ETF buy/sell orders; all initial purchases excluded; assigned by execution date",
                "benchmark_initial_orders": {"BH_12": 12, "BH_SPY": 1},
                "execution": "next_session_close", "notes": notes}
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nGARCH fits: {len(fits)}; historical-volatility fallbacks: {fits.fallback.sum()}")
    print(f"Saved results to: {OUTPUT_DIR}")
    print(notes)
    return metrics, paths, trades, signals, fits, checks, comparison, trading, period_tables


if __name__ == "__main__":
    (performance_metrics, daily_paths, trade_log, signal_log, garch_fit_log,
     consistency_checks, report_comparison, trading_activity, period_tables) = main()
    training_table = period_tables["Training"]
    validation_table = period_tables["Validation"]
    testing_table = period_tables["Testing"]
