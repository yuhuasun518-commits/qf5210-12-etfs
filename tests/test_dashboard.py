from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_data import MODELS, calculate_metrics, load_bundle, monthly_returns, period_curves, wealth_and_drawdown


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_all_archived_metrics_match(bundle):
    assert len(bundle["metrics"]) == 18
    assert len(bundle["reconciliation"]) == 144
    assert bundle["reconciliation"].Match.all()


@pytest.mark.parametrize("period,observations", [("Training", 3271), ("Validation", 939), ("Testing", 17)])
def test_rebased_curves_and_trade_counts(bundle, period, observations):
    wealth, drawdown = period_curves(bundle, period, MODELS)
    assert len(wealth) == observations + 1
    assert (wealth.iloc[0] == 100000).all()
    expected = bundle["metrics"].query("Period == @period").set_index("Model")
    np.testing.assert_allclose(wealth.iloc[-1], expected.loc[MODELS, "Ending value"])
    np.testing.assert_allclose(drawdown.min(), expected.loc[MODELS, "Max drawdown"])
    assert (expected["Number of trades"] == expected["ETF buy orders"] + expected["ETF sell orders"]).all()
    assert (expected.loc[["BH_12", "BH_SPY"], "Number of trades"] == 0).all()


def test_first_day_loss_is_a_drawdown():
    wealth, dd = wealth_and_drawdown([-0.1, 0.05], 100)
    np.testing.assert_allclose(wealth, [100, 90, 94.5])
    assert dd.min() == pytest.approx(-0.1)


def test_monthly_returns_compound_and_leave_missing_months_blank():
    r = pd.Series([.1, -.1, .05], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-03-02"]))
    monthly = monthly_returns(r)
    assert monthly.loc[2026, 1] == pytest.approx(-.01)
    assert np.isnan(monthly.loc[2026, 2])
    assert monthly.loc[2026, 3] == pytest.approx(.05)


def test_missing_risk_free_dates_are_rejected(bundle):
    sample = bundle["paths"]["M1"].iloc[-3:]
    with pytest.raises(ValueError, match="BIL observations"):
        calculate_metrics(sample, bundle["bil"].drop(sample.index[0]), bundle["trades"], "M1")


def test_zero_downside_and_zero_volatility_are_undefined(bundle):
    sample = bundle["paths"]["M1"].iloc[-3:].copy()
    sample["net_return"] = .001
    bil = pd.Series(0., index=sample.index)
    result = calculate_metrics(sample, bil, bundle["trades"], "M1")
    assert np.isnan(result["Sortino ratio"])
    assert np.isnan(result["Sharpe ratio"])


@pytest.mark.parametrize("period", ["Training", "Validation", "Testing"])
def test_app_portfolio_selections(period, monkeypatch):
    monkeypatch.setenv("QF5210_LOCAL_MODE", "1")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=40).run()
    assert not app.exception
    app.selectbox(key="period").select(period)
    for model in MODELS:
        app.selectbox(key="focus").select(model).run()
        assert not app.exception, [e.message for e in app.exception]
        assert not app.error
        assert len(app.metric) >= 8
    if period == "Testing":
        assert "17 sessions" in app.warning[0].value
    app.multiselect(key="comparison").set_value([]).run()
    assert not app.exception
    assert not app.error
