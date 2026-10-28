"""Streamlit dashboard for the supplied QF5210 12-ETF backtest snapshot."""
import calendar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import COLORS, LABELS, MODELS, TICKERS, load_bundle, monthly_returns, period_curves

st.set_page_config(page_title="QF5210 | Alpha Trading", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {max-width:1500px;padding-top:2rem;padding-bottom:3rem;}
h1 {letter-spacing:-1.2px;font-weight:750!important;}
h2,h3 {letter-spacing:-.4px;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stMetricLabel"] {color:#64748b;}
.eyebrow {font-size:.75rem;letter-spacing:.16em;color:#0d9488;font-weight:700;}
.subtitle {color:#64748b;margin-bottom:1.6rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Checking archived results…", ttl=60)
def cached_bundle():
    return load_bundle()


def style_chart(fig, height=390, percent=False):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=15, t=25, b=15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial, sans-serif", color="#334155"),
                      legend=dict(orientation="h", y=-.18, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title=None)
    if percent:
        fig.update_yaxes(tickformat=".0%")
    return fig


def line_chart(frame, focus, percent=False, money=True):
    fig = go.Figure()
    for model in frame.columns:
        fig.add_trace(go.Scatter(x=frame.index, y=frame[model], name=LABELS[model], mode="lines",
                                line=dict(color=COLORS[model], width=3 if model == focus else 1.7,
                                          dash="dash" if model.startswith("BH") else "solid"),
                                hovertemplate=("%{y:.2%}" if percent else "$%{y:,.0f}" if money else "%{y:.3f}") + "<extra>%{fullData.name}</extra>"))
    return style_chart(fig, percent=percent)


def fmt(value, kind="number"):
    if not np.isfinite(value):
        return "N/A"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "money":
        return f"${value:,.0f}"
    return f"{value:.3f}"


try:
    bundle = cached_bundle()
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"The archived results could not be loaded safely: {exc}")
    st.info("Keep the Data Collection and Strategy Code folders beside streamlit_app.py. Restore the complete snapshot if files were edited.")
    st.stop()

manifest = bundle["manifest"]
metrics = bundle["metrics"]
with st.sidebar:
    st.markdown("### QF5210\nAlpha Trading")
    st.caption("12 ETFs · 4 strategies · 2 benchmarks")
    period = st.selectbox("Reporting period", list(manifest["periods"]), index=1, key="period")
    focus = st.selectbox("Focus portfolio", MODELS, format_func=LABELS.get, key="focus")
    chosen = st.multiselect("Compare portfolios", MODELS, default=MODELS, format_func=LABELS.get, key="comparison")
    st.caption("The focus portfolio is always included.")
    selected = [m for m in MODELS if m in chosen or m == focus]
    st.divider()
    st.caption("SNAPSHOT THROUGH")
    st.write("23 October 2026")
    st.caption("PERIOD STARTING VALUE")
    st.write("USD 100,000")
    st.caption("Net of 5 bps transaction costs on buys and sells. Results use archived data; no live prices.")

start, end = manifest["periods"][period]
table = metrics.loc[metrics.Period.eq(period) & metrics.Model.isin(selected)].copy()
row = table.set_index("Model").loc[focus]
sample = bundle["paths"][focus].loc[start:end]
wealth, drawdown = period_curves(bundle, period, selected)

st.markdown('<div class="eyebrow">QF5210 · SYSTEMATIC STRATEGIES</div>', unsafe_allow_html=True)
st.title("12-ETF performance dashboard")
st.markdown(f'<div class="subtitle">{period} &nbsp; / &nbsp; {start} → {end} &nbsp; / &nbsp; {int(row["Observations"]):,} trading sessions</div>', unsafe_allow_html=True)
if row["Observations"] < 252:
    st.warning(f"{period} contains only {int(row['Observations'])} sessions. CAGR, volatility, Sharpe and Sortino are annualised estimates from this short sample. Use total return as the direct observed result.")

st.subheader(LABELS[focus])
st.caption(manifest["models"][focus])
cards = [("Total return", "Total return", "percent"), ("Annual return (CAGR)", "Annual return", "percent"),
         ("Sharpe · excess over BIL", "Sharpe ratio", "number"), ("Maximum drawdown", "Max drawdown", "percent")]
for col, (title, metric, kind) in zip(st.columns(4), cards):
    col.metric(title, fmt(row[metric], kind))
cards2 = [("Ending value", "Ending value", "money"), ("Annual volatility", "Annual volatility", "percent"),
          ("Sortino · excess over BIL", "Sortino ratio", "number")]
cols = st.columns(4)
for col, (title, metric, kind) in zip(cols, cards2):
    col.metric(title, fmt(row[metric], kind))
cols[3].metric("ETF orders · excluding initial", f"{int(row['Number of trades']):,}")
st.write("")
performance, risk, allocation, activity, methodology = st.tabs(["Performance", "Risk", "Allocations", "Trading activity", "Methodology & checks"])

with performance:
    st.subheader("Growth of $100,000")
    st.caption("Rebased at the beginning of the selected period. Existing holdings continue across period boundaries.")
    st.plotly_chart(line_chart(wealth, focus), use_container_width=True)
    st.subheader("Portfolio comparison")
    display = table.set_index("Model")[["Total return", "Annual return", "Annual volatility", "Sharpe ratio", "Sortino ratio", "Max drawdown", "Ending value", "Number of trades"]]
    display.index = display.index.map(LABELS)
    display.index.name = "Portfolio"
    formats = {c: "{:.2%}" for c in ["Total return", "Annual return", "Annual volatility", "Max drawdown"]}
    formats.update({"Sharpe ratio": "{:.3f}", "Sortino ratio": "{:.3f}", "Ending value": "${:,.0f}", "Number of trades": "{:,.0f}"})
    st.dataframe(display.style.format(formats, na_rep="N/A"), use_container_width=True)
    st.download_button("Download period metrics", table.to_csv(index=False).encode(), f"{period.lower()}_dashboard_metrics.csv", "text/csv", key="download_metrics")
    st.subheader("Across training, validation and testing")
    st.caption("Each period starts at $100,000 for reporting. Compare within periods; the horizons differ substantially.")
    across = metrics.loc[metrics.Model.eq(focus)].set_index("Period")[["Observations", "Total return", "Annual return", "Sharpe ratio", "Max drawdown"]]
    st.dataframe(across.style.format({"Observations": "{:,}", "Total return": "{:.2%}", "Annual return": "{:.2%}", "Sharpe ratio": "{:.3f}", "Max drawdown": "{:.2%}"}), use_container_width=True)
    st.subheader("Monthly returns")
    monthly = monthly_returns(sample.net_return)
    fig = go.Figure(go.Heatmap(z=monthly.to_numpy(), x=[calendar.month_abbr[i] for i in monthly.columns],
                             y=[str(y) for y in monthly.index], colorscale=[[0, "#be123c"], [.5, "#f8fafc"], [1, "#0d9488"]],
                             zmid=0, text=monthly.to_numpy(), texttemplate="%{text:.1%}",
                             hoverongaps=False, hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
                             colorbar=dict(tickformat=".0%")))
    fig = style_chart(fig, max(220, min(620, 65 + 38 * len(monthly))))
    fig.update_yaxes(type="category", autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Compounded net daily returns. The first and last month include only sessions inside this period; October 2026 ends on 23 October. Blank months have no observations.")

with risk:
    st.subheader("Drawdown from the period high")
    st.plotly_chart(line_chart(drawdown, focus, percent=True), use_container_width=True)
    st.caption("The initial $100,000 is included as a high-water mark, so a loss on the first day counts.")
    a, b = st.columns(2)
    with a:
        st.subheader("Return and risk")
        fig = go.Figure()
        for _, item in table.iterrows():
            m = item.Model
            fig.add_trace(go.Scatter(x=[item["Annual volatility"]], y=[item["Annual return"]],
                                    mode="markers+text", text=[m], textposition="top center", name=LABELS[m],
                                    marker=dict(color=COLORS[m], size=16 if m == focus else 12),
                                    hovertemplate="Annual volatility: %{x:.2%}<br>CAGR: %{y:.2%}<extra>%{fullData.name}</extra>"))
        fig = style_chart(fig, 360, percent=True)
        fig.update_layout(showlegend=False, hovermode="closest")
        fig.update_xaxes(title="Annual volatility", tickformat=".0%")
        fig.update_yaxes(title="Annual return (CAGR)")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Daily return distribution")
        fig = go.Figure(go.Histogram(x=sample.net_return, nbinsx=45, marker_color=COLORS[focus], name=LABELS[focus]))
        fig = style_chart(fig, 360)
        fig.update_xaxes(title="Net daily return", tickformat=".1%")
        fig.update_yaxes(title="Sessions")
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Best session", fmt(row["Best day"], "percent"))
    b.metric("Worst session", fmt(row["Worst day"], "percent"))
    c.metric("Positive sessions", fmt(row["Positive days"], "percent"))
    st.caption("Positive sessions is a daily-return frequency, not a trade win rate.")
    st.subheader("Rolling 126-session Sharpe")
    if len(sample) < 126:
        st.info("This period has fewer than 126 sessions. A rolling Sharpe chart is not available.")
    else:
        rolling = {}
        for m in selected:
            r = bundle["paths"][m].loc[start:end].net_return
            ex = r - bundle["bil"].reindex(r.index)
            sd = ex.rolling(126).std(ddof=1)
            rolling[m] = np.sqrt(252) * ex.rolling(126).mean() / sd.where(sd > 1e-15)
        st.plotly_chart(line_chart(pd.DataFrame(rolling).iloc[125:], focus, money=False), use_container_width=True)
        st.caption("Uses only observations inside the selected reporting period; 252-session annualisation, daily excess returns over BIL.")

with allocation:
    st.subheader("End-of-day allocation")
    st.caption(f"{LABELS[focus]} · actual holdings after any trades executed at that day's close.")
    fig = go.Figure()
    palette = px.colors.qualitative.Safe
    for i, ticker in enumerate(TICKERS):
        if sample[ticker].max() > 1e-8:
            fig.add_trace(go.Scatter(x=sample.index, y=sample[ticker], stackgroup="weights", name=ticker,
                                    line=dict(width=.5, color=palette[i % len(palette)]),
                                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>"))
    fig = style_chart(fig, 430, percent=True)
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns([1, 1])
    last_weights = sample[TICKERS].iloc[-1]
    last_weights = last_weights[last_weights > 1e-8].sort_values(ascending=False)
    with a:
        st.subheader(f"Holdings on {end}")
        fig = go.Figure(go.Pie(labels=last_weights.index, values=last_weights, hole=.65,
                              marker=dict(colors=[palette[TICKERS.index(t) % len(palette)] for t in last_weights.index]),
                              textinfo="label+percent", hovertemplate="%{label}: %{percent}<extra></extra>"))
        st.plotly_chart(style_chart(fig, 340), use_container_width=True)
    with b:
        st.subheader("Average allocation in period")
        average = sample[TICKERS].mean().sort_values(ascending=False)
        st.dataframe(average[average > 1e-8].rename("Average weight").to_frame().style.format("{:.2%}"), use_container_width=True)
    signals = bundle["signals"].loc[bundle["signals"].model.eq(focus) & bundle["signals"].signal_date.between(start, end)]
    with st.expander("Decision signals and target weights"):
        if signals.empty:
            st.info("No strategy decisions in this period. Buy-and-hold benchmarks do not rebalance.")
        else:
            st.dataframe(signals, hide_index=True, use_container_width=True)
        st.caption("Targets are decided at close t. If the 3-percentage-point trade band is exceeded, execution occurs at close t+1.")

with activity:
    st.subheader("Orders, rebalances and costs")
    a, b, c, d = st.columns(4)
    a.metric("Buy orders", f"{int(row['ETF buy orders']):,}")
    b.metric("Sell orders", f"{int(row['ETF sell orders']):,}")
    c.metric("Portfolio rebalances", f"{int(row['Portfolio rebalances']):,}")
    d.metric("Actual ledger fees", f"${row['Actual fees (USD)']:,.2f}")
    st.caption("Order and rebalance counts exclude initial purchases. Fees are actual USD from the continuous portfolio, including the initial purchase cost in training; they are not rebased to the period's $100,000.")
    traded = bundle["trades"].loc[bundle["trades"].model.eq(focus) & bundle["trades"].event.eq("rebalance")
                                     & bundle["trades"].execution_date.between(start, end)].copy()
    st.subheader("Execution log")
    if traded.empty:
        st.info("No rebalance orders were executed by this portfolio in the selected period.")
    st.dataframe(traded, hide_index=True, use_container_width=True)
    st.download_button("Download execution log", traded.to_csv(index=False).encode(), f"{focus}_{period.lower()}_trades.csv", "text/csv")
    st.download_button("Download daily returns and holdings", sample.to_csv().encode(), f"{focus}_{period.lower()}_daily.csv", "text/csv")
    st.caption("Each execution-log row is one portfolio event containing one or more ETF orders; the archive does not contain individual round-trip trade P&L.")

with methodology:
    st.subheader("Models and measurement")
    st.dataframe(pd.DataFrame([{"Model": m, "Definition": manifest["models"][m]} for m in MODELS]), hide_index=True, use_container_width=True)
    st.markdown("""
- **Net returns:** archived daily returns after transaction costs; no new backtest or parameter search runs in this dashboard.
- **Total return:** compounded net daily returns over the selected period.
- **CAGR:** compounded growth raised to `252 / number of sessions`, minus one.
- **Volatility:** sample standard deviation of net returns × √252.
- **Sharpe:** mean daily excess return over BIL ÷ sample standard deviation of those excess returns × √252.
- **Sortino:** mean daily excess return over BIL ÷ downside deviation × √252. Downside deviation is `sqrt(mean(min(excess, 0)²))` over all observations. Added by this dashboard; absent from the original report.
- **Maximum drawdown:** largest decline from running peak wealth, including the period's initial capital.
- **Trading activity:** executed ETF orders by execution date, excluding initial purchases. One rebalance can contain several orders.

Undefined ratios are shown as N/A. Reporting periods rebase wealth only; they do not reset holdings or strategy state.
""")
    st.subheader("Snapshot checks")
    a, b, c = st.columns(3)
    a.metric("Metric reconciliation", f"{len(bundle['reconciliation'])}/{len(bundle['reconciliation'])}")
    b.metric("Input checksums matched", f"{len(bundle['hashes'])}/{len(bundle['hashes'])}")
    passed = bundle["causality"].passed.astype(str).str.lower().eq("true").sum()
    c.metric("Archived causality checks", f"{passed}/{len(bundle['causality'])}")
    st.caption("Metric and checksum checks run when the snapshot is loaded. Causality checks are the supplied backtest's archived results, not a new independent audit.")
    with st.expander("View metric reconciliation"):
        st.dataframe(bundle["reconciliation"], hide_index=True, use_container_width=True)
    with st.expander("View archived causality checks and input hashes"):
        st.dataframe(bundle["causality"], hide_index=True, use_container_width=True)
        st.dataframe(bundle["hashes"], hide_index=True, use_container_width=True)
    st.subheader("Data quality and research limitations")
    st.dataframe(bundle["issues"], hide_index=True, use_container_width=True)
    st.caption("The supplied collector flagged XLE's large return on 9 March 2020 for review and retained the observation. The dashboard preserves it.")
    st.markdown("The fixed ETF universe carries retrospective selection risk. Prior inspection of October results is an acknowledged research limitation. Testing has only 17 observations. Adjusted prices incorporate distributions and are not historical executable cash quotes. The model assumes fractional long-only holdings, no leverage or tax, full fills at the next close, and no final liquidation.")
    with st.expander("Backtest parameters and source manifest"):
        st.json(manifest)
    st.download_button("Download all-period dashboard metrics", metrics.to_csv(index=False).encode(), "all_period_dashboard_metrics.csv", "text/csv")

st.divider()
st.caption("QF5210 · Alpha Trading | Archived snapshot · USD | Dashboard calculations reconciled to the supplied backtest")
