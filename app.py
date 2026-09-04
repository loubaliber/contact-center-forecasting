"""Contact center call volume forecasting dashboard.

Recommends a forecasting method based on a live backtest instead of a
neutral toggle, and translates the recommendation into a staffing cost.
Decomposition, autocorrelation, and the full walk-forward backtest
detail behind this page's numbers live in the companion notebook
(Forecasting_Notebook_Baliber.ipynb) — this page only shows what
changes the staffing decision.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Contact Center Forecast", layout="wide")

METHODS = ["Seasonal Naive", "Holt-Winters"]
BACKTEST_HORIZON = 14
BACKTEST_FOLDS = 5
DEFAULT_HOURLY_WAGE = 22.0


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/calls.csv", parse_dates=["date"])


def naive_forecast(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Repeat the last `window` days' pattern (copy last week forward)."""
    last_week = history.iloc[-window:].to_numpy()
    reps = math.ceil(horizon / window)
    return np.tile(last_week, reps)[:horizon]


def holt_winters_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(
        history, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    return model.forecast(horizon).to_numpy()


FORECASTERS = {"Seasonal Naive": naive_forecast, "Holt-Winters": holt_winters_forecast}


@st.cache_data
def backtest(history: pd.Series, horizon: int, n_folds: int) -> tuple[pd.DataFrame, int]:
    """Walk-forward backtest: MAE/MAPE per method, averaged over n_folds folds."""
    rows = []
    folds_used = 0
    for fold in range(n_folds):
        cut = len(history) - horizon * (fold + 1)
        if cut < horizon * 4:
            break
        train, test = history.iloc[:cut], history.iloc[cut:cut + horizon]
        for name, forecaster in FORECASTERS.items():
            pred = forecaster(train, horizon)
            rows.append(
                {
                    "method": name,
                    "mae": mean_absolute_error(test, pred),
                    "mape": mean_absolute_percentage_error(test, pred) * 100,
                }
            )
        folds_used += 1
    summary = pd.DataFrame(rows).groupby("method")[["mae", "mape"]].mean()
    return summary, folds_used


def erlang_c_service_level(n: int, calls_per_interval: float, aht_seconds: float,
                            interval_seconds: int, target_answer_seconds: int) -> float:
    """Service level (share of calls answered within target_answer_seconds) for n agents."""
    a = (calls_per_interval * aht_seconds) / interval_seconds  # traffic intensity, erlangs
    if n <= a:
        return 0.0
    erlang_b = 1.0
    for i in range(1, n + 1):
        erlang_b = (a * erlang_b) / (i + a * erlang_b)
    p_wait = (n * erlang_b) / (n - a * (1 - erlang_b))
    if p_wait <= 0:
        return 1.0
    return 1 - p_wait * math.exp(-(n - a) * (target_answer_seconds / aht_seconds))


def erlang_c_agents(calls_per_interval: float, aht_seconds: float, interval_seconds: int,
                     target_sl: float, target_answer_seconds: int) -> int:
    """Minimum agents to hit target service level, via Erlang C."""
    traffic_intensity = (calls_per_interval * aht_seconds) / interval_seconds  # erlangs
    n = max(1, math.ceil(traffic_intensity))
    while n < 200:
        sl = erlang_c_service_level(n, calls_per_interval, aht_seconds, interval_seconds, target_answer_seconds)
        if sl >= target_sl:
            return n
        n += 1
    return n


def agents_for_forecast(forecast_values: np.ndarray, avg_aht: float, target_sl: float) -> int:
    avg_calls = float(np.mean(forecast_values))
    calls_per_hour = avg_calls / 24
    return erlang_c_agents(calls_per_hour, avg_aht, 3600, target_sl, 20)


df = load_data()
history = df.set_index("date")["calls"]
avg_aht = float(df["aht_seconds"].tail(30).mean())

st.title("Contact Center Call Volume Forecast")
st.caption(
    "Synthetic data — daily call volume and AHT (Average Handle Time, the "
    "average length of a call in seconds) for one queue, 3 years of history."
)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
with c2:
    target_sl = st.slider("Target service level", 0.5, 0.95, 0.80, step=0.05)

# ---------------------------------------------------------------------------
# Backtest-driven verdict
# ---------------------------------------------------------------------------
scores, folds_used = backtest(history, BACKTEST_HORIZON, BACKTEST_FOLDS)
winner = scores["mae"].idxmin()
runner_up = [m for m in METHODS if m != winner][0]

winner_forecast = FORECASTERS[winner](history, horizon)
runner_up_forecast = FORECASTERS[runner_up](history, horizon)

agents_winner = agents_for_forecast(winner_forecast, avg_aht, target_sl)
agents_runner_up = agents_for_forecast(runner_up_forecast, avg_aht, target_sl)
cost_delta_per_hour = (agents_runner_up - agents_winner) * DEFAULT_HOURLY_WAGE

if cost_delta_per_hour == 0:
    cost_sentence = (
        f"At this forecast horizon and target, both methods round to the same "
        f"**{agents_winner}-agent** headcount — a {scores.loc[runner_up, 'mae'] - scores.loc[winner, 'mae']:.0f} "
        f"calls/day accuracy gap is too small to change the staffing number once Erlang C "
        f"rounds to a whole agent, at this daily, flat-spread-across-24-hours level of detail. "
        f"The gap shows up more at finer (e.g. half-hourly) staffing granularity — see the notebook."
    )
else:
    cost_sentence = (
        f"Staffing off {runner_up}'s forecast instead would cost "
        f"**\\${abs(cost_delta_per_hour):,.0f}/hr {'more' if cost_delta_per_hour > 0 else 'less'}** "
        f"at the same {target_sl:.0%} target service level "
        f"(assumes \\${DEFAULT_HOURLY_WAGE:.0f}/hr per agent)."
    )

st.success(
    f"**{winner} beats {runner_up}** in backtesting — "
    f"{scores.loc[winner, 'mae']:.0f} vs {scores.loc[runner_up, 'mae']:.0f} calls/day "
    f"average error (MAE, lower is better), averaged over the last {folds_used} "
    f"backtest windows of {BACKTEST_HORIZON} days each. {cost_sentence}"
)

method = st.radio(
    "Forecasting method to use below",
    METHODS,
    index=METHODS.index(winner),
    format_func=lambda m: f"{m} — recommended" if m == winner else f"{m} — backtest runner-up",
    horizontal=True,
)
st.caption(
    "This backtest runs on daily-aggregated data. The companion notebook backtests the same "
    "two methods on half-hourly data, where noise behaves differently at that resolution — "
    "the two can legitimately disagree on which method wins. Both results are real; they're "
    "answering the question at different granularities."
)
forecast_values = winner_forecast if method == winner else runner_up_forecast

# ---------------------------------------------------------------------------
# Chart: actual vs. both forecasts
# ---------------------------------------------------------------------------
future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent["date"], y=recent["calls"], name="Actual calls", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(
    x=future_dates, y=winner_forecast, name=f"{winner} (recommended)",
    line=dict(color="#2ca02c", dash="dash"),
))
fig.add_trace(go.Scatter(
    x=future_dates, y=runner_up_forecast, name=f"{runner_up} (runner-up)",
    line=dict(color="#d62728", dash="dot"),
))
fig.update_layout(
    title="Actual vs. forecasted daily call volume (last 90 days + forecast)",
    xaxis_title="Date",
    yaxis_title="Calls",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Error metrics, head to head
# ---------------------------------------------------------------------------
st.subheader("Backtest error, head to head")
st.caption(
    f"MAE (Mean Absolute Error, in calls/day) and MAPE (Mean Absolute Percentage Error) "
    f"from {folds_used}-fold walk-forward backtesting — each fold trains on the past and "
    f"tests on a {BACKTEST_HORIZON}-day window it never saw."
)
display_scores = scores.rename(index=lambda m: f"{m} (recommended)" if m == winner else f"{m} (runner-up)")
st.dataframe(
    display_scores.rename(columns={"mae": "MAE (calls/day)", "mape": "MAPE (%)"}).round(1),
    width="stretch",
)

# ---------------------------------------------------------------------------
# Staffing
# ---------------------------------------------------------------------------
st.subheader("What this means for staffing")
st.caption(
    "Erlang C staffing estimate: minimum agents needed to answer the target "
    "share of calls within 20 seconds, given the forecasted volume and AHT."
)

agents_needed = agents_winner if method == winner else agents_runner_up

m1, m2, m3 = st.columns(3)
m1.metric("Avg. forecasted calls/day", f"{float(np.mean(forecast_values)):,.0f}")
m2.metric("Avg. handle time (last 30d)", f"{avg_aht:,.0f}s")
m3.metric(f"Agents needed ({method})", agents_needed)

st.subheader("What-if: staff a different number of agents")
st.caption(
    "Override the recommended headcount and see the resulting service level and hourly cost, "
    "instead of only the number Erlang C says you need."
)

wcol1, wcol2 = st.columns(2)
with wcol1:
    what_if_agents = st.number_input(
        "Agents to staff", min_value=1, max_value=200, value=agents_needed, step=1
    )
with wcol2:
    hourly_wage = st.number_input("Hourly wage per agent ($)", min_value=1.0, value=DEFAULT_HOURLY_WAGE, step=1.0)

calls_per_hour = float(np.mean(forecast_values)) / 24
what_if_sl = erlang_c_service_level(
    n=int(what_if_agents),
    calls_per_interval=calls_per_hour,
    aht_seconds=avg_aht,
    interval_seconds=3600,
    target_answer_seconds=20,
)

w1, w2, w3 = st.columns(3)
w1.metric("Resulting service level", f"{what_if_sl:.0%}",
          delta=f"{(what_if_sl - target_sl):+.0%} vs. target", delta_color="normal")
w2.metric("Hourly staffing cost", f"${what_if_agents * hourly_wage:,.0f}/hr")
w3.metric("Cost vs. recommended headcount",
          f"${(what_if_agents - agents_needed) * hourly_wage:,.0f}/hr",
          delta_color="inverse")

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
aht_load_corr = df["calls"].corr(df["aht_seconds"])
n_spike_days = int(df["is_spike_day"].sum())

st.caption(
    f"Handle time and call volume are negatively correlated (r = {aht_load_corr:.2f}) — "
    "agents move faster when the queue is busy."
)
st.caption(
    f"{n_spike_days} of the {len(df)} days in this dataset are injected demand spikes "
    "(outages, promotions, holiday rushes) — by construction, no forecasting method here "
    "could have predicted them from history alone. Treat the forecast above as what happens "
    "on a typical day, not a guarantee against a spike."
)

with st.expander("Show raw data"):
    st.dataframe(df.tail(30), width="stretch")

st.caption(
    "For time series decomposition, autocorrelation, and the full walk-forward backtest "
    "behind these numbers, see `Forecasting_Notebook_Baliber.ipynb`."
)
