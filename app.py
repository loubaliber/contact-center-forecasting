"""Contact center call volume forecasting — sample dashboard (synthetic data)."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Contact Center Forecast", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/calls.csv", parse_dates=["date"])
    return df


def naive_forecast(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Repeat the average of the last `window` days' weekday pattern."""
    last_week = history.iloc[-window:].to_numpy()
    reps = math.ceil(horizon / window)
    return np.tile(last_week, reps)[:horizon]


def holt_winters_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(
        history, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    return model.forecast(horizon).to_numpy()


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


df = load_data()

st.title("Contact Center Call Volume Forecast")
st.caption(
    "Synthetic data — daily call volume and AHT (Average Handle Time, "
    "the average length of a call in seconds) for one queue."
)

col1, col2, col3 = st.columns(3)
with col1:
    horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
with col2:
    method = st.radio(
        "Forecasting method",
        ["Naive (last week's pattern)", "Holt-Winters (trend + seasonality)"],
        index=1,
    )
with col3:
    target_sl = st.slider("Target service level", 0.5, 0.95, 0.80, step=0.05)

history = df.set_index("date")["calls"]

if method.startswith("Naive"):
    forecast_values = naive_forecast(history, horizon)
else:
    forecast_values = holt_winters_forecast(history, horizon)

future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_series = pd.Series(forecast_values, index=future_dates)

recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=recent["date"], y=recent["calls"], name="Actual calls", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, name="Forecast", line=dict(color="#d62728", dash="dash")))
fig.update_layout(
    title="Actual vs. forecasted daily call volume (last 90 days + forecast)",
    xaxis_title="Date",
    yaxis_title="Calls",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, width="stretch")

st.subheader("What this means for staffing")
st.caption(
    "Erlang C staffing estimate: minimum agents needed to answer the target "
    "share of calls within 20 seconds, given the forecasted volume and AHT."
)

avg_forecast_calls = float(np.mean(forecast_values))
avg_aht = float(df["aht_seconds"].tail(30).mean())
calls_per_hour = avg_forecast_calls / 24
agents_needed = erlang_c_agents(
    calls_per_interval=calls_per_hour,
    aht_seconds=avg_aht,
    interval_seconds=3600,
    target_sl=target_sl,
    target_answer_seconds=20,
)

naive_avg_calls = float(np.mean(naive_forecast(history, horizon)))
naive_calls_per_hour = naive_avg_calls / 24
naive_agents = erlang_c_agents(
    calls_per_interval=naive_calls_per_hour,
    aht_seconds=avg_aht,
    interval_seconds=3600,
    target_sl=target_sl,
    target_answer_seconds=20,
)

m1, m2, m3 = st.columns(3)
m1.metric("Avg. forecasted calls/day", f"{avg_forecast_calls:,.0f}")
m2.metric("Avg. handle time (last 30d)", f"{avg_aht:,.0f}s")
m3.metric(f"Agents needed ({method.split()[0]})", agents_needed,
          delta=int(agents_needed - naive_agents) if not method.startswith("Naive") else None,
          delta_color="inverse")

st.caption(
    "Delta shown is agents needed under this method vs. the naive last-week-pattern method — "
    "negative means this method avoids overstaffing relative to naive."
)

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
    hourly_wage = st.number_input("Hourly wage per agent ($)", min_value=1.0, value=22.0, step=1.0)

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

with st.expander("Show raw data"):
    st.dataframe(df.tail(30), width="stretch")
