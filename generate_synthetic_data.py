"""Generate synthetic contact center call volume + AHT data.

Half-hourly granularity, 3 years of history, one queue:
- intraday shape: two peaks (mid-morning, early afternoon), low overnight
  floor (some after-hours volume, not zero — most centers aren't fully closed)
- weekly seasonality (weekday vs weekend, via a per-day multiplier)
- slow upward trend across the 3 years
- yearly seasonality (holiday season bump)
- a handful of injected spike days (outage / promo / holiday rush) that
  scale every interval in that day up
- Gaussian noise at the interval level
- AHT (average handle time, seconds): roughly flat with noise, dips
  slightly when load is high (agents rush when queues build)

Also writes a daily aggregate (data/calls.csv) for the Streamlit dashboard,
which only needs day-level numbers.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

START = "2023-01-01"
NUM_DAYS = 1095  # 3 years
INTERVALS_PER_DAY = 48  # 30-minute intervals

WEEKDAY_MULTIPLIER = {
    0: 1.15,  # Mon
    1: 1.05,  # Tue
    2: 1.00,  # Wed
    3: 1.00,  # Thu
    4: 1.10,  # Fri
    5: 0.55,  # Sat
    6: 0.40,  # Sun
}

DAILY_BASE_VOLUME = 480
TREND_PER_DAY = 0.15
YEARLY_AMPLITUDE = 60
DAILY_NOISE_STD = 18
INTERVAL_NOISE_STD = 3.5

BASE_AHT = 300  # seconds
AHT_NOISE_STD = 8

SPIKE_DAY_OFFSETS = [45, 120, 260, 400, 540, 610, 680, 780, 860, 950, 1030]
SPIKE_MULTIPLIER = 1.9

OVERNIGHT_FLOOR = 0.05
MORNING_PEAK_HOUR = 10.5
MORNING_PEAK_WIDTH = 2.0
AFTERNOON_PEAK_HOUR = 14.5
AFTERNOON_PEAK_WIDTH = 2.3
AFTERNOON_PEAK_RELATIVE_HEIGHT = 0.85


def intraday_shape() -> np.ndarray:
    """Relative call weight for each half-hour interval of a day, summing to 1."""
    hours = np.arange(INTERVALS_PER_DAY) * 0.5
    morning = np.exp(-0.5 * ((hours - MORNING_PEAK_HOUR) / MORNING_PEAK_WIDTH) ** 2)
    afternoon = AFTERNOON_PEAK_RELATIVE_HEIGHT * np.exp(
        -0.5 * ((hours - AFTERNOON_PEAK_HOUR) / AFTERNOON_PEAK_WIDTH) ** 2
    )
    weights = OVERNIGHT_FLOOR + morning + afternoon
    return weights / weights.sum()


def generate_daily_totals() -> pd.DataFrame:
    dates = pd.date_range(START, periods=NUM_DAYS, freq="D")
    day_idx = np.arange(NUM_DAYS)

    weekday_mult = np.array([WEEKDAY_MULTIPLIER[d.weekday()] for d in dates])
    trend = TREND_PER_DAY * day_idx
    yearly = YEARLY_AMPLITUDE * np.sin(2 * np.pi * day_idx / 365.25 + np.pi / 2)
    noise = RNG.normal(0, DAILY_NOISE_STD, NUM_DAYS)

    daily_total = (DAILY_BASE_VOLUME + trend + yearly) * weekday_mult + noise

    spike_mask = np.zeros(NUM_DAYS, dtype=bool)
    spike_mask[[o for o in SPIKE_DAY_OFFSETS if o < NUM_DAYS]] = True
    daily_total = np.where(spike_mask, daily_total * SPIKE_MULTIPLIER, daily_total)
    daily_total = np.clip(daily_total, 20, None)

    return pd.DataFrame({"date": dates, "daily_total": daily_total, "is_spike_day": spike_mask})


def generate_intraday(daily: pd.DataFrame) -> pd.DataFrame:
    shape = intraday_shape()
    rows = []
    load_reference = daily["daily_total"].mean() / INTERVALS_PER_DAY

    for _, day_row in daily.iterrows():
        base_calls = day_row["daily_total"] * shape
        noisy_calls = base_calls + RNG.normal(0, INTERVAL_NOISE_STD, INTERVALS_PER_DAY)
        noisy_calls = np.clip(noisy_calls, 0, None).round().astype(int)

        load_factor = noisy_calls / load_reference
        aht = BASE_AHT - 15 * np.clip(load_factor - 1, 0, None) + RNG.normal(
            0, AHT_NOISE_STD, INTERVALS_PER_DAY
        )
        aht = np.clip(aht, 120, None).round().astype(int)

        timestamps = pd.date_range(day_row["date"], periods=INTERVALS_PER_DAY, freq="30min")
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "calls": noisy_calls,
                    "aht_seconds": aht,
                    "is_spike_day": day_row["is_spike_day"],
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def aggregate_daily(intraday: pd.DataFrame) -> pd.DataFrame:
    intraday = intraday.copy()
    intraday["date"] = intraday["timestamp"].dt.floor("D")
    grouped = intraday.groupby("date").apply(
        lambda g: pd.Series(
            {
                "calls": g["calls"].sum(),
                "aht_seconds": round(np.average(g["aht_seconds"], weights=g["calls"].clip(lower=1))),
                "is_spike_day": bool(g["is_spike_day"].iloc[0]),
            }
        ),
        include_groups=False,
    )
    return grouped.reset_index()


if __name__ == "__main__":
    daily_totals = generate_daily_totals()
    intraday_df = generate_intraday(daily_totals)
    intraday_df.to_csv("data/calls_intraday.csv", index=False)
    print(f"wrote data/calls_intraday.csv ({len(intraday_df)} rows)")

    daily_df = aggregate_daily(intraday_df)
    daily_df.to_csv("data/calls.csv", index=False)
    print(f"wrote data/calls.csv ({len(daily_df)} rows)")
