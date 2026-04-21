"""
generate_data.py
Generates 2 years of synthetic weekly demand data for 5 Seattle-area SKU categories.
Event uplift is applied to weeks containing major Seattle events.
Outputs: data/demand.csv, data/events.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent

SKUS = {
    "beverages":   {"base": 500, "trend": 0.8,  "seasonal_amp": 60,  "noise": 30,  "lead_time_mean": 7,  "lead_time_std": 1.5},
    "snacks":      {"base": 320, "trend": 0.5,  "seasonal_amp": 40,  "noise": 20,  "lead_time_mean": 5,  "lead_time_std": 1.0},
    "merchandise": {"base": 150, "trend": 0.3,  "seasonal_amp": 25,  "noise": 15,  "lead_time_mean": 14, "lead_time_std": 3.0},
    "accessories": {"base": 200, "trend": 0.4,  "seasonal_amp": 30,  "noise": 18,  "lead_time_mean": 10, "lead_time_std": 2.0},
    "apparel":     {"base": 120, "trend": 0.2,  "seasonal_amp": 35,  "noise": 12,  "lead_time_mean": 21, "lead_time_std": 4.0},
}

# Seattle event schedule — 2023-01-01 through 2024-12-29
EVENTS = [
    # Seahawks home games (Sept–Jan, ~68k attendance)
    ("2023-09-10", "Seahawks home", 68000, "Lumen Field"),
    ("2023-09-24", "Seahawks home", 68000, "Lumen Field"),
    ("2023-10-01", "Seahawks home", 68000, "Lumen Field"),
    ("2023-10-22", "Seahawks home", 68000, "Lumen Field"),
    ("2023-11-05", "Seahawks home", 68000, "Lumen Field"),
    ("2023-11-23", "Seahawks home", 68000, "Lumen Field"),
    ("2023-12-10", "Seahawks home", 68000, "Lumen Field"),
    ("2023-12-31", "Seahawks home", 68000, "Lumen Field"),
    ("2024-01-07", "Seahawks home", 68000, "Lumen Field"),
    ("2024-09-08", "Seahawks home", 68000, "Lumen Field"),
    ("2024-09-22", "Seahawks home", 68000, "Lumen Field"),
    ("2024-10-06", "Seahawks home", 68000, "Lumen Field"),
    ("2024-10-20", "Seahawks home", 68000, "Lumen Field"),
    ("2024-11-03", "Seahawks home", 68000, "Lumen Field"),
    ("2024-11-17", "Seahawks home", 68000, "Lumen Field"),
    ("2024-12-01", "Seahawks home", 68000, "Lumen Field"),
    ("2024-12-22", "Seahawks home", 68000, "Lumen Field"),
    # Mariners season (Apr–Sept, ~30k attendance, selected games)
    ("2023-04-02", "Mariners opener", 42000, "T-Mobile Park"),
    ("2023-05-27", "Mariners home",   30000, "T-Mobile Park"),
    ("2023-06-10", "Mariners home",   30000, "T-Mobile Park"),
    ("2023-07-01", "Mariners home",   30000, "T-Mobile Park"),
    ("2023-07-15", "All-Star",        43000, "T-Mobile Park"),
    ("2023-08-05", "Mariners home",   30000, "T-Mobile Park"),
    ("2023-09-03", "Mariners home",   30000, "T-Mobile Park"),
    ("2024-03-28", "Mariners opener", 43000, "T-Mobile Park"),
    ("2024-05-18", "Mariners home",   30000, "T-Mobile Park"),
    ("2024-06-22", "Mariners home",   30000, "T-Mobile Park"),
    ("2024-07-06", "Mariners home",   30000, "T-Mobile Park"),
    ("2024-08-17", "Mariners home",   30000, "T-Mobile Park"),
    # PAX West (late Aug, 4-day event ~70k)
    ("2023-09-01", "PAX West",        70000, "Convention Center"),
    ("2024-08-30", "PAX West",        70000, "Convention Center"),
    # Seafair (late July)
    ("2023-07-29", "Seafair",         50000, "Lake Washington"),
    ("2024-07-27", "Seafair",         50000, "Lake Washington"),
    # Kraken home games (Oct–Apr, ~17k)
    ("2023-10-12", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2023-11-18", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2023-12-16", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-01-20", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-02-10", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-03-09", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-10-05", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-11-02", "Kraken home",     17000, "Climate Pledge Arena"),
    ("2024-12-07", "Kraken home",     17000, "Climate Pledge Arena"),
    # Major concerts / Climate Pledge Arena
    ("2023-05-06", "Taylor Swift",    69000, "Lumen Field"),
    ("2023-06-03", "Concert",         17000, "Climate Pledge Arena"),
    ("2023-08-12", "Concert",         17000, "Climate Pledge Arena"),
    ("2024-07-20", "Concert",         17000, "Climate Pledge Arena"),
    ("2024-09-14", "Concert",         17000, "Climate Pledge Arena"),
]


def attendance_bucket(n: int) -> str:
    if n < 15_000:  return "small"
    if n < 40_000:  return "medium"
    return "large"


def event_uplift(attendance: int, sku: str) -> float:
    """Return demand multiplier for a given event attendance and SKU category."""
    bucket = attendance_bucket(attendance)
    base_uplift = {"small": 0.10, "medium": 0.22, "large": 0.38}[bucket]
    # Merchandise and apparel spike harder on event weeks
    sku_modifier = {"beverages": 1.0, "snacks": 1.1, "merchandise": 1.8, "accessories": 1.4, "apparel": 1.6}
    return base_uplift * sku_modifier.get(sku, 1.0)


def generate_weekly_demand(start="2023-01-01", weeks=104) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=weeks, freq="W-MON")
    records = []
    for sku, params in SKUS.items():
        t = np.arange(weeks)
        # Trend
        trend = params["base"] + params["trend"] * t
        # Annual seasonality (52-week period)
        seasonality = params["seasonal_amp"] * np.sin(2 * np.pi * t / 52)
        # Noise
        noise = RNG.normal(0, params["noise"], weeks)
        demand = trend + seasonality + noise
        demand = np.clip(demand, 0, None)
        records.append(pd.DataFrame({"date": dates, "sku": sku, "demand": demand.round(0).astype(int)}))
    return pd.concat(records, ignore_index=True)


def build_event_df() -> pd.DataFrame:
    rows = []
    for date_str, name, attendance, venue in EVENTS:
        rows.append({
            "date": pd.Timestamp(date_str),
            "event_name": name,
            "attendance": attendance,
            "venue": venue,
            "attendance_bucket": attendance_bucket(attendance),
        })
    return pd.DataFrame(rows)


def apply_event_uplift(demand_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """Apply event uplift to weeks that contain an event."""
    df = demand_df.copy()
    # Snap event dates to the nearest Monday week start
    event_weeks = event_df.copy()
    event_weeks["week"] = event_weeks["date"].dt.to_period("W").apply(lambda p: p.start_time)
    # Aggregate max attendance per week (some weeks have multiple events)
    week_max = event_weeks.groupby("week")["attendance"].max().reset_index()
    week_max.columns = ["date", "event_attendance"]
    df = df.merge(week_max, on="date", how="left")
    df["event_attendance"] = df["event_attendance"].fillna(0)
    # Apply uplift
    for sku in SKUS:
        mask = (df["sku"] == sku) & (df["event_attendance"] > 0)
        uplift = df.loc[mask, "event_attendance"].apply(lambda a: event_uplift(a, sku))
        df.loc[mask, "demand"] = (df.loc[mask, "demand"] * (1 + uplift)).round(0).astype(int)
    return df


def main():
    print("Generating synthetic demand data (104 weeks, 5 SKUs)...")
    demand_df = generate_weekly_demand()
    event_df  = build_event_df()
    demand_df = apply_event_uplift(demand_df, event_df)

    demand_df.to_csv(OUT / "demand.csv", index=False)
    event_df.to_csv(OUT / "events.csv", index=False)
    print(f"  Saved demand.csv  — {len(demand_df)} rows ({demand_df['sku'].nunique()} SKUs)")
    print(f"  Saved events.csv  — {len(event_df)} events")
    print(f"  Date range: {demand_df['date'].min().date()} → {demand_df['date'].max().date()}")
    print(f"  Demand range per week: {demand_df['demand'].min()} – {demand_df['demand'].max()} units")


if __name__ == "__main__":
    main()
