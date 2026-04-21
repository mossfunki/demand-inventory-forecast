"""
forecast.py
Trains Prophet (primary) and XGBoost (comparison) demand forecasting models.
Evaluates on an 8-week held-out test set, produces 14-day rolling forecasts.
Outputs: outputs/forecasts.csv, outputs/eval_metrics.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("prophet not installed — using statsmodels STL + linear extrapolation as fallback")
    from statsmodels.tsa.seasonal import STL

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_WEEKS = 8


def load_data():
    demand = pd.read_csv(DATA_DIR / "demand.csv", parse_dates=["date"])
    events = pd.read_csv(DATA_DIR / "events.csv", parse_dates=["date"])
    return demand, events


def make_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Encode event attendance as weekly features."""
    ev = events.copy()
    ev["week"] = ev["date"].dt.to_period("W").apply(lambda p: p.start_time)
    week_att  = ev.groupby("week")["attendance"].max().reset_index()
    week_att.columns = ["date", "event_attendance"]

    df = df.merge(week_att, on="date", how="left")
    df["event_attendance"] = df["event_attendance"].fillna(0)
    df["is_large_event"]  = (df["event_attendance"] >= 40_000).astype(int)
    df["is_medium_event"] = ((df["event_attendance"] >= 15_000) & (df["event_attendance"] < 40_000)).astype(int)
    df["is_small_event"]  = ((df["event_attendance"] > 0)       & (df["event_attendance"] < 15_000)).astype(int)
    df["week_of_year"]    = df["date"].dt.isocalendar().week.astype(int)
    df["month"]           = df["date"].dt.month
    return df


def train_xgboost(train: pd.DataFrame) -> GradientBoostingRegressor:
    features = ["week_of_year", "month", "is_large_event", "is_medium_event", "is_small_event", "event_attendance"]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(train[features], train["demand"])
    return model


def predict_xgboost(model, df: pd.DataFrame) -> np.ndarray:
    features = ["week_of_year", "month", "is_large_event", "is_medium_event", "is_small_event", "event_attendance"]
    return model.predict(df[features])


def forecast_prophet(train: pd.DataFrame, test: pd.DataFrame, future_weeks: int = 14):
    """Train Prophet on train split, forecast test + future_weeks ahead."""
    prophet_df = train[["date", "demand"]].rename(columns={"date": "ds", "demand": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.95,
    )
    # Add event regressors
    for col in ["is_large_event", "is_medium_event", "is_small_event"]:
        m.add_regressor(col)

    # Merge regressors into prophet df
    reg_cols = train[["date", "is_large_event", "is_medium_event", "is_small_event"]].rename(columns={"date": "ds"})
    prophet_df = prophet_df.merge(reg_cols, on="ds", how="left")

    m.fit(prophet_df)

    # Future dataframe
    last_date  = train["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta("7D"), periods=TEST_WEEKS + future_weeks, freq="W-MON")
    future = pd.DataFrame({"ds": future_dates})

    # Fill regressors for future dates (from test + zero-padded)
    test_reg = test[["date", "is_large_event", "is_medium_event", "is_small_event"]].rename(columns={"date": "ds"})
    future = future.merge(test_reg, on="ds", how="left").fillna(0)

    forecast = m.predict(future)
    return m, forecast


def fallback_forecast(train: pd.DataFrame, test: pd.DataFrame, future_weeks: int = 14):
    """STL decomposition + linear trend extrapolation — fallback if Prophet not available."""
    all_preds = []
    total_periods = len(test) + future_weeks
    t_train = np.arange(len(train))
    t_future = np.arange(len(train), len(train) + total_periods)

    stl = STL(train["demand"].values, period=52, robust=True)
    res = stl.fit()
    trend_coeffs = np.polyfit(t_train, res.trend, deg=1)
    trend_future = np.polyval(trend_coeffs, t_future)

    week_of_year = (train["date"].dt.isocalendar().week.astype(int) - 1)
    seasonal_by_week = pd.Series(res.seasonal).groupby(week_of_year.values).mean()

    future_woy = pd.date_range(
        start=train["date"].max() + pd.Timedelta("7D"), periods=total_periods, freq="W-MON"
    ).isocalendar().week.astype(int) - 1
    seasonal_future = np.array([seasonal_by_week.get(w, 0) for w in future_woy])

    yhat = trend_future + seasonal_future
    dates = pd.date_range(
        start=train["date"].max() + pd.Timedelta("7D"), periods=total_periods, freq="W-MON"
    )
    return pd.DataFrame({"ds": dates, "yhat": yhat, "yhat_lower": yhat * 0.88, "yhat_upper": yhat * 1.12})


def run(sku: str, demand: pd.DataFrame, events: pd.DataFrame):
    df = demand[demand["sku"] == sku].sort_values("date").reset_index(drop=True)
    df = make_event_features(df, events)

    split = len(df) - TEST_WEEKS
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # --- XGBoost ---
    xgb_model = train_xgboost(train)
    test["xgb_pred"]  = predict_xgboost(xgb_model, test)
    train["xgb_pred"] = predict_xgboost(xgb_model, train)

    xgb_mape  = mean_absolute_percentage_error(test["demand"], test["xgb_pred"])

    # --- Prophet or fallback ---
    if PROPHET_AVAILABLE:
        _, prophet_fc = forecast_prophet(train, test, future_weeks=14)
    else:
        prophet_fc = fallback_forecast(train, test, future_weeks=14)

    test_fc = prophet_fc[prophet_fc["ds"].isin(test["date"])]
    if len(test_fc):
        prophet_mape = mean_absolute_percentage_error(
            test["demand"].values,
            test_fc["yhat"].clip(0).values[:len(test)]
        )
    else:
        prophet_mape = float("nan")

    # --- Build full forecast output ---
    # Historical actuals + test predictions + 14-week future
    future_fc = prophet_fc[~prophet_fc["ds"].isin(test["date"])].copy()
    future_fc["sku"]    = sku
    future_fc["split"]  = "future"
    future_fc["actual"] = np.nan

    test_out = test[["date", "demand"]].rename(columns={"date": "ds", "demand": "actual"}).copy()
    test_out = test_out.merge(prophet_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
    test_out["sku"]   = sku
    test_out["split"] = "test"

    forecast_df = pd.concat([test_out, future_fc[["ds", "sku", "split", "actual", "yhat", "yhat_lower", "yhat_upper"]]], ignore_index=True)

    metrics = {"sku": sku, "prophet_mape": round(prophet_mape, 4), "xgb_mape": round(xgb_mape, 4), "test_weeks": TEST_WEEKS}
    print(f"  [{sku:12s}]  Prophet MAPE: {prophet_mape:.2%}   XGBoost MAPE: {xgb_mape:.2%}")
    return forecast_df, metrics


def main():
    print("Loading data...")
    demand, events = load_data()

    all_forecasts = []
    all_metrics   = []

    for sku in demand["sku"].unique():
        fcast, metrics = run(sku, demand, events)
        all_forecasts.append(fcast)
        all_metrics.append(metrics)

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    metrics_df   = pd.DataFrame(all_metrics)

    forecasts_df.to_csv(OUTPUT_DIR / "forecasts.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "eval_metrics.csv", index=False)

    print(f"\nForecasts saved  → outputs/forecasts.csv")
    print(f"Metrics saved    → outputs/eval_metrics.csv")
    print(f"\nAverage Prophet MAPE across SKUs: {metrics_df['prophet_mape'].mean():.2%}")
    print(f"Average XGBoost MAPE across SKUs: {metrics_df['xgb_mape'].mean():.2%}")


if __name__ == "__main__":
    main()
