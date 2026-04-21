"""
inventory.py
Calculates dynamic safety stock and reorder points using forecast demand.
Compares dynamic vs. static ROP model on the held-out test period.
Outputs: outputs/inventory_plan.csv, outputs/model_comparison.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SERVICE_LEVEL = 0.95          # 95% service level
Z = stats.norm.ppf(SERVICE_LEVEL)   # ≈ 1.645

SKU_PARAMS = {
    "beverages":   {"lead_time_mean": 7,  "lead_time_std": 1.5, "holding_cost_pct": 0.20, "stockout_cost": 8.00},
    "snacks":      {"lead_time_mean": 5,  "lead_time_std": 1.0, "holding_cost_pct": 0.18, "stockout_cost": 5.00},
    "merchandise": {"lead_time_mean": 14, "lead_time_std": 3.0, "holding_cost_pct": 0.25, "stockout_cost": 15.00},
    "accessories": {"lead_time_mean": 10, "lead_time_std": 2.0, "holding_cost_pct": 0.22, "stockout_cost": 12.00},
    "apparel":     {"lead_time_mean": 21, "lead_time_std": 4.0, "holding_cost_pct": 0.28, "stockout_cost": 20.00},
}


def daily_demand_stats(weekly_demand: pd.Series, lead_time_mean: float):
    """Convert weekly demand series to daily stats over the lead-time window."""
    daily = weekly_demand / 7
    mu_d  = daily.mean()
    std_d = daily.std()
    # Demand variability over lead time
    sigma_lt = np.sqrt(lead_time_mean * std_d**2)
    return mu_d, sigma_lt


def safety_stock(sigma_lt: float) -> float:
    """SS = Z × σ_LT — standard safety stock formula at target service level."""
    return Z * sigma_lt


def reorder_point(mu_daily: float, lead_time_mean: float, ss: float) -> float:
    """ROP = (avg_daily_demand × lead_time) + safety_stock"""
    return (mu_daily * lead_time_mean) + ss


def economic_order_quantity(annual_demand: float, order_cost: float, unit_cost: float, holding_pct: float) -> float:
    """EOQ = sqrt(2DS / iC)"""
    h = unit_cost * holding_pct
    if h <= 0:
        return annual_demand / 4
    return np.sqrt((2 * annual_demand * order_cost) / h)


def simulate_stockouts(demand_series: pd.Series, rop: float, lead_time: float, initial_stock: float = None) -> int:
    """Simulate how many weeks end with a stockout under a given ROP."""
    stock = initial_stock if initial_stock else rop * 2
    order_pending = 0
    order_due_in  = 0
    stockouts = 0
    for d in demand_series:
        stock -= d
        if stock <= 0:
            stockouts += 1
            stock = 0
        # Place order when stock hits ROP
        if stock <= rop and order_pending == 0:
            order_pending = rop * 1.5  # order up to 1.5× ROP
            order_due_in  = int(np.ceil(lead_time / 7))
        if order_due_in > 0:
            order_due_in -= 1
            if order_due_in == 0:
                stock += order_pending
                order_pending = 0
    return stockouts


def run(demand_df: pd.DataFrame, forecasts_df: pd.DataFrame):
    rows = []
    comparison_rows = []

    for sku, params in SKU_PARAMS.items():
        sku_demand = demand_df[demand_df["sku"] == sku].sort_values("date")
        sku_fc     = forecasts_df[forecasts_df["sku"] == sku].copy()

        lt_mean = params["lead_time_mean"]
        lt_std  = params["lead_time_std"]

        # --- STATIC MODEL: ROP based on historical average ---
        mu_d_hist, sigma_lt_hist = daily_demand_stats(sku_demand["demand"], lt_mean)
        ss_static  = safety_stock(sigma_lt_hist)
        rop_static = reorder_point(mu_d_hist, lt_mean, ss_static)

        # --- DYNAMIC MODEL: ROP updated weekly using rolling 8-week forecast ---
        future_weeks = sku_fc[sku_fc["split"].isin(["test", "future"])].copy()
        future_weeks["yhat_daily"] = future_weeks["yhat"].clip(0) / 7

        # Look-ahead window = lead time in weeks
        lt_weeks = int(np.ceil(lt_mean / 7))

        dynamic_rows = []
        for i, row in future_weeks.iterrows():
            # Use the next lt_weeks of forecast as the demand distribution
            lookahead = future_weeks["yhat"].iloc[max(0, i - future_weeks.index[0]):
                                                   i - future_weeks.index[0] + lt_weeks].clip(0)
            if len(lookahead) < 2:
                lookahead = future_weeks["yhat"].clip(0)
            mu_d_fc, sigma_lt_fc = daily_demand_stats(lookahead, lt_mean)
            ss_dyn   = safety_stock(sigma_lt_fc)
            rop_dyn  = reorder_point(mu_d_fc, lt_mean, ss_dyn)
            dynamic_rows.append({
                "date": row["ds"],
                "sku": sku,
                "forecast_demand": round(row["yhat"], 1) if not np.isnan(row["yhat"]) else None,
                "forecast_lower":  round(row["yhat_lower"], 1) if "yhat_lower" in row else None,
                "forecast_upper":  round(row["yhat_upper"], 1) if "yhat_upper" in row else None,
                "safety_stock":    round(ss_dyn, 1),
                "reorder_point":   round(rop_dyn, 1),
                "lead_time_days":  lt_mean,
                "service_level":   f"{SERVICE_LEVEL:.0%}",
                "split":           row["split"],
            })
            rows.extend(dynamic_rows[-1:])

        # --- STOCKOUT SIMULATION: static vs dynamic ---
        test_demand = sku_demand.tail(8)["demand"]
        so_static  = simulate_stockouts(test_demand, rop_static, lt_mean)

        test_fc_rop = [r["reorder_point"] for r in dynamic_rows if r["split"] == "test"]
        dyn_rop_mean = np.mean(test_fc_rop) if test_fc_rop else rop_static
        so_dynamic  = simulate_stockouts(test_demand, dyn_rop_mean, lt_mean)

        comparison_rows.append({
            "sku":                      sku,
            "static_rop":              round(rop_static, 1),
            "dynamic_rop_mean_test":   round(dyn_rop_mean, 1),
            "static_stockout_weeks":   so_static,
            "dynamic_stockout_weeks":  so_dynamic,
            "stockout_reduction_pct":  round((so_static - so_dynamic) / max(so_static, 1) * 100, 1),
            "safety_stock_static":     round(ss_static, 1),
            "lead_time_days":          lt_mean,
        })

        print(f"  [{sku:12s}]  Static ROP: {rop_static:.0f}   Dynamic ROP (mean): {dyn_rop_mean:.0f}   "
              f"Stockouts: {so_static} → {so_dynamic}  (−{comparison_rows[-1]['stockout_reduction_pct']}%)")

    plan_df = pd.DataFrame(rows)
    comp_df = pd.DataFrame(comparison_rows)

    plan_df.to_csv(OUTPUT_DIR / "inventory_plan.csv", index=False)
    comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print(f"\nInventory plan  → outputs/inventory_plan.csv")
    print(f"Model comparison → outputs/model_comparison.csv")
    avg_reduction = comp_df["stockout_reduction_pct"].mean()
    print(f"\nAverage stockout reduction (dynamic vs. static): {avg_reduction:.1f}%")
    return plan_df, comp_df


def main():
    demand_df   = pd.read_csv(DATA_DIR / "demand.csv", parse_dates=["date"])
    forecasts_df = pd.read_csv(OUTPUT_DIR / "forecasts.csv", parse_dates=["ds"])
    print("Running inventory optimization...")
    run(demand_df, forecasts_df)


if __name__ == "__main__":
    main()
