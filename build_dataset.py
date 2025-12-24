"""Merge hourly datasets and add features/labels for modeling."""
from pathlib import Path
import pandas as pd


data_dir = Path("DATA")
data_dir.mkdir(exist_ok=True)

# Input paths
cons_path = data_dir / "consumption_hourly.csv"
price_path = data_dir / "price_hourly.csv"
wind_path = data_dir / "wind_hourly.csv"

# Load hourly data
cons = pd.read_csv(cons_path)
price = pd.read_csv(price_path)
wind = pd.read_csv(wind_path)

# Parse timestamps
for df in (cons, price, wind):
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)

# Merge on timestamp
merged_raw = (
    cons.merge(price, on="startTime", how="inner")
    .merge(wind, on="startTime", how="inner")
    .sort_values("startTime")
    .reset_index(drop=True)
)

# Save merged (raw) dataset
final_merged_path = data_dir / "final_merged_hourly.csv"
merged_raw.to_csv(final_merged_path, index=False)

# Add derived targets and time features
ml_df = merged_raw.copy()
ml_df["price_cents_per_kwh"] = ml_df["price_eur_per_mwh"] / 10
ml_df["is_expensive"] = (ml_df["price_cents_per_kwh"] > 10).astype(int)
ml_df["hour"] = ml_df["startTime"].dt.hour
ml_df["day_of_week"] = ml_df["startTime"].dt.dayofweek
ml_df["is_weekend"] = ml_df["day_of_week"].isin([5, 6]).astype(int)

# Save ML-ready dataset
final_ml_path = data_dir / "final_ml_dataset.csv"
ml_df.to_csv(final_ml_path, index=False)

# Summary
print(f"Saved merged hourly dataset: {final_merged_path} ({len(merged_raw)} rows)")
print(f"Saved ML dataset: {final_ml_path} ({len(ml_df)} rows)")
