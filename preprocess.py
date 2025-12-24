"""Preprocess raw Fingrid CSVs: detect numeric columns, normalize timestamps, resample to hourly, and save cleaned files."""
from pathlib import Path
import pandas as pd


raw_dir = Path("Rawdata")
data_dir = Path("DATA")
data_dir.mkdir(exist_ok=True)


def pick_path(*candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def load_numeric_time_series(path: Path, value_name: str) -> pd.DataFrame:
    """Load a CSV, auto-detect delimiter, parse startTime UTC, keep first numeric column."""
    df = pd.read_csv(path, sep=None, engine="python")
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    value_col = None
    for col in df.columns:
        if col in ("startTime", "endTime"):
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        if as_num.notna().any():
            df[col] = as_num
            value_col = col
            break
    if value_col is None:
        raise ValueError(f"No numeric value column found in {path}")
    return df[["startTime", value_col]].rename(columns={value_col: value_name})


# Locate raw files (handle en dash vs hyphen and fallback to project/DATA roots)
consumption_path = pick_path(
    raw_dir / "Electricity consumption forecast – updated every 15 minutes.csv",
    raw_dir / "Electricity consumption forecast - updated every 15 minutes.csv",
    Path("Electricity consumption forecast – updated every 15 minutes.csv"),
    Path("Electricity consumption forecast - updated every 15 minutes.csv"),
    data_dir / "Electricity consumption forecast – updated every 15 minutes.csv",
    data_dir / "Electricity consumption forecast - updated every 15 minutes.csv",
)
price_path = pick_path(
    raw_dir / "Imbalance price (15 min).csv",
    Path("Imbalance price (15 min).csv"),
    data_dir / "Imbalance price (15 min).csv",
)
wind_path = pick_path(
    raw_dir / "Wind power generation forecast - updated once a day.csv",
    Path("Wind power generation forecast - updated once a day.csv"),
    data_dir / "Wind power generation forecast - updated once a day.csv",
    data_dir / "wind_hourly.csv",  # fallback if only hourly wind exists
)

# Load raw data
cons_15 = load_numeric_time_series(consumption_path, "consumption_15min")
price_15 = load_numeric_time_series(price_path, "price_15min_eur_per_mwh")
wind_daily = load_numeric_time_series(wind_path, "wind_daily")

# Resample to hourly
cons_hourly = (
    cons_15.set_index("startTime")
    .resample("H")
    .mean()
    .reset_index()
    .rename(columns={"consumption_15min": "consumption_forecast"})
)
price_hourly = (
    price_15.set_index("startTime")
    .resample("H")
    .mean()
    .reset_index()
    .rename(columns={"price_15min_eur_per_mwh": "price_eur_per_mwh"})
)
wind_hourly = (
    wind_daily.set_index("startTime")
    .resample("H")
    .ffill()
    .reset_index()
    .rename(columns={"wind_daily": "wind_forecast"})
)

# Save hourly cleaned files
cons_path_out = data_dir / "consumption_hourly.csv"
price_path_out = data_dir / "price_hourly.csv"
wind_path_out = data_dir / "wind_hourly.csv"

cons_hourly.to_csv(cons_path_out, index=False)
price_hourly.to_csv(price_path_out, index=False)
wind_hourly.to_csv(wind_path_out, index=False)

# Summary
print(f"Saved hourly files:\n- {cons_path_out} ({len(cons_hourly)} rows)\n- {price_path_out} ({len(price_hourly)} rows)\n- {wind_path_out} ({len(wind_hourly)} rows)")
