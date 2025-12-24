import pandas as pd

# Load 15-minute consumption forecast (semicolon-delimited file)
cons = pd.read_csv(
    "Electricity consumption forecast – updated every 15 minutes.csv",
    sep=";"
)
cons["startTime"] = pd.to_datetime(cons["startTime"])  # convert to datetime
cons_hourly = (
    cons.set_index("startTime")
        .resample("h")["Electricity consumption forecast - updated every 15 minutes"].mean()  # hourly average
        .reset_index()
        .rename(columns={"Electricity consumption forecast - updated every 15 minutes": "consumption_forecast"})
)
cons_hourly.to_csv("Electricity consumption forecast – hourly.csv", index=False)

# Load 15-minute imbalance price (semicolon-delimited file)
price = pd.read_csv(
    "Imbalance price (15 min).csv",
    sep=";"
)
price["startTime"] = pd.to_datetime(price["startTime"])  # convert to datetime
price_hourly = (
    price.set_index("startTime")
         .resample("h")["Imbalance price"].mean()        # hourly average
         .reset_index()
         .rename(columns={"Imbalance price": "price_eur_per_mwh"})
)
price_hourly.to_csv("Imbalance price - hourly.csv", index=False)
