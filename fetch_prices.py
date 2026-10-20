"""Fetch the cutting-tool PPI series from FRED and convert the tungsten
carbide series to INR.

v2 changes: project-relative prices/ folder; the static USD->INR rate is
now a named, documented constant (the paper lists it as a limitation).
"""
import os
import datetime
import pandas as pd
from pandas_datareader import data as pdr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICES_DIR = os.path.join(BASE_DIR, "prices")
USD_TO_INR = 83.0  # static rate used throughout the project (see paper, Section 6)

os.makedirs(PRICES_DIR, exist_ok=True)

series_code_tools = "WPU1135"  # FRED: PPI, metal cutting tools
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime.today()

print("Fetching Cutting Tools PPI from FRED...")
df_tools = pdr.DataReader(series_code_tools, "fred", start, end)
df_tools.to_csv(os.path.join(PRICES_DIR, "tool_price_history.csv"))
print("Saved tool price history")
print(df_tools.tail())

input_file = os.path.join(PRICES_DIR, "tungsten_carbide_price_trend_2015_2026.csv")
output_file = os.path.join(PRICES_DIR, "tungsten_carbide_price_trend_in_inr.csv")
try:
    df_tungsten = pd.read_csv(input_file)
    df_tungsten["Global_Avg_Price_INR_per_KG"] = df_tungsten["Estimated_Global_Avg_Price_USD_per_KG"] * USD_TO_INR
    df_tungsten["China_Market_Price_INR_per_KG"] = df_tungsten["Estimated_China_Market_Price_USD_per_KG"] * USD_TO_INR
    df_tungsten.to_csv(output_file, index=False)
    print(f"Converted tungsten prices saved to {output_file}")
    print(df_tungsten.tail())
except FileNotFoundError:
    print("tungsten_carbide_price_trend_2015_2026.csv not found in prices/. Please place your dataset there.")
