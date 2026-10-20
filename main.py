"""Stage 4/5: stock planning, maintenance scheduling and the dashboard.

Run order: process_data.py -> train.py -> fetch_prices.py -> predict.py
-> main.py  (main re-runs predict.py automatically when data/predict/
exists; otherwise it reuses the existing results/results.csv.)

v2 changes (see README changelog):
- Project-relative paths everywhere; predict is re-run with
  sys.executable instead of a bare "python".
- FIX (scheduler bound): maintenance dates are never pushed earlier
  than today. v1 claimed this bound in the paper but did not implement
  it. If capacity is still full on reaching today, the job stays on
  today and a capacity-overflow warning is printed.
- FIX (forecast horizon): v1 requested 4 monthly points starting AT the
  last observed month and labelled the output "next 3 months". v2
  forecasts exactly the next 3 months after the last observation.
- FIX (stock count): v1 added ceil(1000/120) = 9 tools once per RESULT
  ROW (12 rows -> 108 tools) and then based its buy/wait suggestion on
  half of that. The production target is a fleet-level quantity, so v2
  computes the required stock once, and the suggestion uses that total.
- The FRED PPI series, fetched but never used in v1, is now reported
  (latest value + date) in the console output and the dashboard.
"""
import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tkinter as tk
from tkinter import ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "results", "results.csv")
PREDICT_DIR = os.path.join(BASE_DIR, "data", "predict")
PRICES_DIR = os.path.join(BASE_DIR, "prices")

CYCLES_PER_TOOL = {"Carbide End Mill": 120, "Coated Insert": 30, "Tool Holder": 500}
REPRESENTATIVE_TOOL = "Carbide End Mill"  # v1 behaviour: one representative tool type fleet-wide (paper, Section 6)
PRODUCTION_RATE = 1000  # target production volume, in cycles
MAX_PER_DAY = 2         # daily maintenance capacity (machines/day)
BUY_WAIT_BAND = 0.05    # forecast avg within +/-5% of current price -> hold

if os.path.isdir(PREDICT_DIR):
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "predict.py")],
                   check=True, cwd=BASE_DIR)
else:
    print("data/predict/ not found - reusing existing", RESULTS_FILE)

df = pd.read_csv(RESULTS_FILE)
ppi = pd.read_csv(os.path.join(PRICES_DIR, "tool_price_history.csv"))
tungsten = pd.read_csv(os.path.join(PRICES_DIR, "tungsten_carbide_price_trend_in_inr.csv"))

ppi["Date"] = pd.to_datetime(ppi["DATE"])
ppi_latest_date = ppi["Date"].max()
ppi_latest = float(ppi.loc[ppi["Date"] == ppi_latest_date, "WPU1135"].iloc[-1])

tungsten["Date"] = pd.to_datetime(tungsten["Date"])
tungsten = tungsten.set_index("Date")
ts = tungsten["Global_Avg_Price_INR_per_KG"].asfreq("MS")
model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None)
fit = model.fit()
forecast_series = fit.forecast(3)  # the next 3 months after the last observation

current_price_inr = float(ts.iloc[-1])
avg_future_price = float(forecast_series.mean())

# Fleet-level stock for the production target (computed once - see changelog).
total_tools_needed = int(np.ceil(PRODUCTION_RATE / CYCLES_PER_TOOL[REPRESENTATIVE_TOOL]))

if avg_future_price > current_price_inr * (1 + BUY_WAIT_BAND):
    suggestion = f"Buy {total_tools_needed} tools now (prices rising)"
elif avg_future_price < current_price_inr * (1 - BUY_WAIT_BAND):
    suggestion = f"Wait, prices dropping. Plan to buy {total_tools_needed} tools later."
else:
    suggestion = f"Hold steady. Keep {total_tools_needed} tools in reserve."

today = datetime.date.today()
schedule_rows = []
day_load = {}
overflow = []
for _, row in df.sort_values(by="fail_date").iterrows():
    machine_name = row["file"].replace(".csv", "")
    fail_date = datetime.datetime.strptime(row["fail_date"], "%Y-%m-%d").date()
    maintenance_date = fail_date - datetime.timedelta(days=1)
    if maintenance_date < today:
        maintenance_date = today
    while day_load.get(maintenance_date, 0) >= MAX_PER_DAY and maintenance_date > today:
        maintenance_date -= datetime.timedelta(days=1)
    if day_load.get(maintenance_date, 0) >= MAX_PER_DAY:
        overflow.append(machine_name)  # capacity full even today: keep today, warn
    day_load[maintenance_date] = day_load.get(maintenance_date, 0) + 1
    schedule_rows.append({
        "Machine": machine_name,
        "Failure Date": fail_date,
        "Maintenance Date": maintenance_date,
    })
if overflow:
    print(f"WARNING: daily capacity ({MAX_PER_DAY}) exceeded on {today} for: {', '.join(overflow)}")

print("\n=== Forecasting Results ===")
for _, row in df.iterrows():
    print(row["file"], row["pred_cycles"], row["true_cycles"],
          row["pred_cycles"] - row["true_cycles"], row["days_left"],
          row["fail_date"], row["risk_level"])
print("\n=== Global Stock Planning ===")
print(f"Cutting Tools PPI (FRED WPU1135, {ppi_latest_date.date()}): {ppi_latest}")
print(f"Current Tungsten Price (INR/kg): {round(current_price_inr, 2)}")
print(f"Forecasted Avg Price (Next 3 Months): {round(avg_future_price, 2)}")
print(f"Total Tools Needed: {total_tools_needed}")
print(f"Suggestion: {suggestion}")
print("\n=== Maintenance Schedule ===")
for r in schedule_rows:
    print(r["Machine"], r["Failure Date"], r["Maintenance Date"])

# ---------------- Dashboard (unchanged views, plus the PPI line) ----------------
root = tk.Tk()
root.title("Tool Stock & Maintenance Planner")

frame0 = ttk.LabelFrame(root, text="Forecasting Results")
frame0.pack(fill="x", padx=10, pady=5)
cols = ("Machine", "Predicted", "Actual", "Deviation", "DaysLeft", "FailDate", "RiskLevel")
tree0 = ttk.Treeview(frame0, columns=cols, show="headings")
for col in cols:
    tree0.heading(col, text=col)
tree0.pack(fill="x")
for _, row in df.iterrows():
    tree0.insert("", "end", values=(
        row["file"].replace(".csv", ""), row["pred_cycles"], row["true_cycles"],
        row["pred_cycles"] - row["true_cycles"], row["days_left"],
        row["fail_date"], row["risk_level"]))

frame1 = ttk.LabelFrame(root, text="Stock Forecast")
frame1.pack(fill="x", padx=10, pady=5)
ttk.Label(frame1, text=f"Cutting Tools PPI (FRED WPU1135, {ppi_latest_date.date()}): {ppi_latest}").pack(anchor="w")
ttk.Label(frame1, text=f"Current Tungsten Price (INR/kg): {round(current_price_inr, 2)}").pack(anchor="w")
ttk.Label(frame1, text=f"Forecasted Avg Price (Next 3 Months): {round(avg_future_price, 2)}").pack(anchor="w")
ttk.Label(frame1, text=f"Tools Stock to Maintain: {total_tools_needed}").pack(anchor="w")
ttk.Label(frame1, text=f"Suggestion: {suggestion}").pack(anchor="w")

frame3 = ttk.LabelFrame(root, text="Maintenance Timeline")
frame3.pack(fill="both", expand=True, padx=10, pady=5)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig, ax = plt.subplots(figsize=(7, 4))
for r in schedule_rows:
    ax.barh(r["Machine"], 1, left=r["Maintenance Date"], color="skyblue")
ax.set_xlabel("Date")
ax.set_ylabel("Machine")
ax.set_title("Maintenance Timeline")
ax.xaxis_date()
fig.autofmt_xdate()
canvas = FigureCanvasTkAgg(fig, master=frame3)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
