"""Regenerate the figures and metrics reported in the paper.

Produces, in figures/:
  fig1_actual_vs_predicted_rul.png - actual vs predicted RUL, random split
      (and the separate test cache, if data/test_features_cache.parquet exists)
  fig2_feature_importance.png      - RUL regressor feature importance
  fig3_confusion_matrix.png        - near-failure classifier, random split
  fig4_tungsten_price_forecast.png - tungsten carbide price + Holt-Winters forecast
  fig5_maintenance_timeline.png    - capacity-constrained schedule from results.csv
  metrics.json                     - every number printed in the paper's Table 3

Inputs (project-relative):
  data/features_cache.parquet        (from process_data.py)
  data/test_features_cache.parquet   (optional; enables Fig. 1 panel b)
  prices/tungsten_carbide_price_trend_in_inr.csv  (from fetch_prices.py)
  results/results.csv                (from predict.py; used for Fig. 5)

Models are retrained inside this script with the same hyperparameters
and seed as train.py, so the figures always match the reported metrics.
Fig. 5 reproduces the sample schedule stored in results.csv using the
scheduling rule of main.py without the current-date bound, because the
sample failure dates are historical.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from features import FEATURES, FAILURE_THRESHOLD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data", "features_cache.parquet")
TEST_CACHE_FILE = os.path.join(BASE_DIR, "data", "test_features_cache.parquet")
PRICE_FILE = os.path.join(BASE_DIR, "prices", "tungsten_carbide_price_trend_in_inr.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "results.csv")
OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["Current L1 mean", "Current L3 mean", "Current L2 mean",
          "Vib RMS (+Y)", "Vib RMS (-Z)", "Vib FFT energy",
          "Tool type", "Tool index", "Tool diameter", "Holder length"]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def new_regressor():
    return XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="hist")


def new_classifier():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist")


df = pd.read_parquet(CACHE_FILE)
X, y = df[FEATURES], df["cycle_to_failure"]
y_stage = (y <= FAILURE_THRESHOLD).astype(int)
metrics = {}

# --- Random 80/20 split (headline protocol) ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
reg = new_regressor().fit(X_tr, y_tr)
pred = reg.predict(X_te)
clf = new_classifier().fit(X_tr, y_stage.loc[X_tr.index])
stage_pred = clf.predict(X_te)
y_stage_te = y_stage.loc[X_te.index]
metrics["random"] = {
    "mae": float(mean_absolute_error(y_te, pred)),
    "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
    "acc": float(accuracy_score(y_stage_te, stage_pred)),
    "p": float(precision_score(y_stage_te, stage_pred)),
    "r": float(recall_score(y_stage_te, stage_pred)),
    "f1": float(f1_score(y_stage_te, stage_pred)),
    "cm": confusion_matrix(y_stage_te, stage_pred).tolist(),
    "n_test": int(len(y_te)),
}

# --- Full-cache models (feature importance, separate test cache) ---
reg_full = new_regressor().fit(X, y)
clf_full = new_classifier().fit(X, y_stage)

has_test_cache = os.path.exists(TEST_CACHE_FILE)
if has_test_cache:
    df2 = pd.read_parquet(TEST_CACHE_FILE)
    y2 = df2["cycle_to_failure"]
    pred2 = reg_full.predict(df2[FEATURES])
    stage_pred2 = clf_full.predict(df2[FEATURES])
    y_stage2 = (y2 <= FAILURE_THRESHOLD).astype(int)
    metrics["separate_test_cache"] = {
        "mae": float(mean_absolute_error(y2, pred2)),
        "rmse": float(mean_squared_error(y2, pred2) ** 0.5),
        "acc": float(accuracy_score(y_stage2, stage_pred2)),
        "p": float(precision_score(y_stage2, stage_pred2, zero_division=0)),
        "r": float(recall_score(y_stage2, stage_pred2, zero_division=0)),
        "f1": float(f1_score(y_stage2, stage_pred2, zero_division=0)),
        "n_test": int(len(y2)),
    }

# --- Tool-grouped cross-validation (honest protocol) ---
gkf = GroupKFold(n_splits=5)
g_mae, g_acc = [], []
for tr, te in gkf.split(X, y, groups=df["ToolIndex"]):
    r = new_regressor().fit(X.iloc[tr], y.iloc[tr])
    g_mae.append(mean_absolute_error(y.iloc[te], r.predict(X.iloc[te])))
    c = new_classifier().fit(X.iloc[tr], y_stage.iloc[tr])
    g_acc.append(accuracy_score(y_stage.iloc[te], c.predict(X.iloc[te])))
metrics["grouped"] = {"mae": float(np.mean(g_mae)), "acc": float(np.mean(g_acc)),
                      "mae_folds": [round(float(v), 2) for v in g_mae]}

# --- Fig. 1: actual vs predicted RUL ---
if has_test_cache:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    panels = [(axes[0], y_te, pred, f"(a) Random 80/20 test (n={len(y_te)})\nMAE = {metrics['random']['mae']:.2f} cycles"),
              (axes[1], y2, pred2, f"(b) Separate test cache (n={len(y2)})\nMAE = {metrics['separate_test_cache']['mae']:.2f} cycles")]
else:
    fig, ax = plt.subplots(figsize=(5, 4))
    panels = [(ax, y_te, pred, f"Random 80/20 test (n={len(y_te)})\nMAE = {metrics['random']['mae']:.2f} cycles")]
for ax, actual, predicted, title in panels:
    ax.scatter(actual, predicted, s=18, alpha=0.65)
    lim = max(actual.max(), predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual cycles to failure"); ax.set_title(title, fontsize=10)
panels[0][0].set_ylabel("Predicted cycles to failure")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig1_actual_vs_predicted_rul.png"), dpi=300); plt.close(fig)

# --- Fig. 2: feature importance ---
imp = pd.Series(reg_full.feature_importances_, index=LABELS).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
imp.plot.barh(ax=ax, color="#4C78A8")
ax.set_xlabel("XGBoost feature importance (gain share)")
ax.set_title(f"RUL regressor \u2014 feature importance (trained on {len(df)} cycles)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig2_feature_importance.png"), dpi=300); plt.close(fig)

# --- Fig. 3: confusion matrix ---
cm = np.array(metrics["random"]["cm"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Healthy", "Near failure"]); ax.set_yticklabels(["Healthy", "Near failure"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
ax.set_title(f"Near-failure classifier \u2014 random 80/20 test\nAccuracy = {metrics['random']['acc'] * 100:.1f}%")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig3_confusion_matrix.png"), dpi=300); plt.close(fig)

# --- Fig. 4: tungsten price + Holt-Winters forecast (next 3 months) ---
t = pd.read_csv(PRICE_FILE, parse_dates=["Date"]).set_index("Date")
ts = t["Global_Avg_Price_INR_per_KG"].asfreq("MS")
fit = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
fc = fit.forecast(3)
metrics["price"] = {"current_inr": float(ts.iloc[-1]), "forecast3_avg_inr": float(fc.mean()),
                    "forecast_dates": [str(d.date()) for d in fc.index]}
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(ts.index, ts.values, color="#4C78A8", label="Tungsten carbide price (INR/kg, USD x 83)")
ax.plot(fc.index, fc.values, "o--", color="#E45756", label="Holt-Winters forecast (damped trend, next 3 months)")
ax.set_ylabel("INR per kg")
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(); ax.set_title("Tungsten carbide price trend and short-horizon forecast")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig4_tungsten_price_forecast.png"), dpi=300); plt.close(fig)

# --- Fig. 5: maintenance timeline from results.csv ---
if os.path.exists(RESULTS_FILE):
    res = pd.read_csv(RESULTS_FILE)
    metrics["results_mae"] = float((res["pred_cycles"] - res["true_cycles"]).abs().mean())
    rows, load = [], {}
    for _, r in res.sort_values("fail_date").iterrows():
        fail_date = datetime.datetime.strptime(r["fail_date"], "%Y-%m-%d").date()
        maint_date = fail_date - datetime.timedelta(days=1)
        while load.get(maint_date, 0) >= 2:
            maint_date -= datetime.timedelta(days=1)
        load[maint_date] = load.get(maint_date, 0) + 1
        rows.append((r["file"].replace(".csv", ""), fail_date, maint_date))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (name, fail_date, maint_date) in enumerate(rows):
        ax.barh(i, 1, left=mdates.date2num(maint_date), color="#72B7B2", height=0.6)
        ax.plot(mdates.date2num(fail_date), i, marker="x", color="#E45756", ms=8, mew=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("Date (bar = maintenance date, x = predicted failure date)")
    ax.set_title(f"Capacity-constrained maintenance schedule (max 2 machines/day) \u2014 predict.py sample (n={len(rows)})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig5_maintenance_timeline.png"), dpi=300); plt.close(fig)

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print(json.dumps(metrics, indent=1))
print("Figures written to", OUT_DIR)
