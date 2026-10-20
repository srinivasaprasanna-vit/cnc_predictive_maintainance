"""Stage 3: inference on new per-cycle logs.

v2 changes (see README changelog):
- Project-relative paths: logs in data/predict/, metadata.xlsx alongside
  them, models read from models/.
- FIX: the near-failure classifier (stage_model.json) is now actually
  loaded and used. v1 trained it but never ran it at inference.
  results/results.csv gains two columns: near_failure (0/1) and
  near_failure_prob (classifier probability).
- Feature extraction comes from features.py (shared with training).
"""
import os
import datetime
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

from features import FEATURES, extract_signal_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICT_DIR = os.path.join(BASE_DIR, "data", "predict")
META_FILE = os.path.join(PREDICT_DIR, "metadata.xlsx")
MODEL_FILE_RUL = os.path.join(BASE_DIR, "models", "rul_model.json")
MODEL_FILE_STAGE = os.path.join(BASE_DIR, "models", "stage_model.json")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "results.csv")

CYCLES_PER_DAY = 4
CRITICAL_MAX = 10   # predicted cycles <= 10  -> Critical
WARNING_MAX = 30    # predicted cycles <= 30  -> Warning, else Safe

rul_model = XGBRegressor()
rul_model.load_model(MODEL_FILE_RUL)
stage_model = XGBClassifier()
stage_model.load_model(MODEL_FILE_STAGE)
df_meta = pd.read_excel(META_FILE, index_col=0)


def extract_features(fname):
    df_raw = pd.read_csv(os.path.join(PREDICT_DIR, fname))
    feats = extract_signal_features(df_raw)

    base_name = fname.replace(".csv", "")
    meta_row = df_meta.loc[df_meta.index == base_name]
    if meta_row.empty:
        print(f"[!] Metadata missing for {fname}, skipping.")
        return None
    meta_row = meta_row.iloc[0]
    feats.update({
        "MillingToolType": meta_row["MillingToolType"],
        "ToolIndex": meta_row["ToolIndex"],
        "ToolDiameter": meta_row["ToolDiameter [mm]"],
        "ToolHolderLength": meta_row["ToolHolderLength [mm]"],
        "cycle_to_failure": meta_row["CycleToFailure"],
        "file": fname,
    })
    return feats


csv_files = [f for f in os.listdir(PREDICT_DIR) if f.endswith(".csv")]
print(f"Found {len(csv_files)} machine logs.\n")

results = []
for fname in csv_files:
    feats = extract_features(fname)
    if feats is None:
        continue
    df_feats = pd.DataFrame([feats])[FEATURES]

    pred_cycles = int(rul_model.predict(df_feats)[0])
    true_cycles = feats["cycle_to_failure"]
    near_failure = int(stage_model.predict(df_feats)[0])
    near_failure_prob = float(stage_model.predict_proba(df_feats)[0][1])

    days_left = max(0, pred_cycles // CYCLES_PER_DAY)
    fail_date = datetime.date.today() + datetime.timedelta(days=days_left)

    if pred_cycles <= CRITICAL_MAX:
        risk_level = "Critical"
    elif pred_cycles <= WARNING_MAX:
        risk_level = "Warning"
    else:
        risk_level = "Safe"

    results.append({
        "file": fname,
        "pred_cycles": pred_cycles,
        "true_cycles": true_cycles,
        "deviation": pred_cycles - true_cycles,
        "days_left": days_left,
        "fail_date": fail_date,
        "risk_level": risk_level,
        "near_failure": near_failure,
        "near_failure_prob": round(near_failure_prob, 3),
    })

print("\n=== Forecasting Results ===")
for r in results:
    print(r)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
print("Saved to", RESULTS_FILE)
