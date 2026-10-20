"""Stage 2: train the RUL regressor and the near-failure classifier.

v2 changes (see README changelog):
- Project-relative paths (models/ folder) instead of hardcoded Windows paths.
- Feature list imported from features.py (identical set and order to v1,
  so the headline results are unchanged: random-split MAE 7.55 cycles,
  classifier accuracy 94.1% on the 758-cycle cache).
- Evaluation now ALSO reports tool-grouped cross-validation and writes
  models/evaluation_metrics.csv, so the honest generalisation numbers
  are produced by the same run, not by hand.
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

from features import FEATURES, FAILURE_THRESHOLD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data", "features_cache.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE_RUL = os.path.join(MODEL_DIR, "rul_model.json")
MODEL_FILE_STAGE = os.path.join(MODEL_DIR, "stage_model.json")
METRICS_FILE = os.path.join(MODEL_DIR, "evaluation_metrics.csv")

df = pd.read_parquet(CACHE_FILE)
X = df[FEATURES]
y_rul = df["cycle_to_failure"]
y_stage = (df["cycle_to_failure"] <= FAILURE_THRESHOLD).astype(int)

X_train, X_test, y_rul_train, y_rul_test = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)

rul_model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="hist")
rul_model.fit(X_train, y_rul_train)
rul_pred = rul_model.predict(X_test)
mae = mean_absolute_error(y_rul_test, rul_pred)
print("RUL MAE (random 80/20):", round(mae, 2))

stage_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist")
stage_model.fit(X_train, y_stage.loc[X_train.index])
stage_pred = stage_model.predict(X_test)
y_stage_test = y_stage.loc[X_test.index]
acc = accuracy_score(y_stage_test, stage_pred)
prec = precision_score(y_stage_test, stage_pred, zero_division=0)
rec = recall_score(y_stage_test, stage_pred, zero_division=0)
f1 = f1_score(y_stage_test, stage_pred, zero_division=0)
print("Stage Accuracy (random 80/20):", round(acc, 4))

# Honest generalisation check: no cycles of a test tool appear in training.
gkf = GroupKFold(n_splits=5)
g_mae, g_acc = [], []
for tr, te in gkf.split(X, y_rul, groups=df["ToolIndex"]):
    r = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="hist")
    r.fit(X.iloc[tr], y_rul.iloc[tr])
    g_mae.append(mean_absolute_error(y_rul.iloc[te], r.predict(X.iloc[te])))
    c = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist")
    c.fit(X.iloc[tr], y_stage.iloc[tr])
    g_acc.append(accuracy_score(y_stage.iloc[te], c.predict(X.iloc[te])))
print("RUL MAE (grouped by tool, 5-fold):", round(float(np.mean(g_mae)), 2))
print("Stage Accuracy (grouped by tool, 5-fold):", round(float(np.mean(g_acc)), 4))

os.makedirs(MODEL_DIR, exist_ok=True)
rul_model.save_model(MODEL_FILE_RUL)
stage_model.save_model(MODEL_FILE_STAGE)
pd.DataFrame([
    {"protocol": "random_80_20", "rul_mae_cycles": round(mae, 2), "stage_accuracy": round(acc, 4),
     "stage_precision": round(prec, 4), "stage_recall": round(rec, 4), "stage_f1": round(f1, 4)},
    {"protocol": "grouped_by_tool_5fold", "rul_mae_cycles": round(float(np.mean(g_mae)), 2),
     "stage_accuracy": round(float(np.mean(g_acc)), 4), "stage_precision": "", "stage_recall": "", "stage_f1": ""},
]).to_csv(METRICS_FILE, index=False)
print("Models and metrics saved to", MODEL_DIR)
