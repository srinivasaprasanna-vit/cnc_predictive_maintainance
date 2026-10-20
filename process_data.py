"""Stage 1: build the feature cache from raw per-cycle CNC logs.

v2 changes (see README changelog):
- Hardcoded Windows paths replaced with project-relative paths:
  raw logs + metadata.xlsx go in  data/train/
  the cache is written to          data/features_cache.parquet
- Feature extraction now comes from features.py (shared with predict.py).
"""
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from features import extract_signal_features, FAILURE_THRESHOLD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
META_FILE = os.path.join(TRAIN_DIR, "metadata.xlsx")
CACHE_FILE = os.path.join(BASE_DIR, "data", "features_cache.parquet")

df_meta = pd.read_excel(META_FILE, index_col=0)
csv_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv") and f.startswith("P")]
print(f"Found {len(csv_files)} machine logs to process.\n")


def process_file(fname):
    try:
        df_raw = pd.read_csv(os.path.join(TRAIN_DIR, fname))
        feats = extract_signal_features(df_raw)

        base_name = fname.replace(".csv", "")
        meta_row = df_meta.loc[df_meta.index == base_name]
        if meta_row.empty:
            print(f"  [!] Metadata missing for {fname}, skipping.")
            return None
        meta_row = meta_row.iloc[0]

        cycle_to_failure = meta_row["CycleToFailure"]
        feats.update({
            "MillingToolType": meta_row["MillingToolType"],
            "ToolIndex": meta_row["ToolIndex"],
            "ToolDiameter": meta_row["ToolDiameter [mm]"],
            "ToolHolderLength": meta_row["ToolHolderLength [mm]"],
            "ADOC": meta_row["ADOC [mm]"],
            "RDOC": meta_row["RDOC [mm]"],
            "HardnessMean": meta_row["HardnessMean [HRC]"],
            "failure_label": int(cycle_to_failure <= FAILURE_THRESHOLD),
            "cycle_to_failure": cycle_to_failure,
            "file": fname,
        })
        return feats
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None


results = Parallel(n_jobs=8)(
    delayed(process_file)(f) for f in tqdm(csv_files, desc="Parallel processing")
)
results = [r for r in results if r is not None]

df_cache = pd.DataFrame(results)
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
df_cache.to_parquet(CACHE_FILE, engine="pyarrow", index=False)
print(f"\nSaved processed features to {CACHE_FILE}")
print(df_cache.head())
