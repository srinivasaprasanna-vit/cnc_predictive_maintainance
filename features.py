"""Shared feature definitions and per-file feature extraction.

Single source of truth used by BOTH process_data.py (training) and
predict.py (inference), so the train and inference feature logic can
never drift apart (a limitation called out in the paper, Section 6).
"""
import numpy as np

# Exact feature set the saved XGBoost models were trained on (v1-compatible).
# NOTE: process_data.py also extracts ADOC, RDOC and HardnessMean into the
# cache, but they are deliberately NOT in this list: an ablation
# (2026-10-20) showed adding them improves the random-split MAE
# (7.55 -> 6.47) but worsens tool-grouped MAE (30.73 -> 33.71) and the
# separate test cache (25.48 -> 28.94), because they are constant per tool
# and act as extra tool-identity fingerprints.
FEATURES = [
    "spindle_load", "energy", "watt", "vib_rms0", "vib_rms1", "vib_fft",
    "MillingToolType", "ToolIndex", "ToolDiameter", "ToolHolderLength",
]

FAILURE_THRESHOLD = 30  # cycles: cycle_to_failure <= 30 is "near failure"


def extract_signal_features(df_raw):
    """Signal features from one raw per-cycle log (a pandas DataFrame).

    Values are computed on the raw channel values exactly as in v1
    (no unit conversion), to stay compatible with the saved models.
    """
    vib_rms0 = np.sqrt(np.mean(df_raw["Accelerometer - Spindle +Y"] ** 2))
    vib_rms1 = np.sqrt(np.mean(df_raw["Accelerometer - Spindle -Z"] ** 2))
    fft_vals = np.fft.fft(df_raw["Accelerometer - Spindle +Y"].values)
    vib_fft = np.sum(np.abs(fft_vals[: len(fft_vals) // 2]))

    # Per-phase spindle current means. The v1 names (spindle_load / watt /
    # energy) are kept for model compatibility; all three are current means.
    spindle_load = df_raw["Current - Spindle L1"].mean()
    watt = df_raw["Current - Spindle L2"].mean()
    energy = df_raw["Current - Spindle L3"].mean()

    return {
        "spindle_load": spindle_load,
        "energy": energy,
        "watt": watt,
        "vib_rms0": vib_rms0,
        "vib_rms1": vib_rms1,
        "vib_fft": vib_fft,
    }
