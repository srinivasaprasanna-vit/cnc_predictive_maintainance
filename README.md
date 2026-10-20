# AI-Powered Predictive Maintenance and Spare Parts Forecasting for CNC Milling Machines

A machine learning pipeline for CNC milling that estimates the Remaining Useful Life (RUL) of cutting tools from spindle vibration and current signals, and uses the estimates for two downstream tasks: tungsten-carbide price-aware spare tool stock planning, and capacity-constrained maintenance scheduling. Results are presented through a Tkinter dashboard.

This repository contains the implementation accompanying the paper by Gowtham R Nair and Srinivasa Prasanna P G, Department of Mechanical Engineering, VIT Chennai.

## Overview

The system consists of five stages:

1. **Feature extraction** — per-cycle machine logs are converted into signal features (vibration RMS, FFT spectral energy, per-phase current means) and joined with tool and workpiece metadata.
2. **Model training** — an XGBoost regressor estimates RUL in cycles; an XGBoost classifier flags cycles within 30 of failure.
3. **Price acquisition** — the FRED Producer Price Index for metal cutting tools (series WPU1135) is fetched, and the tungsten carbide price series is converted to INR.
4. **Inference** — trained models are applied to new logs to produce per-machine RUL predictions, risk levels, and projected failure dates.
5. **Planning and dashboard** — a Holt-Winters forecast of tungsten carbide price drives a buy/hold/wait stock recommendation, and a greedy scheduler assigns maintenance dates subject to a daily capacity limit.

## Dataset

The dataset is not included in this repository (approximately 112 GB). All experiments use the open CNC milling dataset published by Piecuch and Żabiński: the full life cycle of 14 cutting tools over 968 milling cycles, recorded on a Haas VF-1 machining centre (8 vibration channels at 25 kHz, 12 current channels at 0.5 kHz).

- Kaggle: https://www.kaggle.com/datasets/grzegorzpiecuch/cnc-milling
- Figshare: doi 10.6084/m9.figshare.28589216
- Dataset paper: G. Piecuch and T. Żabiński, "A new open dataset from a milling process – data for classification and estimation of tool life," Scientific Data 12, 650 (2025). doi: 10.1038/s41597-025-04923-y

The dataset creators request citation of the dataset paper on reuse.

Place the raw per-cycle CSV files and `metadata.xlsx` in `data/train/` for training, or in `data/predict/` for inference.

## Project Structure

```
.
├── features.py        # Shared feature definitions and signal feature extraction
├── process_data.py    # Stage 1: build the feature cache from raw logs
├── train.py           # Stage 2: train the RUL regressor and near-failure classifier
├── fetch_prices.py    # Stage 3: fetch price series (FRED PPI, tungsten carbide)
├── predict.py         # Stage 4: inference on new logs
├── main.py            # Stage 5: stock planning, scheduling, dashboard
├── requirements.txt
├── data/              # Input logs and feature cache (not tracked)
├── models/            # Trained models and evaluation metrics (generated)
├── prices/            # Price series (generated)
└── results/           # Inference output (generated)
```

`features.py` is the single definition of the model feature set, imported by both the training and inference code.

## Requirements

- Python 3.10 or later
- Dependencies listed in `requirements.txt` (pandas, NumPy, XGBoost, scikit-learn, statsmodels, pandas-datareader, PyArrow, Matplotlib, joblib, tqdm, openpyxl)

## Installation

```bash
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
```

## Usage

Run the stages in order:

```bash
python process_data.py   # requires data/train/ to be populated
python train.py          # trains both models, writes models/ and evaluation metrics
python fetch_prices.py   # requires prices/tungsten_carbide_price_trend_2015_2026.csv
python predict.py        # requires data/predict/ to be populated and trained models present
python main.py           # prints the planning summary and opens the dashboard
```

`main.py` re-runs `predict.py` automatically when `data/predict/` exists; otherwise it uses the existing `results/results.csv`.

## Results

Evaluated on the 758-cycle feature cache (random seed 42):

| Evaluation protocol | RUL MAE | Classifier accuracy |
|---|---|---|
| Random 80/20 split | 7.55 cycles | 94.1% (precision 0.914, recall 0.930, F1 0.922) |
| Grouped by tool, 5-fold cross-validation | 30.73 cycles | 61.1% |

The random split places cycles of the same tool in both partitions, so its figures are optimistic for unseen tools. The grouped protocol, in which no tool appears in both training and test data, is the appropriate estimate for new tools. Both protocols are reported by `train.py` on every run and saved to `models/evaluation_metrics.csv`.

## Configuration

Operating constants are defined at the top of the relevant scripts:

| Parameter | Value |
|---|---|
| Near-failure threshold | 30 cycles |
| Risk levels | Critical ≤ 10, Warning ≤ 30 predicted cycles |
| Production rate | 4 cycles/day |
| Maintenance capacity | 2 machines/day |
| Buy/wait decision band | ±5% of current price |
| Tool life (carbide end mill / coated insert / tool holder) | 120 / 30 / 500 cycles |
| USD to INR conversion rate | 83 (static) |