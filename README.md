# Alpha Model

This repository contains the implementation of the Alpha Model, a machine learning project for tick-data-based predictions on the NQ market. It uses a Rust-based preprocessing pipeline for high-performance feature engineering and a Python-based training environment for model development.

## Repository Structure

-   `data/`: Default location for all data.
    -   `raw/`: Input data.
    -   `preprocessed/`: Output from the preprocessing pipeline.
-   `deployment/`: Real-time inference implementation.
-   `docs/`: Project documentation.
-   `preprocessing/`: The Rust-based feature engineering pipeline.
-   `training/`: The Python-based model training scripts.
-   `runs/`: Default location for model artifacts and logs.

## Getting Started

### Prerequisites

-   **Rust:** The preprocessing pipeline is written in Rust. Install it via [rustup](https://rustup.rs/).
-   **Python:** The training scripts use Python. Tested with Python 3.10+.
-   **NodeJS:** For the script running via [NPM](https://nodejs.org/en/download) (technically not needed, you could also just run the commands plain)
-   **NinjaTrader Data:** The pipeline expects raw tick data exported from NinjaTrader. The data could be aquired for example from [here (40$)](https://www.priceisking.com/products/market-replay-data-for-ninjatrader-8?srsltid=AfmBOoqr37TAUJuwkuNPos1J1iOCdTosOf2WZJXQEtTYnOe_f2E-J8Dv&variant=35316803338395) or directly through [Ninjatrader (free)](https://ninjatrader.com/support/helpguides/nt8/NT%20HelpGuide%20English.html?set_up12.htm) (only last 90 days) and then converted to .csv using [this](https://github.com/eugeneilyin/nrdtocsv).

### Installation

1.  **Set up the Python environment:**
    ```bash
    npm run packages
    ```

## Workflow

### 1. Data Preparation

The preprocessing pipeline expects raw Ninjatrader L2 data in CSV format, without a header. The data should be placed in `data/raw/`.

The expected CSV columns are (default from [this export](https://github.com/eugeneilyin/nrdtocsv)):
1.  `MarketDataType`: (0: Ask, 1: Bid, 2: Last, ...)
2.  `Timestamp`: `YYYYMMDDhhmmss` format.
3.  `Timestamp offset`: in 100-nanosecond units.
4.  `Operation`: (0: Add, 1: Update, 2: Remove)
5.  `Position`: Order book position.
6.  `MarketMaker`: Market maker identifier.
7.  `Price`: Price value.
8.  `Volume`: Volume value.


### 2. Preprocessing

The Rust pipeline processes the raw data into feature-rich Parquet files.

1.  **Configure:** Edit `preprocessing/config.yaml` to match your setup. The default configuration is a good starting point. You can specify input/output paths, resolutions, and other parameters.

2.  **Run:**
    ```bash
    npm run preprocessing
    ```
    You can also specify a different config file: `npm run preprocessing --config /path/to/your/config.yaml`

    The output will be saved to the `feature_output_path` defined in your config (by default `data/preprocessed/`).

### 3. Training

The Python scripts train a Temporal Convolutional Network (TCN) on the preprocessed features.

1.  **Configure:** Edit `training/config.yaml`. Here you can define paths, data parameters, model hyperparameters, and training settings.

2.  **Run:**
    ```bash
    npm run training
    ```
    The script uses the settings from `training/config.yaml` to load data, build the model, and run the training process. Model artifacts will be saved to the directories specified in the config file (by default under `runs/`).

## Configuration

### Preprocessing (`preprocessing/config.yaml`)

This file controls the Rust preprocessing pipeline. Key sections:
-   `instrument`: Instrument metadata (symbol, tick size, etc.).
-   `io`: Input and output paths.
-   `resolutions`: Configuration for different time resolutions (fast, mid, slow).
-   `normalization`: Parameters for rolling window normalizers.
-   `labeling`: Specification for target labels.
-   `data_split`: Ratios for train/validation/test splits.

### Training (`training/config.yaml`)

This file controls the Python training process. Key sections:
-   `paths`: Paths for data, models, and checkpoints.
-   `data`: Parameters for data loading and feature selection.
-   `training`: Training hyperparameters (batch size, epochs, learning rate, etc.).
-   `model`: TCN model architecture.
-   `chunking`: Configuration for memory-aware chunk-based training.

## Overview

**Goal:**
Predict short-horizon price direction in a market where “time to move” varies (sometimes seconds, sometimes many minutes), using a single multi-scale neural network with minimal Python-side work.

**Core modeling idea:**

1.  **Define labels in event/price space, not fixed clock time:**
    – Example: label = 1 if price hits `+X ticks` before `−Y ticks` within the next `K` events/bars; label = 0 otherwise. (Default 10 pips / 40 ticks in each direction)
    – This makes targets more stable across fast vs slow markets.

2.  **Use a single multi-scale model (multi-branch TCN):**
    – Branch for high resolution (microstructure, fast patterns).
    – Branch for medium resolution (intermediate patterns).
    – Branch for low resolution (trend/regime context).
    – Outputs from all branches are concatenated and passed to a shared dense + sigmoid head that predicts a single probability in (0,1).

**Preprocessing approach (three resolutions precomputed in Rust):**
Do the heavy lifting in Rust, then keep Python thin:

1.  **In Rust (one main pipeline + parameterized resampling):**
    a) Clean and standardize raw tick data into a canonical format (trades/quotes, midprice, spread, etc.).
    b) Build multiple bar streams at different resolutions, e.g.:
    – `fast` : 1-second (or small event-based) bars
    – `mid` : 10-second bars
    – `slow` : 60-second bars
    c) For each resolution, compute features:
    – returns, rolling volatility, volume, spread, imbalance, etc.
    d) Define labels once using your chosen event-based rule (“hit +X before −Y within K events from time t”).
    – Ensure each bar stream gets aligned labels that refer to the same decision time t.
    e) Save as three Parquet files (Pattern A):
    – `data_fast.parquet`
    – `data_mid.parquet`
    – `data_slow.parquet`
    Each with: `timestamp/index`, features for that resolution, and `label`.

2.  **In Python (minimal work):**
    a) Load the three Parquet files.
    b) Align rows across resolutions by timestamp/index so each training sample has:
    – a window from the fast series,
    – a window from the mid series,
    – a window from the slow series,
    – and a single label.
    c) Convert these into tensors:
    – `X_fast : (batch, L_fast, F_fast)`
    – `X_mid  : (batch, L_mid, F_mid)`
    – `X_slow : (batch, L_slow, F_slow)`
    – `y      : (batch,)`
    d) Feed them into the multi-branch CNN/TCN model and train.

**Result:**
You end up with:

-   A single multi-scale neural network that outputs one probability (up vs down) and is robust to fast and slow markets.
-   Most complexity (cleaning, feature construction, multi-resolution resampling, labeling) lives in Rust.
-   Python is mainly a thin layer: load Parquet → build windows → train/evaluate the model.
