# Project Alpha

Project Alpha is a research-grade machine learning pipeline for short-horizon NQ futures direction prediction using NinjaTrader tick/L2 market data. The project combines a high-performance Rust preprocessing engine with Python-based model training, evaluation, and real-time inference tooling.

The core idea is to move expensive market-data processing into Rust, export clean multi-resolution Parquet datasets, and keep Python focused on sequence construction, model training, and experiment analysis.

## Repository description

**Multi-resolution NQ futures ML research pipeline using Rust preprocessing, Parquet feature stores, and Python TCN training for short-horizon market-direction experiments.**

## Project status

This is a market microstructure research project, not a production trading product. It is designed to test whether L2/tick-derived features can produce useful short-horizon signals under realistic constraints. No profitability is implied by the repository.

## What it does

- Parses NinjaTrader tick/L2 CSV exports.
- Reconstructs market state and derives microstructure features.
- Builds multiple time resolutions: fast, mid, and slow.
- Creates first-touch labels for different target horizons.
- Exports feature datasets as Parquet.
- Trains a multi-resolution TCN-style model in Python.
- Includes scripts for offline evaluation and real-time feature/inference workflows.

## Why this project matters

Project Alpha demonstrates practical ML engineering for noisy, high-volume time-series data:

- **Performance engineering:** Rust handles heavy preprocessing, normalization, labeling, and feature generation.
- **Data architecture:** Parquet outputs separate preprocessing from training and make experiments repeatable.
- **Multi-scale modeling:** fast, mid, and slow streams allow the model to see microstructure and broader regime context.
- **Config-driven workflows:** preprocessing, training, evaluation, and deployment behavior are controlled through YAML and npm scripts.
- **Research discipline:** labels, splits, normalization, and evaluation are explicit instead of hidden inside ad-hoc notebooks.

## High-level architecture

```text
NinjaTrader CSV exports
        ↓
Rust preprocessing engine
        ↓
Cleaned market state + engineered features
        ↓
Multi-resolution Parquet datasets
        ↓
Python training pipeline
        ↓
TCN model + evaluation artifacts
        ↓
Optional real-time feature stream and inference tooling
```

## Tech stack

| Area | Technology |
|---|---|
| Preprocessing | Rust |
| Training | Python |
| Model family | Temporal Convolutional Network / multi-scale sequence model |
| Data format | Parquet |
| Configuration | YAML |
| Data source | NinjaTrader tick/L2 CSV exports |
| Orchestration | npm scripts for repeatable commands |

## Repository structure

```text
ProjectAlpha/
├── data/
│   ├── raw/                    # Raw NinjaTrader CSV exports
│   └── preprocessed/           # Generated Parquet features and checkpoints
├── preprocessing/              # Rust feature engineering and realtime processing
│   └── config.yaml             # Preprocessing configuration
├── training/                   # Python training code and model config
│   └── config.yaml             # Training configuration
├── evaluation/                 # Offline evaluation and sweep tooling
├── deployment/                 # Realtime inference tooling
├── deployment-simple/          # Simplified realtime trigger flow
├── runs/                       # Generated models, logs, caches, and inference output
├── package.json                # Common commands
└── README.md
```

## Data assumptions

The pipeline expects NinjaTrader-style L2/tick CSV data, typically exported or converted into rows containing:

1. Market data type: bid, ask, last, etc.
2. Timestamp in `YYYYMMDDhhmmss` format.
3. Timestamp offset in 100-nanosecond units.
4. Operation: add, update, remove.
5. Order book position.
6. Market maker identifier.
7. Price.
8. Volume.

The project is configured around NQ futures with a `0.25` tick size.

## Preprocessing

The Rust preprocessing pipeline is responsible for:

- Cleaning and standardizing raw tick/L2 records.
- Reconstructing bid/ask depth state.
- Building time-based bars at multiple resolutions.
- Computing feature groups such as:
  - top-of-book values
  - depth and liquidity features
  - order-flow features
  - trade features
  - volatility features
  - time-of-day encodings
  - regime/context features
- Generating first-touch labels such as `t20`, `t40`, `t60`, and `t100`.
- Saving Parquet files for downstream training.

Default resolutions:

| Resolution | Bar size | Purpose |
|---|---:|---|
| `fast` | 1 second | Microstructure and immediate order-flow behavior |
| `mid` | 10 seconds | Intermediate context |
| `slow` | 60 seconds | Broader regime context |

Run preprocessing:

```bash
npm run preprocessing
```

## Training

The Python training pipeline loads the generated Parquet files and trains a sequence model using fast, mid, and slow feature streams.

The default training target is `t40`, representing a 40-tick first-touch direction label. Configuration lives in:

```text
training/config.yaml
```

Run training:

```bash
npm run training
```

The training config controls:

- feature roots
- target label
- sequence length
- batch size
- learning rate
- early stopping
- TCN architecture
- threshold sweeps
- trade-simulation parameters
- device settings

## Evaluation and real-time workflow

Project Alpha includes scripts for:

- Offline evaluation sweeps.
- Real-time feature generation from a live NinjaTrader log file.
- Realtime inference using a saved model bundle.
- Simplified hotkey-style trigger workflows for experimentation.

Example commands from `package.json`:

```bash
npm run realtime
npm run deployment
npm run deployment:simple
npm run evaluation
```

These workflows should be treated as research tooling and require careful validation before any live use.

## Installation

### Prerequisites

- Rust via `rustup`
- Python 3.10+
- Node.js 14+ for npm script orchestration
- NinjaTrader tick/L2 data exported or converted to CSV

### Install Python dependencies

```bash
npm run packages
```

### Prepare data

Place raw CSV files in the configured raw data directory. The default paths are defined in:

```text
preprocessing/config.yaml
training/config.yaml
```

Then run:

```bash
npm run preprocessing
npm run training
```

## Recruiter notes

Project Alpha is a strong technical portfolio project because it shows end-to-end ML systems thinking: high-throughput Rust preprocessing, explicit feature engineering, Parquet-based dataset design, sequence modeling, configuration management, evaluation tooling, and real-time inference experiments. It also demonstrates awareness of market-data leakage risks, label design, and the difference between research signals and production trading readiness.
