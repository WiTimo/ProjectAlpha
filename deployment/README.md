# Deployment Guide

## 1. Realtime Feature Generator

The Rust binary mirrors the batch preprocessing pipeline: it tails the
NinjaTrader Level-2 log, rebuilds quotes/order-flow exactly like the offline
reader, aggregates bars, and emits fully normalized Phase 5 feature vectors as
JSONL.

```bash
cd preprocessing
cargo run --bin realtime \
	-- --resolutions fast,mid,slow \
	--source C:/Ninjatrader/L2Log.txt \
	--emit ../runs/realtime/features.jsonl
```

To reuse the fully warmed causal scalers from batch preprocessing, point `--norm-state-dir` at the checkpoint folder (e.g. `../data/preprocessed/training/.checkpoints/normalization/latest`). This eliminates the cold-start distribution shift at market open.

Important flags:

- `--config` – optional pipeline config file (YAML/JSON). Defaults to the built-in example.
- `--resolutions` – **must be** `fast,mid,slow` for both training and deployment. All three are required and must be present in every run.
- `--source` – NinjaTrader `L2Log.txt` (or any compatible feed).
- `--emit` – JSONL output path consumed by the inference watcher.
- `--follow`, `--poll-ms`, `--tick-size`, `--levels` – runtime overrides when needed.
- `--norm-warmup` – discard this many initial base bars to allow rolling normalization statistics (EWMA/windowed means) to stabilize before emitting features for inference. Set to the largest rolling window (e.g., 150–200) if early-bar distribution mismatch impacts live predictions.
- `--norm-state-dir` – optional directory containing `fast.json`, `mid.json`, `slow.json` scaler snapshots produced by the Rust batch pipeline (`.checkpoints/normalization/latest`). When provided, realtime scalers are seeded with the saved state instead of starting cold each session.
- `--session-gap-secs` – maximum idle gap before the realtime preprocessor resets multi-resolution caches and restarts the warmup sequence. Keep this aligned with the largest gap between files in training (default = 900 seconds) to match the forward-fill semantics applied offline.

**Note:** The pipeline will error if any resolution is missing. All three (fast, mid, slow) must be present for correct operation.

```json
{
	"resolution": "fast",
	"bar_index": 12345,
	"start_timestamp_ns": 1732137600000000000,
	"end_timestamp_ns": 1732137601000000000,
	"features": { "mid_return_bar": 0.00042, ... }
}
```

The realtime loop **requires** all three resolutions. It buffers until every slower stream has produced at least one bar, matching the training pipeline's requirement that multi-resolution columns be fully populated before inference. After a configurable idle gap (`--session-gap-secs`) the cache resets, forcing a fresh warmup so that stale slow-resolution data is never forwarded into a new session. If any resolution is missing, inference will not run.

## 2. Realtime Inference

Point the Python watcher at the feature stream plus the exported model bundle:

```bash
cd deployment
python realtime_inference.py \
	--model-bundle ../runs/models/phase5_model.pt \
	--features ../runs/realtime/features.jsonl \
	--resolutions fast mid slow
```

**Important:** The deployment script will error if any resolution is missing. You must provide all three: `fast`, `mid`, and `slow`.

The script loads the Phase 5 TCN bundle (model + scaler stats), standardizes the incoming feature columns (including any `@resolution` suffixes), warms up the sequence buffer to match the trained window length, runs inference, and logs the `up` probability for every configured target once the buffer is primed. All three resolutions are required for correct operation.

## End-to-end checklist

1. Install dependencies: `npm run packages` (installs both training + deployment requirements).
2. Start the Rust realtime generator (Step 1) – verify that
	 `runs/realtime/features.jsonl` grows as the NinjaTrader log updates.
3. Start the Python inference watcher (Step 2) – observe probability logs in the terminal.
4. Optionally feed the predictions into downstream execution logic (not included here).
