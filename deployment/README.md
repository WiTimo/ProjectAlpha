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

Important flags:

- `--config` – optional pipeline config file (YAML/JSON). Defaults to the built-in example.
- `--resolutions` – comma-separated list of bar streams to emit. The first entry is treated as the base stream written to JSONL; slower resolutions feed multi-resolution suffixes (e.g., `mid_return_bar@mid`).
- `--source` – NinjaTrader `L2Log.txt` (or any compatible feed).
- `--emit` – JSONL output path consumed by the inference watcher.
- `--follow`, `--poll-ms`, `--tick-size`, `--levels` – runtime overrides when needed.

The emitter creates `runs/realtime/features.jsonl` by default, appending one JSON
record per completed bar:

```json
{
	"resolution": "fast",
	"bar_index": 12345,
	"start_timestamp_ns": 1732137600000000000,
	"end_timestamp_ns": 1732137601000000000,
	"features": { "mid_return_bar": 0.00042, ... }
}
```

When multiple resolutions are enabled the realtime loop buffers until every
slower stream has produced at least one bar, matching the training pipeline's
requirement that multi-resolution columns be fully populated before inference.

## 2. Realtime Inference

Point the Python watcher at the feature stream plus the exported model bundle:

```bash
cd deployment
python realtime_inference.py \
	--model-bundle ../runs/models/phase5_model.pt \
	--features ../runs/realtime/features.jsonl \
	--resolution fast
```

The script loads the Phase 5 TCN bundle (model + scaler stats), standardizes the
incoming feature columns (including any `@resolution` suffixes), warms up the
sequence buffer to match the trained window length, runs inference, and logs the
`up` probability for every configured target once the buffer is primed. Set
`--resolution all` to observe every emitted base stream if needed.

## End-to-end checklist

1. Install dependencies: `npm run packages` (installs both training + deployment requirements).
2. Start the Rust realtime generator (Step 1) – verify that
	 `runs/realtime/features.jsonl` grows as the NinjaTrader log updates.
3. Start the Python inference watcher (Step 2) – observe probability logs in the terminal.
4. Optionally feed the predictions into downstream execution logic (not included here).
