# Evaluation Replay Runner

This folder contains a small wrapper script that lets you **replay historical
raw CSV data through the exact same realtime stack** used in deployment:

- Rust realtime preprocessor (`preprocessing/src/bin/realtime.rs`)
- Python inference watcher (`deployment/realtime_inference.py`)

The only difference from live trading is that the input is a finite set of
CSV files under `data/eval` and the replay runs as fast as your machine
allows (no need to wait for normal market speed).

## 1. Folder layout

- `evaluation/realtime_eval.py` – main entry point.
- `data/eval/` – place your NinjaTrader-style raw CSV files here.
- `runs/evaluation/` – output directory for generated feature streams and logs.

## 2. Preparing data

1. Copy the raw Level-2 CSV files you want to test with into:
   - `data/eval/*.csv`
2. Ensure you have a trained model bundle and normalization snapshots:
   - Model bundle (default): `runs/models/best_model.pt`
   - Normalization state (default): `data/preprocessed/.checkpoints/normalization/latest/`

These defaults mirror the training/deployment setup. You can override them via
CLI flags if needed.

## 3. Installing dependencies

From the repo root, run:

```bash
npm run packages
```

This installs Python dependencies for both training and deployment. You also
need Rust + Cargo installed, since the evaluation stack reuses the realtime
preprocessor binary.

## 4. Running an evaluation replay

From the repo root:

```bash
python evaluation/realtime_eval.py
```

What this does:

1. Concatenates all `data/eval/*.csv` files into a temporary replay log.
2. Runs the Rust realtime preprocessor on that log:
   - Uses the same resolutions as deployment (`fast,mid,slow` by default).
   - Uses the same normalization checkpoint directory as training.
   - Emits features to `runs/evaluation/features_eval.jsonl`.
3. Runs the Python inference watcher on the generated features:
   - Loads `runs/models/best_model.pt` by default.
   - Replays the entire features file from the beginning (no tailing).
   - Prints per-target probabilities to stdout as in deployment.

At the end, the temporary features file is removed unless you ask to keep it.

## 5. Useful flags

Examples:

```bash
# Use a specific model bundle and keep the generated features file
python evaluation/realtime_eval.py \
  --model-bundle runs/models/phase5_model.pt \
  --keep-features

# Point at a custom eval data directory and normalization snapshot folder
python evaluation/realtime_eval.py \
  --raw-dir data/eval/my_experiment \
  --norm-state-dir data/preprocessed/.checkpoints/normalization/latest

# Forward additional args directly to the deployment watcher (e.g. thresholds)
python evaluation/realtime_eval.py \
  --inference-extra-args --trade-threshold 0.8 --entry-min-price-move 10.0
```

Key arguments:

- `--model-bundle` – path to the `.pt` or `.pkl` bundle used in deployment.
- `--raw-dir` – directory containing raw `.csv` files for evaluation.
- `--output-dir` – where `features_eval.jsonl` is written.
- `--resolutions` – resolutions (must include `fast mid slow` for TCN model).
- `--norm-state-dir` – directory with `fast.json`, `mid.json`, `slow.json` scaler states.
- `--keep-features` – prevent deletion of `features_eval.jsonl`.
- `--inference-extra-args` – everything after this flag is passed to
  `deployment/realtime_inference.py`.

## 6. How it mirrors deployment

The evaluation runner **does not implement its own preprocessing or inference**
logic. Instead it:

1. Calls the Rust `realtime` binary with:
   - `--resolutions` identical to deployment.
   - `--norm-state-dir` pointing at the same normalization snapshots.
2. Calls `deployment/realtime_inference.py` with:
   - The same model bundle format and standardization logic.
   - `--replay-existing` so it reads the features file from the beginning.

This ensures that any behaviour you see during evaluation is produced by
the same codepaths that run in live deployment—just driven by historical
CSV data instead of the live NinjaTrader log.

