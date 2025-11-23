# Project Alpha — Step-by-step Pipeline Setup and Run Guide

This guide is a hands-on checklist to get the full Project Alpha pipeline running (preprocessing → training → realtime inference) with the improvements you just added (normalization snapshotting, label tie-breaker, session gap handling).

The instructions assume you are working on Windows but using `bash.exe` (Git Bash / WSL-compatible shells) as your shell. Commands are copy/paste-ready for that environment.

---

**Overview / goals**
- Build and run the Rust preprocessing pipeline to produce per-resolution Parquet feature files and persisted causal normalization snapshots.
- Train the Python model(s) using the produced Parquet datasets.
- Run the Rust realtime feature generator seeded with the saved normalization state to avoid cold-start distribution drift.
- Run the Python realtime inference watcher against the JSONL stream.
- Provide verification steps to compare offline (batch) vs realtime outputs.

---

## 0. Prerequisites
- Rust toolchain (stable): https://rustup.rs
- Python 3.10+ (recommended) and `venv`
- Node/npm if using repository scripts (some helper scripts may exist in `package.json`)
- Cargo & Git available in your PATH

Confirm basic tools:

```bash
rustc --version
cargo --version
python --version
pip --version
git --version
```

---

## 1. Build the Rust preprocessing crate
1. Open your bash shell in the repo root `c:/Users/timowilde/Documents/coding/ProjectAlpha`.

2. Build the preprocessing crate (this compiles the CLI + realtime binary):

```bash
cd preprocessing
cargo build --release
```

3. Run unit tests for preprocessing (optional quick sanity):

```bash
cargo test
```

If tests fail, inspect the test outputs—many helper modules include unit tests that are lightweight.

---

## 2. Dry-run and inspect config
1. The repository includes a default pipeline config. Do a dry-run to confirm configured resolutions and IOs (prints plan, no files written):

```bash
cargo run --release -- --dry-run
```

2. To run a single file (legacy mode) or a directory, inspect `preprocessing/src/config/mod.rs` default paths, then override with flags:

```bash
cargo run --release -- --input data/raw/training/TODO --output data/preprocessed/training/TODO
```

Replace `TODO` with your actual file / directory paths.

---

## 3. Run batch preprocessing to produce features + scaler state
We recommend running on historical raw logs first so scalers warm up.

1. Run preprocessing on a training raw folder (example):

```bash
# from repo root
cd preprocessing
cargo run --release -- --input data/raw/training --output data/preprocessed/training
```

2. Output layout (defaults from `PipelineConfig::example()`):
- Features written to: `data/preprocessed/training/<resolution>/<source>.parquet` (e.g. `data/preprocessed/training/fast/20250101.parquet`)
- Labels: written alongside features under the output root (check `write_labels_parquet` in `executor.rs`)
- Normalization checkpoints:
  - Per-file: `<checkpoint_path>/normalization/<resolution>/<source>.json`
  - Latest per resolution: `<checkpoint_path>/normalization/latest/<resolution>.json`

Check the configured checkpoint directory by inspecting your pipeline config or the example default in `preprocessing/src/config/mod.rs` (`io.checkpoint_path`).

3. Verify checkpoints exist:

```bash
ls data/preprocessed/training/.checkpoints/normalization/latest
# expect files: fast.json mid.json slow.json (or similarly named)
```

If those JSON snapshot files exist, the normalization state export worked.

---

## 4. Training (Python)
1. Create a virtual environment and install requirements:

```bash
cd training
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare feature + label dataset input paths in the training command. The training entrypoint is `training/tcn.py`. Typical run might look like:

```bash
cd ..  # repo root
python training/tcn.py \
  --feature-root data/preprocessed/training \
  --label-root data/preprocessed/training \
  --resolutions fast mid slow \
  --seq-len 64 \
  --batch-size 32 \
  --out runs/models/phase5_model.pt
```

Adjust flags according to script's CLI help. The training code performs multi-resolution augmentation and will forward-fill slower-resolution columns per `source_file` (this is expected and consistent with deployment after session handling).

3. After training completes, the model bundle should be saved to `runs/models/phase5_model.pt` (or another location you configured). The bundle includes scaler statistics for standardization used at inference time.

Notes:
- If logistic regression is preferred for production, check `training/` for helpers to export a scikit-learn model (not always present). Otherwise export TCN.

---

## 5. Realtime feature generation (Rust) — seed with saved normalization state
The realtime binary (`preprocessing/bin/realtime`) now supports seeding causal scalers from saved JSON snapshots using `--norm-state-dir`. This eliminates cold-start differences vs batch.

1. Example run (tail file and emit JSONL):

```bash
# from repo root
cargo run --bin realtime --release -- \
  --resolutions fast,mid,slow \
  --source C:/Ninjatrader/L2Log.txt \
  --emit runs/realtime/features.jsonl \
  --norm-warmup 0 \
  --norm-state-dir data/preprocessed/training/.checkpoints/normalization/latest \
  --session-gap-secs 900
```

Flags explained:
- `--norm-warmup`: optionally discard initial base bars if you still want an additional warmup window in realtime (set to 0 if seeding is trusted).
- `--norm-state-dir`: directory containing `fast.json`, `mid.json`, `slow.json` (the "latest" snapshots produced by batch runs).
- `--session-gap-secs`: if the tailed stream has a long idle gap, the preprocessor will clear the multi-resolution cache and restart warmup (prevents slow/residual data leaking across sessions).

2. Check that JSONL lines are emitted to `runs/realtime/features.jsonl`

```bash
tail -n 20 runs/realtime/features.jsonl
```

Each line is a JSON payload with `features` map and optionally `*@<resolution>` suffixed columns for multi-resolution features.

---

## 6. Realtime inference (Python)
1. Ensure `deployment/` requirements are installed (use the same venv or a dedicated one):

```bash
pip install -r deployment/requirements.txt
```

2. Run the inference watcher pointing at the feature JSONL and the model bundle:

```bash
python deployment/realtime_inference.py \
  --model-bundle runs/models/phase5_model.pt \
  --features runs/realtime/features.jsonl \
  --resolutions fast mid slow
```

The watcher reads JSONL, applies the training standardization (from the model bundle), warms up a sequence buffer, and emits probabilities once primed.

---

## 7. Parity verification (suggested)
To ensure deployment fidelity, run parity checks between offline Parquet features and replaying the same raw logs through the realtime generator:

1. Run preprocessing batch on a single raw file -> produce `data/preprocessed/training/<resolution>/<stem>.parquet` and a per-file normalization snapshot under `.checkpoints/normalization/<resolution>/<stem>.json`.

2. Replay the same raw file with `realtime` binary (set `--source` to that file and `--start-from-end=false`) while pointing `--norm-state-dir` either at the per-file json or latest snapshots. Capture the emitted JSONL to `tmp_replay.jsonl`.

3. Use a short Python script to load the Parquet features and the JSONL payloads and compare numeric columns (abs diff < 1e-6 for well-warmed features). If differences appear, check warmup counters and whether `--norm-warmup` was used.

A minimal parity script outline:
- Read Parquet into pandas DataFrame (fast resolution)
- Read JSONL messages into list of dicts, build DataFrame of features
- Join on `start_timestamp_ns` and compare numeric columns

---

## 8. Troubleshooting & common pitfalls
- If `runs/realtime/features.jsonl` is empty:
  - Check that `realtime` binary receives events (tailing the source file). Try `--start-from-end=false` for a test file.
  - Check `--resolutions` are all present. The realtime preprocessor requires all configured resolutions to produce multi-resolution payloads.

- If model predictions are poor initially:
  - Confirm you seeded normalization via `--norm-state-dir` (otherwise initial bars will have different distributions).
  - Optionally increase `--norm-warmup` to discard more initial base bars.

- If batch preprocessing fails to produce checkpoints:
  - Ensure the `io.checkpoint_path` directory is writeable and that your `PipelineConfig` points to the expected location.
  - Inspect logs for messages about persisting scaler state (the executor will write per-file and latest JSONs).

- If training complains about missing columns:
  - Verify `training/tcn.py` is given the same base feature columns as produced by the preprocessing schema; `training` expects columns like `start_timestamp_ns`, `end_timestamp_ns`, and named features used in config.

---

## 9. Quick command checklist (copyable)

```bash
# Build preprocess
cd preprocessing
cargo build --release
cargo test

# Run batch preprocess (example)
cargo run --release -- --input data/raw/training --output data/preprocessed/training
ls data/preprocessed/training/.checkpoints/normalization/latest

# Train
cd ../training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tcn.py --feature-root ../data/preprocessed/training --label-root ../data/preprocessed/training --resolutions fast mid slow --seq-len 64 --batch-size 32 --out ../runs/models/phase5_model.pt

# Run realtime generator (seed scalers)
cd ../preprocessing
cargo run --bin realtime --release -- --resolutions fast,mid,slow --source C:/Ninjatrader/L2Log.txt --emit ../runs/realtime/features.jsonl --norm-state-dir ../data/preprocessed/training/.checkpoints/normalization/latest --session-gap-secs 900

# Run inference watcher
cd ../deployment
source ../training/.venv/bin/activate
pip install -r requirements.txt
python realtime_inference.py --model-bundle ../runs/models/phase5_model.pt --features ../runs/realtime/features.jsonl --resolutions fast mid slow
```

---

If you'd like, I can:
- Add the small parity-check Python script into `tools/` and a simple runner that automates steps 1–3 of the parity verification.
- Add example `cargo` and `python` npm scripts to `package.json` to standardize the commands.

Tell me which of these you'd like next and I will add them and run tests where applicable.