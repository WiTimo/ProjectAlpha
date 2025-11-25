# Project Alpha – Training, Evaluation & Deployment Overview

This document explains how the system is wired end‑to‑end:

- what happens during **preprocessing & training**
- how the model is **exported** and used in **deployment**
- how **offline evaluation** works (both trade‑level and label‑level)
- how all these stages are connected and where they intentionally differ

The goal is to make it easy to reason about why a model looks “good” during training but close to random in trade‑level evaluation, and how to debug that gap.

---

## 1. Data & Preprocessing

### 1.1 Raw data

- Live/eval raw data lives under `data/eval/*.csv` (NinjaTrader‑style L2 logs).
- Training data comes from preprocessed parquet files under `data/preprocessed/{fast,mid,slow}/*.parquet`.
  - These parquet files are produced by the Rust preprocessing pipeline in `preprocessing` (see `npm run preprocessing` and `npm run realtime` in `package.json`).

### 1.2 Preprocessed parquet → streaming cache

The Python training pipeline never reads raw CSV directly. Instead it operates on a **streaming cache** built from the parquet files:

- Entry point: `training/tcn.py`.
- It uses `src.data.loader.cache_streaming_files` to build a cache under `runs/cache/stream`:
  - For every selected day (parquet stem, e.g. `20240610`):
    - Loads synchronized features per resolution from `data/preprocessed/{res}/{stem}.parquet`.
    - Computes tri‑class labels (`down / flat / up`) via `derive_triclass_targets` based on:
      - `trade_simulation.tick_size`
      - `trade_simulation.target_ticks`
      - `trade_simulation.stop_ticks`
      - optional `label_lookahead_bars` horizon.
    - Saves:
      - standardized‑ready features: `runs/cache/stream/{res}/{stem}_{res}.npy`
      - labels: `runs/cache/stream/{stem}_targets.npy`
      - valid label mask: `runs/cache/stream/{stem}_target_mask.npy`
      - walk‑forward exit indices: `runs/cache/stream/{stem}_exit_idx.npy`
      - timestamps & prices: `runs/cache/stream/{stem}_timestamps.npy`, `runs/cache/stream/{stem}_prices.npy`
      - per‑file feature sums / sums of squares (for global mean/std computation).

These cached arrays are the **single source of truth** for:

- training features and labels
- validation / test features and labels
- label‑level offline evaluation on specific days (see `evaluation/offline_class_eval.py`).

### 1.3 Standardization

After caching, `tcn.py` calls:

- `compute_stats(train_entries, resolutions, col_map)`
  - Computes global mean/std per feature using only **train** entries.
- `standardize_entries(entries, stats, clip_value)`
  - Applies `(x - mean) / std` in‑place on cache `.npy` files.
  - Replaces NaN/Inf and optionally clips to `[-standardize_clip, +standardize_clip]` from `training/config.yaml`.

The resulting standardized features are what the model sees during training and what are later mirrored in deployment via the `scaler` metadata saved in the model bundle.

---

## 2. Training

### 2.1 Entry point and configuration

- Main script: `training/tcn.py` (invoked via `npm run training`).
- Loads configuration from `training/config.yaml`:
  - `paths.*`: feature root, cache dir, model export dir.
  - `data.*`: resolutions, feature set, splits, standardization clip, etc.
  - `training.*`: target horizon, sequence length, batch size, optimizer and early‑stopping settings.
  - `trade_simulation.*`: tick size and target/stop ticks used for label generation and in‑training trade simulation.
  - `system.*`: device, num_workers, etc.

### 2.2 Dataset construction

Within `tcn.py`:

1. **Feature column setup**
   - Uses `FEATURE_SET_COLUMNS[feature_set]` from `src.definitions` and `data.resolutions` to define the flattened channel ordering and `feature_columns` list.
2. **Split train/val/test by day**
   - `entries` (cache metadata) are sorted by `start_ts` and split into:
     - `train_entries`
     - `val_entries`
     - `test_entries`
   - Split sizes are controlled by `data.val_days` and `data.test_days` (in days, based on the ordered stems).
3. **Standardization**
   - As described above, uses only training entries to compute means/stds and standardizes all entries.
4. **Dataloaders**
   - Builds `MultiResolutionSequenceDataset(train_entries, seq_len, resolutions)` and analogous for val.
   - Each `__getitem__` returns `(x, y, meta)`:
     - `x`: tensor of shape `[channels, seq_len]` (concatenated resolutions).
     - `y`: tri‑class label vector; the primary target is `y[:, 0]`.
     - `meta`: `(file_idx, target_idx)` for diagnostics.

### 2.3 Model and loss

Model definition:

- `src.model.network.DilatedTCN` with:
  - `num_inputs = total_input_channels` (length of `feature_columns`).
  - `num_classes = NUM_TARGET_CLASSES` (tri‑class: down/flat/up).
  - `num_channels = [hidden_dim] * layers`, `kernel_size`, `dropout`, `dilation_base` from config.

Loss & optimization:

- `train_epoch` uses `torch.nn.functional.cross_entropy` with class weights computed from training label distribution:
  - `src.evaluation.metrics.get_class_weights` (powered by `training.class_weight_power`).
  - Optional gradient clipping via `max_gradient_norm`.
- Scheduler: `ReduceLROnPlateau` on validation AUC, `mode="max"`.

### 2.4 Validation, metrics, and in‑training trade simulation

`evaluate_and_log` in `src.training.engine` performs rich validation:

- Computes weighted & unweighted validation loss.
- Aggregates:
  - tri‑class probabilities per sample
  - primary class labels
  - move vs flat targets (UP+DOWN vs FLAT) for AUC.
- Optionally trains/compares a **logistic regression baseline** on the last step of the sequence (`train_logistic_baseline` in `src.evaluation.baselines`):
  - reports baseline accuracy/AUC and correlation with TCN outputs.
- Optionally runs a **label‑aware trade simulator** (`src.evaluation.trade_simulator.TradeSimulator`) on the prediction records:
  - uses the same label definition (walk‑forward target/stop) to simulate entries/exits on the *evaluation set*, not the raw price series.
  - sweeps thresholds (`trade_simulation.threshold_sweep`) and records expectancy/net ticks.

The epoch results are summarized via `generate_epoch_report` and printed to stdout/log.

### 2.5 Model export (deployment bundle)

On improvement in AUC (subject to baseline margin & patience), `tcn.py` exports a deployment bundle to `paths.model_export_dir` (usually `runs/models/best_model.pt`) containing:

- `model_state_dict`: TCN state.
- `feature_columns`: flattened feature order.
- `target_columns`: list of training targets (e.g. `["t40"]`).
- `training` metadata:
  - `sequence_len`, model hyperparameters, feature set and resolutions, primary target name.
- `scaler`: means/stds per feature column used during training standardization.

This bundle is the only artifact needed by the deployment and offline evaluation pipelines (plus the normalization snapshots for the Rust realtime preprocessor).

---

## 3. Deployment (Realtime Inference)

### 3.1 Entry point and pipeline

- Main script: `deployment/realtime_inference.py` (typically invoked via `npm run deployment` or by the offline evaluator).
- Responsibilities:
  - Load the exported model bundle (`runs/models/best_model.pt`).
  - Watch a live or replayed feature stream (`features.jsonl`) produced by the Rust realtime preprocessor.
  - Maintain a sliding window of `sequence_len` steps.
  - Standardize features with the same means/stds as training.
  - Run the TCN and produce tri‑class probabilities per step.
  - Trigger hotkeys and/or simulate trades based on probability thresholds and price moves.

### 3.2 Model loading and standardization

`load_model` in `realtime_inference.py`:

- Loads the bundle and reconstructs the TCN with the same architecture as in training.
- Extracts:
  - `feature_columns`
  - `scaler.means` / `scaler.stds` (per‑column)
  - `sequence_len`, `target_columns` and hyperparameters.
- For each feature row, `standardize` replicates the training logic:
  - `(x - mean) / std`
  - replace NaN/Inf
  - clip to a configurable range (`ALPHA_STANDARDIZE_CLIP`, default `10.0`), analogous to training `standardize_clip`.

### 3.3 Feature stream handling

The realtime feature stream comes from the Rust `realtime` binary and is a JSONL file with rows like:

- `{"resolution": "fast", "features": {...}, "start_timestamp_ns": ..., "end_timestamp_ns": ..., ...}`

`FeatureTail` in `realtime_inference.py`:

- Tails the features file (or reads from start in replay mode).
- `parse_feature_line` validates JSON and the presence of `features` and resolution.

The inference loop:

- Standardizes each feature row using the bundle’s `means`/`stds` and the `feature_columns` ordering.
- Fills a deque of length `sequence_len`; once full, stacks into `[seq_len, channels]`, transposes to `[channels, seq_len]`, and runs the TCN.
- Interprets logits as tri‑class probabilities via softmax.

### 3.4 Live trade triggers

In live mode (no replay), the script can emit Windows hotkeys for up/down entries:

- `trade_threshold`: minimum probability to trigger a trade in a given direction.
- `trigger_target`: which training target to watch if multiple targets exist.
- `trigger_cooldown`: minimum seconds between triggers in the same direction.
- `entry_min_price_move` / `entry_price_feature`: optional price‑change gating between subsequent entries in the same direction.

These triggers are based solely on model probabilities and price features; they do **not** inspect labels during deployment.

---

## 4. Offline Evaluation (Trade‑Level)

### 4.1 High‑level idea

Offline trade evaluation answers: *“If we had run the current model with a certain threshold/TP/SL on these historical days, what win rate and PnL would we have seen?”*

It deliberately mirrors the **deployment** stack, not the training stack:

- Uses raw CSVs from `data/eval/*.csv`.
- Runs them through the same Rust realtime preprocessor used in live trading.
- Runs the same `deployment/realtime_inference.py` in replay mode (`--replay-existing`).
- Simulates trades based on probability thresholds, TP/SL and cooldown logic.

### 4.2 Offline evaluation runner

Entry point: `evaluation/realtime_eval.py` (invoked via `npm run evaluation`):

1. **Build concatenated replay log**
   - `build_concatenated_log` merges all `data/eval/*.csv` files into a single `eval_replay.csv` ordered stream.
2. **Run Rust realtime preprocessor**
   - Calls the `realtime` Rust binary (building it if needed) with:
     - `--resolutions fast,mid,slow`
     - `--source eval_replay.csv`
     - `--emit <tmp_features.jsonl>`
     - `--norm-state-dir ...` pointing at normalization snapshots.
   - This produces a features JSONL file identical in structure to live trading.
3. **Run Python inference in replay mode**
   - Invokes `deployment/realtime_inference.py` with:
     - `--model-bundle runs/models/best_model.pt`
     - `--features <tmp_features.jsonl>`
     - `--resolutions fast mid slow`
     - `--replay-existing` (read from start, not tail)
     - arguments for:
       - `--eval-log-jsonl` (per‑tick logs)
       - `--eval-summary-json` (aggregated summary)
       - `--eval-threshold-sweep` (list of probability thresholds to sweep)
       - `--eval-take-profit`, `--eval-stop-loss` (in price units of the chosen price feature)
       - plus any extra args from `npm run evaluation` (`--trade-threshold`, `--entry-min-price-move`, etc.).

### 4.3 Trade simulation logic (offline)

When `--replay-existing` is set and `trade_threshold` is provided, `realtime_inference.py`:

- Keeps track of open trades (direction, entry price, timestamps, etc.).
- For each tick where the buffer is warm and probabilities are available:
  - Aggregates threshold sweep statistics for various probability thresholds.
  - For the configured `trade_threshold`:
    - Decides whether an up/down trade is allowed (cooldown + single‑open constraints using data timestamps).
    - Applies an optional `entry_min_price_move` filter using `entry_price_feature` (e.g. `mid_close_price`).
    - If a trade is opened, records entry information and increments `hotkey_trades` counters in eval stats.
  - On subsequent ticks, measures price move relative to entry and closes trades when TP/SL is hit:
    - `eval_take_profit`, `eval_stop_loss` are interpreted as raw units of the price feature.
    - Computes `pips_move`, result (`win`/`loss`), and duration.

At the end, it writes an **evaluation summary** JSON under `runs/evaluation/` containing:

- row counts and resolutions coverage
- threshold‑sweep trigger counts
- trade counts (wins, losses, up/down trades)
- aggregate net pips and win rate
- per‑hour statistics

Importantly, **this offline eval never looks at labels**; it only uses prices and model probabilities to simulate trades.

---

## 5. Offline Evaluation (Label‑Level, per‑day)

To understand how the model behaves on the **exact same days** used for offline trade eval, but in terms of classification performance, there is an additional script:

- `evaluation/offline_class_eval.py`

This script:

1. Reads the stems of all files in `data/eval/*.csv` (e.g. `20240610`, `20240625`, …).
2. Reuses the **cached training features and labels** for those stems from `runs/cache/stream`:
   - Uses `StreamingFileEntry` and `MultiResolutionSequenceDataset` with the same `sequence_len` and resolutions as training.
3. Loads the TCN from the same model bundle (`runs/models/best_model.pt`) and runs it on those days.
4. Computes and logs:
   - **Global tri‑class accuracy** across all eval days.
   - **Global move‑vs‑flat AUC** (same definition as in training: `UP+DOWN` vs `FLAT`).
   - **Per‑day accuracy and move‑vs‑flat AUC**.

This gives a direct answer to: *“On the evaluation days, how good is the model at predicting the labels it was actually trained on?”* independent of any trade‑level TP/SL or threshold choice.

Run it from repo root, for example:

- `python evaluation/offline_class_eval.py`
- `python evaluation/offline_class_eval.py --model-bundle runs/models/best_model.pt --log-level DEBUG`

If there is no cache for a given eval day (because it wasn’t included in training), that day is skipped and a warning is logged.

---

## 6. How Training, Evaluation and Deployment Differ (and Connect)

### 6.1 Shared components

All three stages share:

- The same **feature definitions** (`FEATURE_SET_COLUMNS` and `feature_columns` ordering).
- The same **standardization scheme** (means/stds and clipping):
  - training uses global stats from train entries;
  - deployment/offline eval reuses those stats from the bundle’s `scaler`.
- The same **model architecture** and parameters:
  - encoded in the bundle’s `training.model` metadata.
- The same **tri‑class label semantics** for training and label‑level eval.

### 6.2 Key differences

1. **Data source & distribution**
   - Training:
     - uses preprocessed parquet under `data/preprocessed`, split into train/val/test by date.
   - Offline trade eval:
     - uses raw CSV days under `data/eval`, which may be a different set of days and market regimes.
   - Label‑level offline eval:
     - uses cached training labels for the eval stems (if those stems were included in preprocessing/training cache).

2. **Objective / metric**
   - Training & label‑level eval:
     - optimize/report classification metrics (tri‑class accuracy and move‑vs‑flat AUC).
   - Offline trade eval:
     - reports **trade‑level** metrics: win rate, net pips, expectancy, threshold‑sweep trigger counts, etc.
     - these depend heavily on the choices of `trade_threshold`, `eval_take_profit`, `eval_stop_loss`, cooldown, and price feature, not just model discrimination.

3. **Use of labels**
   - Training & in‑training trade simulation:
     - use walk‑forward labels computed from price moves to define ground truth.
   - Deployment & offline trade eval:
     - never inspect labels; they operate purely on price and predicted probabilities.
   - Label‑level offline eval:
     - uses labels, but only to compute metrics—it doesn’t simulate the exact TP/SL path on real prices like the trade simulator does.

4. **Temporal treatment and horizons**
   - Training labels are defined via `derive_triclass_targets` with `target_ticks`, `stop_ticks`, and optional `label_lookahead_bars`.
   - Offline trade eval uses TP/SL distances in raw price units and may have different implicit horizons (`eval_take_profit`, `eval_stop_loss`, and the speed of the replayed market).

### 6.3 Interpreting discrepancies

Because of these differences, it is expected that:

- A model with **moderate classification accuracy / AUC** (e.g. ~55% accuracy, AUC > 0.5) may still show a **near 50–50 win rate** in trade‑level offline eval, depending on the chosen thresholds and TP/SL.
- Label‑level offline eval (`offline_class_eval.py`) is the best tool to check whether the model’s **signal generalizes** to the eval days independently of trade settings.
- Offline trade eval (`evaluation/realtime_eval.py` + `deployment/realtime_inference.py`) is the best tool to test specific **trading rules** (thresholds, TP/SL, gating) using the model’s probabilities.

---

## 7. Practical Workflow

When iterating on the system, a typical loop is:

1. **Preprocess & train**
   - Ensure parquet data exists under `data/preprocessed` (via Rust preprocessing).
   - Run `npm run training` to train a TCN and export `runs/models/best_model.pt`.
2. **Check training/validation metrics**
   - Inspect logs for classification metrics and in‑training simulated trade metrics.
3. **Check label‑level performance on eval days**
   - Run `python evaluation/offline_class_eval.py`.
   - Confirm whether accuracy/AUC on eval days is similar to training/test.
4. **Check trade‑level performance on eval days**
   - Run `npm run evaluation` (or directly `python evaluation/realtime_eval.py` with desired args).
   - Inspect `runs/evaluation/offline_eval_*.json` summaries for win rate, net pips, and threshold sweep behavior.
5. **Adjust targets and thresholds**
   - If label‑level metrics are good but trade‑level metrics are weak, adjust:
     - the mapping from labels to trades (which target, thresholds, TP/SL, cooldown, gating).
   - If label‑level metrics are also weak, revisit:
     - feature set, label definition (`label_lookahead_bars`, `target_ticks`/`stop_ticks`), model capacity and regularization.

This separation of concerns—classification vs trade simulation, training vs deployment data—helps pinpoint whether issues come from the **model**, the **data/labels**, or the **trade mapping**. 

