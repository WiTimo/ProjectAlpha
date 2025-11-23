# Phase 1-4 – Python Validation

# Scope

This directory hosts the lightweight validation pipeline for the Group A (core), Group B (top-of-book
plus depth), Group C (per-level structure), and Group D (order-flow adds/cancels) feature sets. It loads the Parquet outputs emitted by the Rust preprocessing crate, performs
the QA checks from `IMPLEMENTATION.md`, standardizes the features using train-only statistics (after a
train-driven winsorization pass that calms extreme depth spikes), and trains a tiny Temporal Convolutional
Network (TCN) as a smoke test. Metrics are logged for the validation and test segments so regressions can
be spotted quickly.

## Quick start

```bash
cd training
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python phase1_tcn.py --feature-root ../data/preprocessed/training --label-root ../data/preprocessed/training \
  --resolution fast --limit-files 4 --sequence-len 32 --epochs 5 --split-mode days --val-days 5 --test-days 5
```

The script accepts several switches:

 `--val-days`, `--test-days`: Number of full source days to reserve when `--split-mode=days`. The train/test split is now handled exclusively via these parameters.

When `--enable-regime-training` is set, the script writes one `.pt` bundle per regime and a `manifest.json` that records
the routing metadata (regime value → bundle filename plus headline metrics). During inference, detect the latest
`volatility_regime` value, load the matching bundle, standardize with the stored scaler stats, and route the window through
the corresponding TCN.

### Multi-resolution windows

Supplying `--resolutions fast mid slow` (or any ordered subset) activates the multi-resolution merger baked into the trainer.
The first entry becomes the base timeline (typically `fast`), while slower bars are joined via `merge_asof` on
`source_file`/`start_timestamp_ns`, forward-filled within each day, and prefixed (e.g. `mid__spread_ticks`, `slow__rv_log`).
Feature selection respects the active `--feature-set`, so Phase 4 runs never see Phase 5-only columns even if they exist in
the Parquets. External test sets automatically receive the same augmentation to guarantee a consistent schema.

Handy npm wrappers are available:

```bash
npm run training:multi         # dilated TCN on fast+mid+slow inputs
npm run training:multi:regimes # multi-resolution regime-specific training with bundle export
```

## Memory-aware chunk training

Training directly from ~80 GB of Parquet blows past 16 GB of RAM if you try to keep everything resident, yet replaying the
entire corpus every epoch saturates even fast SSDs. The updated `training/tcn.py` therefore works in RAM-sized chunks:

- `cache_streaming_files` still writes normalized `.npy` shards per resolution, but the trainer now groups those shards into
  ~1–2 GB chunks (`chunking.chunk_size_gb`). Each chunk is loaded fully into RAM, transformed into sliding windows once, and
  reused for a few "local" epochs before being released.
- While the model trains on chunk _N_, background workers prefetch chunk _N+1_ (`chunking.prefetch_chunks`) so disk I/O is
  hidden behind GPU compute. With a 120 MB/s SSD, a 2 GB chunk takes ~17 seconds to stream, which is negligible compared to
  5–10 minutes of compute reuse.
- Tune `chunking.local_epochs` (default 3) to decide how hard you want to squeeze each chunk before moving on. Increasing it
  reduces disk traffic further, while lowering it increases sample diversity per global epoch.
- All chunk controls live in `training/config.yaml` under the `chunking` section. Set `chunking.enabled=false` to fall back
  to the legacy mmap-based loader if you have enough RAM or want apples-to-apples benchmarks.

## Phase-5 single-head workflow

- `python training/tcn.py` now defaults to `--primary-only`, so unless you pass `--allow-multi-targets` the run will
  focus on the main `--target` (t40), matching the review guidance to stabilize the baseline before revisiting other
  horizons.
- Default hyperparameters favor a much smaller, better-regularized network (hidden=48, layers=2, dropout=0.3,
  weight decay=1e-3) plus gradient clipping. This keeps the TCN honest relative to the elastic-net logistic baseline.
- The logistic-regression reference automatically performs a threshold sweep on the validation split (net ticks by
  default) and then reports train/val/test trade stats for the selected level, so you can ship a calibrated threshold
  without a side notebook.
- `npm run training` wraps the recommended command line (fast+mid+slow inputs, single-head model, logistic sweep) so you
  can rerun the Phase-5 evaluation with a single shortcut.

### Splitting strategy

Day-level splits reduce regime drift by ensuring each split consists of distinct trading days instead of interleaved rows.
By default the earliest `N - val_days - test_days` days become training, the next `val_days` become validation, and the latest
`test_days` become in-sample test data. The train/test split is now handled exclusively via these parameters. This mirrors the recommendation from `IMPLEMENTATION.md` to evaluate on separate sessions rather than random slices.

### Feature coverage

The current build expects (and sanity-checks) the following normalized columns emitted by preprocessing:

- Group A: `mid_return_bar`, `spread_ticks`, `imbalance_best`, `trade_volume_sum_rel`, `trade_count_log`, `rv_log`.
- Group B: `spread_change_ticks`, `mid_range_rel`, `cum_bid_size_l_rel`, `cum_ask_size_l_rel`, `imbalance_l`.
- Group C (per-level, k = 1..3): `bid_offset_level_k_ticks`, `ask_offset_level_k_ticks`, `bid_size_level_k_rel`, `ask_size_level_k_rel`.
- Group D (order-flow aggregates): `limit_add_bid_volume_rel`, `limit_add_ask_volume_rel`, `limit_cancel_bid_volume_rel`, `limit_cancel_ask_volume_rel`, `limit_of_imbalance`.

Any missing column will cause the script to abort early so depth/flow regressions surface immediately. Before
standardization, train quantiles (0.1% / 99.9%) are used to winsorize the most volatile columns including the
level size relatives and flow volumes so book spikes remain bounded. The same bounds are applied to validation/test to avoid
leakage. During training, ultra-stable columns (variance below `1e-9`) are automatically dropped; a warning is
logged so you can confirm whether the removal is expected (e.g., if a particular resolution generates flat
`rv_log`). The logistic-regression baseline still uses elastic-net regularization, making it stable as the feature
set grows.

All logging is printed to stdout so it can be captured in CI or a notebook. See `phase1_tcn.py --help` for the full list of options.
