import numpy as np
import torch
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import load_config, setup_logging
from src.definitions import FEATURE_SET_COLUMNS, NUM_TARGET_CLASSES, CROSS_RES_COLUMNS
from src.data.loader import discover_feature_files, cache_streaming_files, compute_stats, standardize_entries
from src.data.dataset import MultiResolutionSequenceDataset
from src.model.network import DilatedTCN
from src.training.engine import train_epoch, evaluate_and_log
from src.evaluation.metrics import get_class_weights
from src.evaluation.baselines import train_logistic_baseline
from src.evaluation.trade_simulator import TradeSimulator
from src.data.loader import StreamingFileEntry


def _run_alignment_diagnostics(
    loader: DataLoader,
    entries: list[StreamingFileEntry],
    resolutions: list[str],
    col_map: dict[str, list[str]],
    max_checks: int = 16,
) -> None:
    """
    Quick integrity check: ensure dataset labels match cached targets and that
    the last timestep features align with the cached feature row for the anchor.
    """
    import numpy as np

    if not entries or len(loader.dataset) == 0:
        logging.info("Alignment diagnostics skipped: empty validation set.")
        return

    mismatches = 0
    feature_mismatch = 0
    checked = 0
    for batch in loader:
        x, y, meta = batch
        x_np = x.numpy()
        y_np = y.numpy()
        meta_np = meta.numpy()

        for i in range(x_np.shape[0]):
            if checked >= max_checks:
                break
            f_idx, t_idx = int(meta_np[i, 0]), int(meta_np[i, 1])
            entry = entries[f_idx]
            cached_targets = np.load(entry.targets_path, mmap_mode="r")
            cached_label = int(cached_targets[t_idx, 0])
            if cached_label != int(y_np[i, 0]):
                mismatches += 1

            # Compare last step features against cached feature row
            start = 0
            last_step = x_np[i, :, -1]
            for res in resolutions:
                cols = col_map[res]
                end = start + len(cols)
                cached_feat = np.load(entry.feature_paths[res], mmap_mode="r")
                raw_row = cached_feat[t_idx]
                if raw_row.shape[0] == end - start:
                    diff = np.mean(np.abs(last_step[start:end] - raw_row))
                    if diff > 1e-3:
                        feature_mismatch += 1
                start = end
            checked += 1
        if checked >= max_checks:
            break

    if mismatches > 0:
        logging.warning("Alignment check: %d/%d label mismatches detected.", mismatches, checked)
    else:
        logging.info("Alignment check: labels match cached targets on %d samples.", checked)

    if feature_mismatch > 0:
        logging.warning(
            "Alignment check: %d/%d feature rows differ (post-standardization). Inspect caching/standardization.",
            feature_mismatch,
            checked,
        )
    else:
        logging.info("Alignment check: last-step features agree with cache on %d samples.", checked)

def main():
    setup_logging()
    config = load_config()
    
    device = torch.device(config['system']['device'])
    logging.info(f"Using device: {device}")

    # 1. Setup Feature Columns
    feature_set = config['data']['feature_set']
    if feature_set == "phase8":
        base_feature_set = "phase7"
    else:
        base_feature_set = feature_set

    base_cols = FEATURE_SET_COLUMNS[base_feature_set]
    resolutions = config['data']['resolutions']
    lookahead_bars = int(config['training'].get('label_lookahead_bars', 0))
    
    # Columns read from Parquet (raw) vs stored in cache (may include derived cross-resolution features).
    col_map_read: dict[str, list[str]] = {}
    col_map_store: dict[str, list[str]] = {}
    total_input_channels = 0
    for i, res in enumerate(resolutions):
        cols_read = base_cols
        if i == 0 and feature_set == "phase8":
            cols_store = base_cols + CROSS_RES_COLUMNS
        else:
            cols_store = base_cols
        col_map_read[res] = cols_read
        col_map_store[res] = cols_store
        total_input_channels += len(cols_store)

    # Flattened feature column order used for both training and realtime inference:
    # base resolution uses raw names; higher resolutions use "@{res}" suffix.
    feature_columns: list[str] = []
    for idx, res in enumerate(resolutions):
        suffix = "" if idx == 0 else f"@{res}"
        for col in col_map_store[res]:
            name = f"{col}{suffix}"
            feature_columns.append(name)

    logging.info(f"Feature Set: {feature_set} | Resolutions: {resolutions}")
    logging.info(f"Total Input Channels: {total_input_channels}")

    # 2. Discover Files
    feature_root = Path(config['paths']['feature_root'])
    primary_res = resolutions[0]
    
    files = discover_feature_files(feature_root, primary_res, config['data']['limit_files'])
    stems = [f.stem for f in files]
    logging.info(f"Found {len(stems)} source files.")

    # 3. Cache & Preprocess (Streaming)
    entries = cache_streaming_files(
        stems,
        config,
        resolutions,
        col_map_read,
        lookahead_bars=lookahead_bars if lookahead_bars > 0 else None,
        store_cols_map=col_map_store,
    )
    logging.info(f"Successfully cached {len(entries)} files.")
    
    if not entries:
        logging.error("No valid entries found after caching. Exiting.")
        sys.exit(1)

    label_hist = np.zeros(NUM_TARGET_CLASSES, dtype=np.int64)
    for entry in entries:
        label_hist += entry.target_counts.sum(axis=0)
    total_labels = int(label_hist.sum())
    if total_labels > 0:
        up_ratio = (label_hist[-1] / total_labels) * 100
        flat_ratio = (label_hist[1] / total_labels) * 100
        logging.info(
            "Label coverage: %d usable anchors | Down=%d Flat=%d Up=%d (Up %.2f%% Flat %.2f%%)",
            total_labels,
            int(label_hist[0]),
            int(label_hist[1]),
            int(label_hist[-1]),
            up_ratio,
            flat_ratio,
        )

    # 4. Split Train/Val/Test
    val_days = config['data']['val_days']
    test_days = config['data']['test_days']
    
    entries.sort(key=lambda x: x.start_ts)
    
    test_entries = entries[-test_days:] if test_days > 0 else []
    remaining = entries[:-test_days] if test_days > 0 else entries
    val_entries = remaining[-val_days:] if val_days > 0 else []
    train_entries = remaining[:-val_days] if val_days > 0 else remaining

    train_stems = [e.stem for e in train_entries]
    val_stems = [e.stem for e in val_entries]
    test_stems = [e.stem for e in test_entries]

    logging.info(f"Split: Train={len(train_entries)} Val={len(val_entries)} Test={len(test_entries)}")
    if train_stems:
        logging.info("Train stems: %s", ", ".join(train_stems))
    if val_stems:
        logging.info("Validation stems (used for TradeSimulator): %s", ", ".join(val_stems))
    if test_stems:
        logging.info("Test stems: %s", ", ".join(test_stems))

    # 5. Standardize
    stats = compute_stats(train_entries, resolutions, col_map_store)
    clip_value = float(config["data"].get("standardize_clip", 0.0))
    standardize_entries(entries, stats, clip_value=clip_value if clip_value > 0 else None)

    # Build per-feature means/stds aligned with feature_columns for export.
    scaler_means: dict[str, float] = {}
    scaler_stds: dict[str, float] = {}
    for idx, res in enumerate(resolutions):
        if res not in stats:
            continue
        means_res, stds_res = stats[res]
        suffix = "" if idx == 0 else f"@{res}"
        cols = col_map_store[res]
        for j, col in enumerate(cols):
            key = f"{col}{suffix}"
            scaler_means[key] = float(means_res[j])
            scaler_stds[key] = float(stds_res[j])

    # 6. Dataloaders
    seq_len = config['training']['sequence_len']
    bs = config['training']['batch_size']
    workers = config['system']['num_workers']

    train_ds = MultiResolutionSequenceDataset(train_entries, seq_len, resolutions)
    val_ds = MultiResolutionSequenceDataset(val_entries, seq_len, resolutions)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers)

    eval_cfg = config.get("evaluation", {})
    enable_logistic = bool(eval_cfg.get("enable_logistic_baseline", True))
    logistic_model = None
    if enable_logistic:
        logging.info("Logistic baseline: enabled (will be trained on validation set).")
        logistic_model = train_logistic_baseline(train_loader, config)
    else:
        logging.info("Logistic baseline: disabled via config; skipping baseline training.")
    trade_simulator = None
    if val_entries:
        trade_simulator = TradeSimulator(
            val_entries,
            threshold=float(config['trade_simulation']['threshold']),
            target_ticks=float(config['trade_simulation']['target_ticks']),
            stop_ticks=float(config['trade_simulation']['stop_ticks']),
            tick_size=float(config['trade_simulation']['tick_size']),
            lookahead_bars=lookahead_bars if lookahead_bars > 0 else None,
        )

    # 7. Model Setup
    hidden = config['model']['hidden_dim']
    layers = config['model']['layers']
    channel_sizes = [hidden] * layers
    
    model = DilatedTCN(
        num_inputs=total_input_channels,
        num_classes=NUM_TARGET_CLASSES,
        num_channels=channel_sizes,
        kernel_size=config['model']['kernel_size'],
        dropout=config['model']['dropout'],
        dilation_base=config['model'].get('dilation_base', 2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=float(config['training'].get('min_learning_rate', 1e-5)),
    )
    _run_alignment_diagnostics(val_loader, val_entries, resolutions, col_map_store)

    total_counts = np.zeros(NUM_TARGET_CLASSES)
    for e in train_entries:
        total_counts += e.target_counts.sum(axis=0)
        
    class_weights = get_class_weights(total_counts, config['training']['class_weight_power']).to(device)
    total_labels = total_counts.sum()
    up_rate = (total_counts[-1] / max(total_labels, 1)) * 100.0
    flat_rate = (total_counts[1] / max(total_labels, 1)) * 100.0
    logging.info(
        "Class Weights: %s | Train label coverage: %d samples (Up %.2f%% Flat %.2f%%)",
        class_weights.cpu().numpy(),
        int(total_labels),
        up_rate,
        flat_rate,
    )

    # 8. Training Loop
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    baseline_margin = float(config['training'].get('baseline_margin', 0.01))
    baseline_patience = int(config['training'].get('baseline_patience', 3))
    min_epochs = int(config['training'].get('min_epochs', 3))
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    baseline_auc = None
    below_baseline_epochs = 0

    # Minimal training metadata stored in the deployment bundle.
    training_metadata = {
        "sequence_len": int(seq_len),
        "model": {
            "hidden_dim": int(hidden),
            "layers": int(layers),
            "kernel_size": int(config['model']['kernel_size']),
            "dropout": float(config['model']['dropout']),
            "dilation_base": int(config['model'].get('dilation_base', 2)),
        },
        "data": {
            "feature_set": feature_set,
            "resolutions": list(resolutions),
            "train_stems": train_stems,
            "val_stems": val_stems,
            "test_stems": test_stems,
        },
        "target": config['training']['target'],
    }

    logging.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights, config)
        
        # New evaluation call
        auc, logistic_auc = evaluate_and_log(
            model,
            val_loader,
            device,
            epoch,
            train_loss,
            class_weights,
            config,
            trade_simulator=trade_simulator,
            logistic_model=logistic_model,
        )

        if logistic_auc is not None and baseline_auc is None:
            baseline_auc = logistic_auc
            logging.info(f"Logistic baseline (val) AUC reference: {baseline_auc:.4f}")

        meets_baseline = baseline_auc is None or auc >= baseline_auc - baseline_margin
        
        if auc > best_auc + config['training']['min_delta'] and meets_baseline:
            best_auc = auc
            best_epoch = epoch
            patience_counter = 0
            best_path = Path(config['paths']['model_export_dir']) / "best_model.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)

            bundle = {
                "model_state_dict": model.state_dict(),
                "feature_columns": feature_columns,
                "target_columns": [config['training']['target']],
                "training": training_metadata,
                "scaler": {
                    "means": scaler_means,
                    "stds": scaler_stds,
                },
            }
            torch.save(bundle, best_path)
            logging.info(f"--> New Best Model Saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1

        scheduler.step(auc)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} complete | Val AUC: {auc:.4f} | LR now {current_lr:.2e}")

        # Save per-epoch checkpoint so models are deployable after any epoch.
        ckpt_dir = Path(config['paths']['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        ckpt_bundle = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "feature_columns": feature_columns,
            "target_columns": [config['training']['target']],
            "training": training_metadata,
            "scaler": {
                "means": scaler_means,
                "stds": scaler_stds,
            },
            "metrics": {
                "val_auc": float(auc),
                "best_auc": float(best_auc),
                "baseline_auc": float(baseline_auc) if baseline_auc is not None else None,
            },
        }
        torch.save(ckpt_bundle, ckpt_path)
        logging.info("Checkpoint saved: %s", ckpt_path)

        if baseline_auc is not None and not meets_baseline:
            below_baseline_epochs += 1
        else:
            below_baseline_epochs = 0

        if patience_counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

        if (
            baseline_auc is not None
            and epoch >= min_epochs
            and below_baseline_epochs >= baseline_patience
            and best_auc < baseline_auc - baseline_margin
        ):
            logging.info(
                "Stopping: TCN Val AUC has stayed below logistic baseline (%.4f) for %d epochs",
                baseline_auc,
                baseline_patience,
            )
            break

    if baseline_auc is not None and best_auc < baseline_auc - baseline_margin:
        logging.info(
            "Run finished: best TCN AUC %.4f is below logistic baseline %.4f; model not promoted.",
            best_auc,
            baseline_auc,
        )
    else:
        logging.info(f"Training Complete. Best Val AUC: {best_auc:.4f} (epoch {best_epoch})")

if __name__ == "__main__":
    main()
