import numpy as np
import torch
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import load_config, setup_logging
from src.definitions import FEATURE_SET_COLUMNS, NUM_TARGET_CLASSES
from src.data.loader import discover_feature_files, cache_streaming_files, compute_stats, standardize_entries
from src.data.dataset import MultiResolutionSequenceDataset
from src.model.network import DilatedTCN
from src.training.engine import train_epoch, evaluate_and_log
from src.evaluation.metrics import get_class_weights
from src.evaluation.baselines import train_logistic_baseline
from src.evaluation.trade_simulator import TradeSimulator

def main():
    setup_logging()
    config = load_config()
    
    device = torch.device(config['system']['device'])
    logging.info(f"Using device: {device}")

    # 1. Setup Feature Columns
    feature_set = config['data']['feature_set']
    base_cols = FEATURE_SET_COLUMNS[feature_set]
    resolutions = config['data']['resolutions']
    
    col_map = {}
    total_input_channels = 0
    for i, res in enumerate(resolutions):
        col_map[res] = base_cols 
        total_input_channels += len(base_cols)

    logging.info(f"Feature Set: {feature_set} | Resolutions: {resolutions}")
    logging.info(f"Total Input Channels: {total_input_channels}")

    # 2. Discover Files
    feature_root = Path(config['paths']['feature_root'])
    primary_res = resolutions[0]
    
    files = discover_feature_files(feature_root, primary_res, config['data']['limit_files'])
    stems = [f.stem for f in files]
    logging.info(f"Found {len(stems)} source files.")

    # 3. Cache & Preprocess (Streaming)
    entries = cache_streaming_files(stems, config, resolutions, col_map)
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
        logging.info(
            "Label coverage: %d usable anchors | Down=%d Up=%d (Up %.2f%%)",
            total_labels,
            int(label_hist[0]),
            int(label_hist[-1]),
            up_ratio,
        )

    # 4. Split Train/Val/Test
    val_days = config['data']['val_days']
    test_days = config['data']['test_days']
    
    entries.sort(key=lambda x: x.start_ts)
    
    test_entries = entries[-test_days:] if test_days > 0 else []
    remaining = entries[:-test_days] if test_days > 0 else entries
    val_entries = remaining[-val_days:] if val_days > 0 else []
    train_entries = remaining[:-val_days] if val_days > 0 else remaining
    
    logging.info(f"Split: Train={len(train_entries)} Val={len(val_entries)} Test={len(test_entries)}")

    # 5. Standardize
    stats = compute_stats(train_entries, resolutions, col_map)
    standardize_entries(entries, stats)

    # 6. Dataloaders
    seq_len = config['training']['sequence_len']
    bs = config['training']['batch_size']
    workers = config['system']['num_workers']

    train_ds = MultiResolutionSequenceDataset(train_entries, seq_len, resolutions)
    val_ds = MultiResolutionSequenceDataset(val_entries, seq_len, resolutions)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers)

    logistic_model = train_logistic_baseline(train_loader, config)
    trade_simulator = None
    if val_entries:
        trade_simulator = TradeSimulator(
            val_entries,
            threshold=float(config['trade_simulation']['threshold']),
            target_ticks=float(config['trade_simulation']['target_ticks']),
            stop_ticks=float(config['trade_simulation']['stop_ticks']),
            tick_size=float(config['trade_simulation']['tick_size']),
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

    total_counts = np.zeros(NUM_TARGET_CLASSES)
    for e in train_entries:
        total_counts += e.target_counts.sum(axis=0)
        
    class_weights = get_class_weights(total_counts, config['training']['class_weight_power']).to(device)
    total_labels = total_counts.sum()
    pos_rate = (total_counts[-1] / max(total_labels, 1)) * 100.0
    logging.info(
        "Class Weights: %s | Train label coverage: %d samples (Up %.2f%%)",
        class_weights.cpu().numpy(),
        int(total_labels),
        pos_rate,
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
            path = Path(config['paths']['model_export_dir']) / "best_model.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            logging.info(f"--> New Best Model Saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1

        scheduler.step(auc)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epoch} complete | Val AUC: {auc:.4f} | LR now {current_lr:.2e}")

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
