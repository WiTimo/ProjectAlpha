import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, roc_auc_score

from src.evaluation.metrics import generate_epoch_report
from src.evaluation.trade_simulator import PredictionRecord, TradeSimulator
from src.definitions import DOWN_CLASS_INDEX, FLAT_CLASS_INDEX, UP_CLASS_INDEX

def train_epoch(model, loader, optimizer, device, class_weights, config):
    model.train()
    total_loss = 0
    steps = 0
    max_grad = config['training'].get('max_gradient_norm', 0)
    
    # Progress bar for training
    pbar = tqdm(loader, desc="Training", leave=False, ncols=100)
    
    for batch in pbar:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits = model(x) 
        target = y[:, 0] # Primary target
        
        loss = F.cross_entropy(logits, target, weight=class_weights, reduction="none").mean()
        loss.backward()
        
        if max_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            
        optimizer.step()
        total_loss += loss.item()
        steps += 1
        
        # Update pbar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / max(steps, 1)

def evaluate_and_log(
    model,
    loader,
    device,
    epoch,
    train_loss,
    class_weights,
    config,
    trade_simulator: TradeSimulator | None = None,
    logistic_model = None,
):
    model.eval()
    total_loss = 0
    val_count = 0
    steps = 0
    
    all_probs = []
    all_targets = []
    logistic_probs = []
    logistic_auc = None
    prediction_records: list[PredictionRecord] = []
    
    sweep_cfg = config.get("trade_simulation", {}).get("threshold_sweep", [])
    min_trades = int(config.get("trade_simulation", {}).get("min_trades", 0))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False, ncols=100):
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
                meta = None
            x, y = x.to(device), y.to(device)
            logits = model(x)
            target = y[:, 0]
            
            # Compute validation loss
            loss = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
            total_loss += loss.sum().item()
            val_count += loss.numel()
            steps += 1
            
            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()
            targets_np = target.cpu().numpy()
            all_probs.append(probs_np)
            all_targets.append(targets_np)

            logistic_batch = None
            if logistic_model is not None:
                last_step = x[:, :, -1].detach().cpu().numpy().reshape(probs_np.shape[0], -1)
                pos_idx = getattr(logistic_model, "positive_index", None)
                if pos_idx is None and hasattr(logistic_model, "classes_"):
                    try:
                        pos_idx = int(np.where(logistic_model.classes_ == 1)[0][0])
                    except Exception:
                        pos_idx = -1
                if pos_idx is None or pos_idx < 0 or pos_idx >= logistic_model.classes_.shape[0]:
                    pos_idx = 1 if logistic_model.classes_.shape[0] > 1 else 0
                logits_lr = logistic_model.predict_proba(last_step)
                logistic_batch = logits_lr[:, pos_idx]
                logistic_probs.append(logistic_batch)
            
            if meta is not None:
                meta_np = meta.cpu().numpy()
                for idx in range(meta_np.shape[0]):
                    move_prob = float(probs_np[idx, DOWN_CLASS_INDEX] + probs_np[idx, UP_CLASS_INDEX])
                    record = PredictionRecord(
                        entry_idx=int(meta_np[idx, 0]),
                        target_idx=int(meta_np[idx, 1]),
                        label=int(targets_np[idx]),
                        move_prob=move_prob,
                        up_prob=float(probs_np[idx, UP_CLASS_INDEX]),
                        down_prob=float(probs_np[idx, DOWN_CLASS_INDEX]),
                        logistic_move_prob=float(logistic_batch[idx]) if logistic_batch is not None else None,
                    )
                    prediction_records.append(record)
            
    val_loss = total_loss / max(val_count, 1)
    logging.info(
        "Validation loss components | total_loss_sum=%.4f, sample_count=%d, mean=%.6f",
        total_loss,
        val_count,
        val_loss,
    )
    
    if not all_targets:
        logging.warning("Validation set empty!")
        return 0.0, None
        
    probs_concat = np.concatenate(all_probs)
    targets_concat = np.concatenate(all_targets)
    logistic_metrics = None
    logistic_trade_summary = None

    move_probs = probs_concat[:, DOWN_CLASS_INDEX] + probs_concat[:, UP_CLASS_INDEX]
    move_targets = (targets_concat != FLAT_CLASS_INDEX).astype(int)

    if logistic_probs:
        logistic_concat = np.concatenate(logistic_probs)
        log_preds = (logistic_concat >= 0.5).astype(int)
        log_acc = accuracy_score(move_targets, log_preds)
        try:
            log_auc = roc_auc_score(move_targets, logistic_concat)
        except ValueError:
            log_auc = 0.5
        logistic_metrics = {"accuracy": log_acc, "auc": log_auc}
        logistic_auc = log_auc

        try:
            corr = float(np.corrcoef(logistic_concat, move_probs)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        logistic_metrics["corr_with_tcn"] = corr
        if corr < -0.1:
            logging.warning(
                "Negative correlation between TCN and logistic outputs (corr=%.3f). Check label polarity/sequence alignment.",
                corr,
            )
    else:
        logistic_concat = None
        logistic_metrics = None

    trade_summary = {
        "entries": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "blocked": 0.0,
        "skipped": 0.0,
        "price_mismatches": 0.0,
        "entry_rate": 0.0,
        "net_ticks": 0.0,
        "expectancy": 0.0,
        "avg_duration_bars": 0.0,
        "population": float(len(prediction_records)),
    }

    if trade_simulator and prediction_records:
        trade_summary = trade_simulator.simulate(prediction_records, use_logistic=False)
    if trade_simulator and logistic_metrics and prediction_records:
        logistic_trade_summary = trade_simulator.simulate(prediction_records, use_logistic=True)

    sweep_result = None
    if trade_simulator and prediction_records and sweep_cfg:
        sweep_metrics = []
        for thr in sweep_cfg:
            summary = trade_simulator.simulate(
                prediction_records, use_logistic=False, threshold=float(thr)
            )
            summary["threshold"] = float(thr)
            sweep_metrics.append(summary)

        if sweep_metrics:
            sweep_metrics.sort(key=lambda item: item["expectancy"], reverse=True)
            above_min = [m for m in sweep_metrics if m["entries"] >= min_trades]
            sweep_result = {
                "min_trades": float(min_trades),
                "best": above_min[0] if above_min else None,
            }
    
    # Generate the rich report
    report_str, auc = generate_epoch_report(
        epoch, 
        train_loss, 
        val_loss, 
        probs_concat, 
        targets_concat, 
        config,
        trade_summary,
        sweep_result,
        logistic_metrics,
        logistic_trade_summary,
    )
    
    # Print the report directly to stdout/log
    print(report_str)
    
    return auc, logistic_auc
