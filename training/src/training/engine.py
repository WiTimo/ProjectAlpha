import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, roc_auc_score

from src.evaluation.metrics import generate_epoch_report
from src.evaluation.trade_simulator import PredictionRecord, TradeSimulator

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
        
        loss = F.cross_entropy(logits, target, weight=class_weights)
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
    steps = 0
    
    all_probs = []
    all_targets = []
    logistic_probs = []
    prediction_records: list[PredictionRecord] = []
    
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
            loss = F.cross_entropy(logits, target, weight=class_weights)
            total_loss += loss.item()
            steps += 1
            
            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()
            targets_np = target.cpu().numpy()
            all_probs.append(probs_np)
            all_targets.append(targets_np)

            logistic_batch = None
            if logistic_model is not None:
                last_step = x[:, :, -1].detach().cpu().numpy().reshape(probs_np.shape[0], -1)
                logistic_batch = logistic_model.predict_proba(last_step)[:, 1]
                logistic_probs.append(logistic_batch)
            
            if meta is not None:
                meta_np = meta.cpu().numpy()
                for idx in range(meta_np.shape[0]):
                    record = PredictionRecord(
                        entry_idx=int(meta_np[idx, 0]),
                        target_idx=int(meta_np[idx, 1]),
                        label=int(targets_np[idx]),
                        tcn_prob=float(probs_np[idx, 1]),
                        logistic_prob=float(logistic_batch[idx]) if logistic_batch is not None else None,
                    )
                    prediction_records.append(record)
            
    val_loss = total_loss / max(steps, 1)
    
    if not all_targets:
        logging.warning("Validation set empty!")
        return 0.0
        
    probs_concat = np.concatenate(all_probs)
    targets_concat = np.concatenate(all_targets)
    logistic_metrics = None
    logistic_trade_summary = None

    if logistic_probs:
        logistic_concat = np.concatenate(logistic_probs)
        log_preds = (logistic_concat >= 0.5).astype(int)
        log_acc = accuracy_score(targets_concat, log_preds)
        try:
            log_auc = roc_auc_score(targets_concat, logistic_concat)
        except ValueError:
            log_auc = 0.5
        logistic_metrics = {"accuracy": log_acc, "auc": log_auc}
    else:
        logistic_concat = None

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
        trade_summary = trade_simulator.simulate(prediction_records, prob_field="tcn_prob")
    if trade_simulator and logistic_metrics and prediction_records:
        logistic_trade_summary = trade_simulator.simulate(prediction_records, prob_field="logistic_prob")
    
    # Generate the rich report
    report_str, auc = generate_epoch_report(
        epoch, 
        train_loss, 
        val_loss, 
        probs_concat, 
        targets_concat, 
        config,
        trade_summary,
        logistic_metrics,
        logistic_trade_summary,
    )
    
    # Print the report directly to stdout/log
    print(report_str)
    
    return auc