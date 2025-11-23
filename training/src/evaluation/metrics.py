from typing import Dict, Optional

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def get_class_weights(count_dict: np.ndarray, power: float) -> torch.Tensor:
    total = count_dict.sum()
    if total == 0:
        return torch.ones(len(count_dict))
    w = total / (len(count_dict) * np.maximum(count_dict, 1))
    w = w ** power
    return torch.from_numpy(w.astype(np.float32))

def generate_epoch_report(
    epoch: int,
    train_loss: float,
    val_loss: float,
    probs: np.ndarray,
    targets: np.ndarray,
    config: dict,
    trade_summary: Dict[str, float],
    trade_sweep: Optional[Dict[str, object]],
    logistic_metrics: Optional[Dict[str, float]],
    logistic_trade_summary: Optional[Dict[str, float]],
) -> str:
    """
    Generates a formatted ASCII report for the epoch.
    """
    # 1. Prediction Statistics
    # Focus on the "UP" class (Index 1)
    up_probs = probs[:, 1]
    pred_mean = np.mean(up_probs)
    pred_std = np.std(up_probs)
    pred_max = np.max(up_probs)

    # 3. Classification Metrics (Global)
    hard_preds = (up_probs >= 0.5).astype(int)
    acc = accuracy_score(targets, hard_preds)
    try:
        auc = roc_auc_score(targets, up_probs)
    except ValueError:
        auc = 0.5

    # 4. Construct Report
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f" EPOCH {epoch} REPORT ")
    lines.append(f"{'='*80}")
    
    # Model Health
    overfit_ratio = val_loss / (train_loss + 1e-6)
    health_status = "HEALTHY" if 0.9 < overfit_ratio < 1.1 else ("OVERFITTING" if overfit_ratio > 1.1 else "UNDERFITTING")
    
    lines.append(f" [MODEL HEALTH] Status: {health_status}")
    lines.append(f" Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Ratio: {overfit_ratio:.2f}")
    lines.append(f" Confidence: {pred_mean:.4f} +/- {pred_std:.4f} (Max: {pred_max:.4f})")
    if pred_std < 0.05:
        lines.append(f" WARNING: Model is outputting constant probabilities (collapsed).")

    # Metrics
    lines.append(f" [CLASSIFICATION] (Validation)")
    acc_line = f" Accuracy:   {acc*100:.2f}%"
    if logistic_metrics:
        acc_line += f" | LogReg: {logistic_metrics['accuracy']*100:.2f}%"
    lines.append(acc_line)

    auc_line = f" ROC AUC:    {auc:.4f}"
    if logistic_metrics:
        auc_line += f" | LogReg: {logistic_metrics['auc']:.4f}"
    lines.append(auc_line)
    if logistic_metrics and "corr_with_tcn" in logistic_metrics:
        lines.append(f" Prob Corr (TCN vs LR): {logistic_metrics['corr_with_tcn']:.3f}")

    threshold = config['trade_simulation']['threshold']
    target_ticks = config['trade_simulation']['target_ticks']
    stop_ticks = config['trade_simulation']['stop_ticks']

    lines.append(f"\n [TRADE SIMULATION] (Threshold > {threshold:.2f} | Target {target_ticks}t / Stop {stop_ticks}t)")
    lines.append(
        " Entries:    {entries:.0f} ({entry_rate:.2f}% of {population:.0f} eligible)".format(**trade_summary)
    )
    lines.append(
        " Wins: {wins:.0f} | Losses: {losses:.0f} | Net: {net_ticks:+.1f} ticks | Expectancy: {expectancy:+.2f}".format(
            **trade_summary
        )
    )
    lines.append(
        " Blocked: {blocked:.0f} | Skipped: {skipped:.0f} | Avg Hold: {avg_duration_bars:.2f} bars | Mismatches: {price_mismatches:.0f}".format(
            **trade_summary
        )
    )

    if logistic_trade_summary:
        lines.append("\n [LOGISTIC REGRESSION] Trade Benchmark")
        lines.append(
            " Entries: {entries:.0f} ({entry_rate:.2f}% of {population:.0f} eligible)".format(
                **logistic_trade_summary
            )
        )
        lines.append(
            " Wins: {wins:.0f} | Losses: {losses:.0f} | Net: {net_ticks:+.1f} ticks | Expectancy: {expectancy:+.2f}".format(
                **logistic_trade_summary
            )
        )
        lines.append(
            " Blocked: {blocked:.0f} | Skipped: {skipped:.0f} | Avg Hold: {avg_duration_bars:.2f} bars".format(
                **logistic_trade_summary
            )
        )

    if trade_sweep and trade_sweep.get("best"):
        best = trade_sweep["best"]
        min_trades = trade_sweep.get("min_trades", 0)
        lines.append("\n [THRESHOLD SWEEP] (diagnostic)")
        lines.append(
            f" Best Thr {best['threshold']:.2f}: Entries {best['entries']:.0f} | Exp {best['expectancy']:+.2f} | Entry Rate {best['entry_rate']:.2f}%"
        )
        if min_trades and best["entries"] < min_trades:
            lines.append(f" NOTE: Best threshold below min trade target ({int(min_trades)}), results likely noisy.")

    lines.append(f"{'='*80}\n")
    
    return "\n".join(lines), auc
