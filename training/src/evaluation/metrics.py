from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from src.definitions import DOWN_CLASS_INDEX, FLAT_CLASS_INDEX, UP_CLASS_INDEX


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
) -> tuple[str, float]:
    """Generate a formatted ASCII report for the epoch."""
    move_probs = probs[:, DOWN_CLASS_INDEX] + probs[:, UP_CLASS_INDEX]
    pred_mean = float(np.mean(move_probs))
    pred_std = float(np.std(move_probs))
    pred_max = float(np.max(move_probs))

    hard_preds = np.argmax(probs, axis=1)
    acc = accuracy_score(targets, hard_preds)

    move_targets = (targets != FLAT_CLASS_INDEX).astype(int)
    try:
        auc = roc_auc_score(move_targets, move_probs)
    except ValueError:
        auc = 0.5

    lines: list[str] = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f" EPOCH {epoch} REPORT ")
    lines.append(f"{'=' * 80}")

    overfit_ratio = val_loss / (train_loss + 1e-6)
    health_status = (
        "HEALTHY"
        if 0.9 < overfit_ratio < 1.1
        else ("OVERFITTING" if overfit_ratio > 1.1 else "UNDERFITTING")
    )

    lines.append(f" [MODEL HEALTH] Status: {health_status}")
    lines.append(f" Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Ratio: {overfit_ratio:.2f}")
    lines.append(f" Move Prob: {pred_mean:.4f} +/- {pred_std:.4f} (Max: {pred_max:.4f})")
    if pred_std < 0.05:
        lines.append(" WARNING: Model is outputting constant move probabilities (collapsed).")

    lines.append(" [CLASSIFICATION] (Validation)")
    acc_line = f" Accuracy (3-class argmax): {acc * 100:.2f}%"
    if logistic_metrics:
        acc_line += f" | LogReg (move/flat): {logistic_metrics['accuracy'] * 100:.2f}%"
    lines.append(acc_line)

    auc_line = f" ROC AUC (move vs flat): {auc:.4f}"
    if logistic_metrics:
        auc_line += f" | LogReg: {logistic_metrics['auc']:.4f}"
    lines.append(auc_line)
    if logistic_metrics and "corr_with_tcn" in logistic_metrics:
        lines.append(f" Prob Corr (TCN vs LR): {logistic_metrics['corr_with_tcn']:.3f}")

    threshold = config["trade_simulation"]["threshold"]
    target_ticks = config["trade_simulation"]["target_ticks"]
    stop_ticks = config["trade_simulation"]["stop_ticks"]

    lines.append(
        f"\n [TRADE SIMULATION] (Move prob > {threshold:.2f} | Target {target_ticks}t / Stop {stop_ticks}t)"
    )
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

    if trade_sweep:
        min_trades = trade_sweep.get("min_trades", 0)
        lines.append("\n [THRESHOLD SWEEP] (diagnostic)")
        best = trade_sweep.get("best")
        if best:
            lines.append(
                f" Best Thr {best['threshold']:.2f}: Entries {best['entries']:.0f} | Exp {best['expectancy']:+.2f} | Entry Rate {best['entry_rate']:.2f}%"
            )
            if min_trades and best["entries"] < min_trades:
                lines.append(
                    f" NOTE: Best threshold below min trade target ({int(min_trades)}), results likely noisy."
                )
        else:
            lines.append(" No threshold met the minimum trade target; skip PnL interpretation.")

    lines.append(f"{'=' * 80}\n")

    return "\n".join(lines), auc
