"""Phase 1 (Group A) validation script.

Loads feature Parquet files emitted by the Rust preprocessing pipeline, runs the
QA steps defined in IMPLEMENTATION.md, standardizes features using train-only
statistics, and trains a tiny Temporal Convolutional Network on the resulting
sequences.  Metrics and sanity checks are logged to stdout.
"""

from __future__ import annotations

import argparse
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FEATURE_COLUMNS = [
    "mid_return_bar",
    "spread_ticks",
    "imbalance_best",
    "trade_volume_sum_rel",
    "trade_count_log",
    "rv_log",
]

LABEL_MAP = {1: 1.0, 0: 0.0, -1: np.nan}
MIN_STD = 1e-9


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_len: int):
        if len(data) != len(targets):
            raise ValueError("Feature and target lengths differ")
        if len(data) < seq_len:
            raise ValueError("Not enough samples for the requested sequence length")
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx : idx + self.seq_len]
        target = self.targets[idx + self.seq_len - 1]
        # rearrange to (channels, seq_len) for Conv1d
        return window.transpose(0, 1), target


class TemporalBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                channels_in,
                channels_out,
                kernel_size=kernel,
                dilation=dilation,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels_out,
                channels_out,
                kernel_size=kernel,
                dilation=dilation,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(channels_in, channels_out, kernel_size=1)
            if channels_in != channels_out
            else nn.Identity()
        )
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, channels, seq)
        out = self.net(x)
        # crop to match input length
        out = out[:, :, : x.size(2)]
        return out + self.downsample(x)


class TinyTCN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden: int = 32,
        levels: int = 2,
        kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        channels_in = num_features
        for level in range(levels):
            dilation = 2**level
            block = TemporalBlock(channels_in, hidden, kernel, dilation, dropout)
            layers.append(block)
            channels_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        return self.classifier(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 validation")
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Root directory with resolution subfolders",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        required=True,
        help="Directory containing per-file label Parquets",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="fast",
        choices=["fast", "mid", "slow"],
        help="Resolution folder to load",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=4,
        help="Maximum number of Parquet files to load per split",
    )
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=32,
        help="Number of bars per training window",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability inside TemporalBlocks",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay to curb overfitting",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience measured in epochs",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=5e-4,
        help="Minimum validation AUC improvement required to reset patience",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip logistic-regression baseline evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device override",
    )
    return parser.parse_args()


def discover_feature_files(
    feature_root: Path, resolution: str, limit: int
) -> List[Path]:
    resolution_dir = feature_root / resolution
    files = sorted(resolution_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files found under {resolution_dir}")
    return files[:limit] if limit > 0 else files


def load_feature_frames(feature_files: Iterable[Path]) -> List[pd.DataFrame]:
    frames = []
    for path in feature_files:
        table = pq.read_table(path)
        df = table.to_pandas()
        df["source_file"] = path.stem
        frames.append(df)
        logging.info("Loaded %s with %d bars", path.name, len(df))
    return frames


def load_labels(label_root: Path, stems: Iterable[str]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for stem in stems:
        label_path = label_root / f"{stem}.parquet"
        if not label_path.exists():
            logging.warning("Missing label file for %s", stem)
            continue
        table = pq.read_table(label_path, columns=["timestamp_ns", "outcome"])
        df = table.to_pandas()
        df["binary"] = df["outcome"].map(LABEL_MAP)
        df.dropna(subset=["binary"], inplace=True)
        result[stem] = df
        logging.info("Loaded labels for %s -> %d events", stem, len(df))
    return result


def assign_labels_to_bars(features: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    starts = features["start_timestamp_ns"].to_numpy()
    ends = features["end_timestamp_ns"].to_numpy()
    label_times = labels["timestamp_ns"].to_numpy()
    target_values = labels["binary"].to_numpy()
    bar_indices = np.searchsorted(starts, label_times, side="right") - 1
    valid = (
        (bar_indices >= 0)
        & (bar_indices < len(starts))
        & (label_times < ends[bar_indices])
    )
    aggregates: Dict[int, List[float]] = {}
    for bar_idx, value in zip(bar_indices[valid], target_values[valid]):
        aggregates.setdefault(int(bar_idx), []).append(float(value))
    bar_labels = np.full(len(features), np.nan)
    for bar_idx, vals in aggregates.items():
        bar_labels[bar_idx] = 1.0 if float(np.mean(vals)) >= 0.5 else 0.0
    return pd.Series(bar_labels, index=features.index, name="target")


def combine_feature_label_frames(
    feature_frames: List[pd.DataFrame], label_frames: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    combined: List[pd.DataFrame] = []
    for frame in feature_frames:
        stem = frame["source_file"].iloc[0]
        labels = label_frames.get(stem)
        if labels is None:
            logging.warning("Skipping %s – no labels available", stem)
            continue
        frame = frame.copy()
        frame["target"] = assign_labels_to_bars(frame, labels)
        before = len(frame)
        frame.dropna(subset=["target"], inplace=True)
        logging.info(
            "Aligned %s -> kept %d/%d bars with labels", stem, len(frame), before
        )
        if not frame.empty:
            combined.append(frame)
    if not combined:
        raise RuntimeError("No feature rows had matching labels")
    df = pd.concat(combined, ignore_index=True)
    df.sort_values("start_timestamp_ns", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def data_quality_checks(df: pd.DataFrame, feature_cols: List[str]) -> None:
    logging.info("Running data quality checks on %d rows", len(df))
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.any():
        logging.warning("NaNs detected:\n%s", nan_counts[nan_counts > 0])
    else:
        logging.info("No NaNs detected in feature columns")

    finite_mask = np.isfinite(df[feature_cols].to_numpy())
    if not finite_mask.all():
        logging.warning("Infinities detected in feature matrix")
    else:
        logging.info("All feature values are finite")

    desc = df[feature_cols].describe(percentiles=[0.5, 0.95]).transpose()
    logging.info(
        "Feature ranges (min / median / 95th / max):\n%s",
        desc[["min", "50%", "95%", "max"]],
    )


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> DatasetSplits:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return DatasetSplits(
        train=df.iloc[:train_end].copy(),
        val=df.iloc[train_end:val_end].copy(),
        test=df.iloc[val_end:].copy(),
        feature_cols=FEATURE_COLUMNS,
    )


def standardize_splits(
    splits: DatasetSplits,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    train, val, test = splits.train, splits.val, splits.test
    cols = splits.feature_cols
    stds = train[cols].std().fillna(0.0)
    keep_cols = stds[stds > MIN_STD].index.tolist()
    dropped = [c for c in cols if c not in keep_cols]
    if dropped:
        logging.warning(
            "Dropping near-constant feature(s): %s",
            ", ".join(dropped),
        )
    if not keep_cols:
        raise ValueError("All feature columns were filtered out")

    cols = keep_cols
    splits.feature_cols = cols
    means = train[cols].mean()
    stds = train[cols].std().replace(0.0, 1.0)
    logging.info("Train means:\n%s", means)
    logging.info("Train stds:\n%s", stds)
    for split in (train, val, test):
        split.loc[:, cols] = (split[cols] - means) / stds
    return (
        train[cols].to_numpy(),
        val[cols].to_numpy(),
        test[cols].to_numpy(),
        train["target"].to_numpy(),
        val["target"].to_numpy(),
        test["target"].to_numpy(),
        cols,
    )


def build_dataloaders(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SequenceDataset(train_x, train_y, seq_len)
    val_ds = SequenceDataset(val_x, val_y, seq_len)
    test_ds = SequenceDataset(test_x, test_y, seq_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def run_logistic_baseline(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
) -> None:
    logging.info("Running logistic-regression baseline for reference")
    try:
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(train_x, train_y)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.warning("Logistic regression baseline failed: %s", exc)
        return

    for name, features, targets in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        probs = clf.predict_proba(features)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(preds, targets.astype(int))
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = float("nan")
        logging.info("LogReg %s: acc=%.3f auc=%.3f", name, acc, auc)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    preds: List[float] = []
    targets: List[float] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds.extend(probs)
            targets.extend(batch_y.cpu().numpy().ravel())
    if not preds:
        return float("nan"), float("nan")
    preds_arr = np.asarray(preds)
    targets_arr = np.asarray(targets)
    acc = accuracy_score((preds_arr >= 0.5).astype(int), targets_arr.astype(int))
    try:
        auc = roc_auc_score(targets_arr, preds_arr)
    except ValueError:
        auc = float("nan")
    return acc, auc


def train_model(
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    num_features: int,
    epochs: int,
    lr: float,
    device: torch.device,
    dropout: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    train_loader, val_loader, _ = loaders
    model = TinyTCN(num_features, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: List[Dict[str, float]] = []
    best_auc = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = max(patience, 0)
    patience_left = patience
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        val_acc, val_auc = evaluate(model, val_loader, device)
        epoch_loss = running_loss / max(batches, 1)
        logging.info(
            "Epoch %d: loss=%.4f val_acc=%.3f val_auc=%.3f",
            epoch,
            epoch_loss,
            val_acc,
            val_auc,
        )
        history.append(
            {"epoch": epoch, "loss": epoch_loss, "val_acc": val_acc, "val_auc": val_auc}
        )
        metric = val_auc if math.isfinite(val_auc) else float("-inf")
        if metric - best_auc >= min_delta:
            best_auc = metric
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            if patience > 0:
                patience_left -= 1
                if patience_left <= 0:
                    logging.info(
                        "Early stopping at epoch %d (best val_auc=%.3f)",
                        epoch,
                        best_auc,
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
    criterion = nn.BCEWithLogitsLoss()
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        val_acc, val_auc = evaluate(model, val_loader, device)
        epoch_loss = running_loss / max(batches, 1)
        logging.info(
            "Epoch %d: loss=%.4f val_acc=%.3f val_auc=%.3f",
            epoch,
            epoch_loss,
            val_acc,
            val_auc,
        )
        history.append(
            {"epoch": epoch, "loss": epoch_loss, "val_acc": val_acc, "val_auc": val_auc}
        )
    return model, history


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    logging.info("Starting Phase 1 validation with args: %s", vars(args))

    feature_files = discover_feature_files(
        args.feature_root, args.resolution, args.limit_files
    )
    feature_frames = load_feature_frames(feature_files)
    label_frames = load_labels(args.label_root, [f.stem for f in feature_files])
    dataset = combine_feature_label_frames(feature_frames, label_frames)
    data_quality_checks(dataset, FEATURE_COLUMNS)

    splits = chronological_split(dataset)
    arrays = standardize_splits(splits)
    train_x, val_x, test_x, train_y, val_y, test_y, active_cols = arrays
    logging.info(
        "Active feature columns (%d): %s",
        len(active_cols),
        ", ".join(active_cols),
    )
    if not args.skip_baseline:
        run_logistic_baseline(train_x, val_x, test_x, train_y, val_y, test_y)

    loaders = build_dataloaders(
        train_x,
        val_x,
        test_x,
        train_y,
        val_y,
        test_y,
        seq_len=args.sequence_len,
        batch_size=args.batch_size,
    )

    device = torch.device(args.device)
    model, _ = train_model(
        loaders,
        len(active_cols),
        args.epochs,
        args.learning_rate,
        device,
        args.dropout,
        args.weight_decay,
        args.patience,
        args.min_delta,
    )

    train_acc, train_auc = evaluate(model, loaders[0], device)
    val_acc, val_auc = evaluate(model, loaders[1], device)
    test_acc, test_auc = evaluate(model, loaders[2], device)
    logging.info(
        "Final metrics -> train: acc=%.3f auc=%.3f | val: acc=%.3f auc=%.3f | test: acc=%.3f auc=%.3f",
        train_acc,
        train_auc,
        val_acc,
        val_auc,
        test_acc,
        test_auc,
    )


if __name__ == "__main__":
    main()
