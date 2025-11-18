"""Phase 1–4 (Groups A–D) validation script.

Loads feature Parquet files emitted by the Rust preprocessing pipeline, runs the
QA steps defined in IMPLEMENTATION.md, standardizes features using train-only
statistics, and trains a tiny Temporal Convolutional Network on the resulting
sequences. Metrics and sanity checks are logged to stdout.
"""

from __future__ import annotations

import argparse
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -------------------------------------------------------
# Feature configuration – updated for current dataset
# -------------------------------------------------------

LEVEL_FEATURE_COUNT = 3

LEVEL_OFFSETS_COLUMNS = [
    f"bid_offset_level_{k}_ticks" for k in range(1, LEVEL_FEATURE_COUNT + 1)
] + [f"ask_offset_level_{k}_ticks" for k in range(1, LEVEL_FEATURE_COUNT + 1)]

LEVEL_SIZE_COLUMNS = [
    f"bid_size_level_{k}_rel" for k in range(1, LEVEL_FEATURE_COUNT + 1)
] + [f"ask_size_level_{k}_rel" for k in range(1, LEVEL_FEATURE_COUNT + 1)]

CORE_FEATURE_COLUMNS = [
    "mid_return_bar",
    "spread_ticks",
    "spread_change_ticks",
    "mid_range_rel",
    "imbalance_best",
    "cum_bid_size_l_rel",
    "cum_ask_size_l_rel",
    "imbalance_l",
    "trade_volume_sum_rel",
    "trade_count_log",
    "rv_log",
]

# Group D – simple order flow
ORDERFLOW_COLUMNS = [
    "limit_add_bid_volume_rel",
    "limit_add_ask_volume_rel",
    "limit_cancel_bid_volume_rel",
    "limit_cancel_ask_volume_rel",
    "limit_of_imbalance",
]

# Optional presence flags – level 1 is always 1.0, so we only use 2/3.
PRESENCE_COLUMNS = [
    "bid_level_2_present",
    "bid_level_3_present",
    "ask_level_2_present",
    "ask_level_3_present",
]

FEATURE_COLUMNS = (
    CORE_FEATURE_COLUMNS
    + ORDERFLOW_COLUMNS
    + LEVEL_OFFSETS_COLUMNS
    + LEVEL_SIZE_COLUMNS
    + PRESENCE_COLUMNS
)

PRICE_COLUMNS = ["mid_close_price", "mid_high_price", "mid_low_price"]

# Label mapping: 1 -> positive, 0/-1 -> negative
LABEL_MAP = {1: 1.0, 0: 0.0, -1: 0.0}

MIN_STD = 1e-9

# Columns we winsorize using train quantiles
WINSOR_COLUMNS = [
    "spread_ticks",
    "spread_change_ticks",
    "mid_range_rel",
    "cum_bid_size_l_rel",
    "cum_ask_size_l_rel",
    "trade_volume_sum_rel",
    "limit_add_bid_volume_rel",
    "limit_add_ask_volume_rel",
    "limit_cancel_bid_volume_rel",
    "limit_cancel_ask_volume_rel",
    # level sizes can be heavy–tailed too:
    *LEVEL_SIZE_COLUMNS,
]

WINSOR_LOWER = 0.001
WINSOR_UPPER = 0.999


# -------------------------------------------------------
# Data containers
# -------------------------------------------------------


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# -------------------------------------------------------
# Dataset + stats helpers
# -------------------------------------------------------


def summarize_split(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    stats = compute_split_stats(df)
    logging.info(
        "Split %s -> rows=%d hit_rate=%.3f mean_spread=%.2f mean_depth=%.2f mean_volume=%.2f",
        name,
        stats["rows"],
        stats["hit_rate"],
        stats["mean_spread"],
        stats["mean_depth"],
        stats["mean_volume"],
    )
    return stats


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


# -------------------------------------------------------
# Tiny TCN model
# -------------------------------------------------------


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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        return self.classifier(features).squeeze(-1)


# -------------------------------------------------------
# CLI + file discovery
# -------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1–4 validation (TCN)")
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
        "--limit",
        type=int,
        default=0,
        help="Maximum number of Parquet files to load per split (0 = all)",
    )
    parser.add_argument(
        "--test-limit-files",
        type=int,
        default=0,
        help="Maximum number of Parquet files to load for the external test root (0 = all)",
    )
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=32,
        help="Number of bars per training window",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
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
        "--trade-threshold",
        type=float,
        default=0.55,
        help="Minimum predicted probability required to open a trade",
    )
    parser.add_argument(
        "--trade-target-ticks",
        type=float,
        default=40.0,
        help="Ticks in favor required to close a trade (10 pips ~= 40 ticks for NQ)",
    )
    parser.add_argument(
        "--trade-stop-ticks",
        type=float,
        default=None,
        help="Optional ticks against entry to treat as stop (defaults to target if omitted)",
    )
    parser.add_argument(
        "--instrument-tick-size",
        type=float,
        default=0.25,
        help="Tick size of the instrument to convert ticks into price moves",
    )
    parser.add_argument(
        "--split-mode",
        choices=["rows", "days"],
        default="days",
        help="Split by raw rows (legacy) or by full calendar days using source files",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=5,
        help="Number of full days reserved for validation when using --split-mode=days",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=5,
        help="Number of full days reserved for testing when using --split-mode=days and no external test root",
    )
    parser.add_argument(
        "--feature-root-test",
        type=Path,
        help="Optional root directory containing held-out test Parquets (mirrors --feature-root layout)",
    )
    parser.add_argument(
        "--label-root-test",
        type=Path,
        help="Optional label directory for the held-out test files (defaults to --label-root)",
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
        # event_index and anchor_price are available but not needed for alignment here
        table = pq.read_table(label_path, columns=["timestamp_ns", "outcome"])
        df = table.to_pandas()
        df["binary"] = df["outcome"].map(LABEL_MAP)
        df.dropna(subset=["binary"], inplace=True)
        result[stem] = df
        logging.info("Loaded labels for %s -> %d events", stem, len(df))
    return result


# -------------------------------------------------------
# Label alignment – updated aggregation (any positive wins)
# -------------------------------------------------------


def assign_labels_to_bars(features: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Assign a binary label to each bar.

    A bar is labeled 1 if *any* event inside that bar has outcome 1,
    otherwise 0 (if at least one non-positive event), or NaN if no
    events fall inside the bar.
    """
    starts = features["start_timestamp_ns"].to_numpy()
    ends = features["end_timestamp_ns"].to_numpy()
    label_times = labels["timestamp_ns"].to_numpy()
    target_values = labels["binary"].to_numpy()

    # Map each label time to its bar index
    bar_indices = np.searchsorted(starts, label_times, side="right") - 1
    valid = (
        (bar_indices >= 0)
        & (bar_indices < len(starts))
        & (label_times < ends[bar_indices])
    )

    bar_labels = np.full(len(features), np.nan, dtype=float)

    if valid.any():
        # Only labels that fall strictly inside a bar
        valid_bar_idx = bar_indices[valid].astype(int)
        valid_vals = target_values[valid].astype(float)

        # Aggregate: bar label = max(binary) over all events in this bar
        tmp = pd.DataFrame({"bar_idx": valid_bar_idx, "val": valid_vals})
        agg = tmp.groupby("bar_idx", sort=False)["val"].max()

        bar_labels[agg.index.to_numpy()] = agg.to_numpy(dtype=float)

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


# -------------------------------------------------------
# QA + standardisation
# -------------------------------------------------------


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


def ensure_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing expected feature columns: " + ", ".join(sorted(missing))
        )


def winsorize_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: Iterable[str],
    lower: float,
    upper: float,
) -> None:
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in columns:
        if col not in train.columns:
            continue
        q_low = train[col].quantile(lower)
        q_high = train[col].quantile(upper)
        if not np.isfinite(q_low) or not np.isfinite(q_high):
            continue
        if q_low > q_high:
            q_low, q_high = q_high, q_low
        bounds[col] = (q_low, q_high)
    if not bounds:
        return
    logging.info(
        "Winsorizing features using train quantiles (%.3f-%.3f): %s",
        lower,
        upper,
        ", ".join(sorted(bounds.keys())),
    )
    for frame in (train, val, test):
        for col, (q_low, q_high) in bounds.items():
            if col in frame.columns:
                frame.loc[:, col] = frame[col].clip(q_low, q_high)


def extract_price_series(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "mid_close": df["mid_close_price"].astype(float).to_numpy(copy=True),
        "mid_high": df["mid_high_price"].astype(float).to_numpy(copy=True),
        "mid_low": df["mid_low_price"].astype(float).to_numpy(copy=True),
    }


def compute_split_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "hit_rate": float("nan"),
            "mean_spread": float("nan"),
            "mean_depth": float("nan"),
            "mean_volume": float("nan"),
            "hourly": pd.DataFrame(columns=["hour", "rows", "win_rate"]),
        }

    stats = {
        "rows": int(len(df)),
        "hit_rate": df["target"].mean(),
        "mean_spread": df.get("spread_ticks", pd.Series(dtype=float)).mean(),
        "mean_depth": (
            df.get("cum_bid_size_l_rel", pd.Series(dtype=float))
            .add(df.get("cum_ask_size_l_rel", pd.Series(dtype=float)), fill_value=0.0)
            .mean()
        ),
        "mean_volume": df.get("trade_volume_sum_rel", pd.Series(dtype=float)).mean(),
    }

    if "start_timestamp_ns" in df.columns:
        hours = pd.to_datetime(df["start_timestamp_ns"], unit="ns", utc=True).dt.hour
        hourly_df = pd.DataFrame({"hour": hours, "target": df["target"]})
        grouped = hourly_df.groupby("hour", dropna=False).agg(
            rows=("target", "size"), win_rate=("target", "mean")
        )
        stats["hourly"] = grouped.reset_index()
    else:
        stats["hourly"] = pd.DataFrame(columns=["hour", "rows", "win_rate"])

    return stats


def format_hourly_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no hourly data)"
    display = df.copy()
    if "win_rate" in display:
        display["win_rate"] = display["win_rate"].astype(float).round(3)
    display["rows"] = display["rows"].astype(int)
    display.sort_values("hour", inplace=True)
    return display.to_string(index=False)


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

    # Winsorization of heavy–tailed features
    winsorize_features(train, val, test, WINSOR_COLUMNS, WINSOR_LOWER, WINSOR_UPPER)

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


# -------------------------------------------------------
# Train/val/test splitting
# -------------------------------------------------------


def split_by_rows(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n == 0:
        raise ValueError("Dataset is empty; cannot split")
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    val_end = min(val_end, n)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def split_by_days(
    df: pd.DataFrame, val_days: int, test_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "source_file" not in df.columns:
        raise KeyError("source_file column missing; day-based split unavailable")
    per_file = df.groupby("source_file")["start_timestamp_ns"].min().sort_values()
    stems = list(per_file.index)
    if not stems:
        raise ValueError("No source files available for splitting")
    total_required = val_days + test_days
    if len(stems) <= total_required:
        raise ValueError(
            "Not enough distinct days (%d) for val/test requirements (%d)"
            % (len(stems), total_required)
        )

    test_files = stems[-test_days:] if test_days > 0 else []
    remaining = stems[: len(stems) - test_days]
    val_files = remaining[-val_days:] if val_days > 0 else []
    train_files = remaining[: len(remaining) - val_days]
    if not train_files:
        raise ValueError("Day-based split produced empty training set")

    def select(files: List[str]) -> pd.DataFrame:
        if not files:
            return pd.DataFrame(columns=df.columns)
        subset = df[df["source_file"].isin(files)].copy()
        subset.sort_values("start_timestamp_ns", inplace=True)
        subset.reset_index(drop=True, inplace=True)
        return subset

    return select(train_files), select(val_files), select(test_files)


def build_splits(
    df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    val_days: int,
    test_days: int,
    external_test: Optional[pd.DataFrame] = None,
) -> DatasetSplits:
    if split_mode == "rows":
        train, val, test = split_by_rows(df, train_ratio, val_ratio)
    else:
        local_test_days = 0 if external_test is not None else test_days
        train, val, test = split_by_days(df, val_days, local_test_days)
    if external_test is not None:
        test = external_test.copy()
        test.sort_values("start_timestamp_ns", inplace=True)
        test.reset_index(drop=True, inplace=True)
    splits = DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_cols=FEATURE_COLUMNS,
    )
    splits.stats = {
        "train": summarize_split("train", splits.train),
        "val": summarize_split("val", splits.val),
        "test": summarize_split("test", splits.test),
    }
    return splits


# -------------------------------------------------------
# Dataloaders + baselines
# -------------------------------------------------------


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
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.25,
            C=0.5,
            n_jobs=-1,
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


# -------------------------------------------------------
# Evaluation + trade sim
# -------------------------------------------------------


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
    return float(acc), float(auc)


def collect_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(batch_probs)
            targets.append(batch_y.cpu().numpy().ravel())
    if not probs:
        return np.asarray([]), np.asarray([])
    return np.concatenate(probs), np.concatenate(targets)


def simulate_trade_entries(
    probabilities: np.ndarray,
    price_series: Dict[str, np.ndarray],
    threshold: float,
    target_ticks: float,
    tick_size: float,
    seq_len: int,
    stop_ticks: Optional[float] = None,
) -> Dict[str, Any]:
    rows = int(len(probabilities))
    target_ticks = float(target_ticks)
    if target_ticks <= 0.0:
        raise ValueError("trade target ticks must be positive")
    if tick_size <= 0.0:
        raise ValueError("tick size must be positive")
    if stop_ticks is None:
        stop_ticks = target_ticks
    stop_ticks = float(abs(stop_ticks))

    closes = price_series["mid_close"]
    highs = price_series["mid_high"]
    lows = price_series["mid_low"]
    offset = max(seq_len - 1, 0)
    available_rows = max(len(closes) - offset, 0)
    usable = min(rows, available_rows)
    if usable <= 0:
        return {
            "entries": 0,
            "rows": rows,
            "entry_rate": float("nan"),
            "threshold": threshold,
            "target_ticks": target_ticks,
            "stop_ticks": stop_ticks if stop_ticks > 0 else None,
            "tick_size": tick_size,
            "avg_hold_bars": 0.0,
            "targets_hit": 0,
            "stops_hit": 0,
            "open_trades": 0,
        }
    if usable != rows:
        logging.warning(
            "Probability/price length mismatch (prob=%d, usable=%d). Truncating to shortest.",
            rows,
            usable,
        )
    filled_entries = 0
    targets_hit = 0
    stops_hit = 0
    open_trades = 0
    hold_lengths: List[int] = []
    next_flat_price_idx = offset  # absolute price index that is safe to enter again
    price_len = len(closes)
    idx = 0
    while idx < usable:
        price_idx = offset + idx
        if price_idx >= price_len:
            break
        if price_idx < next_flat_price_idx:
            idx += 1
            continue
        prob = probabilities[idx]
        if prob < threshold:
            idx += 1
            continue
        entry_price = closes[price_idx]
        if not math.isfinite(entry_price):
            idx += 1
            continue
        filled_entries += 1
        target_price = entry_price + target_ticks * tick_size
        stop_price = (
            entry_price - stop_ticks * tick_size if stop_ticks > 0 else float("-inf")
        )
        exit_idx: Optional[int] = None
        cursor = price_idx
        while cursor < price_len:
            high = highs[cursor]
            low = lows[cursor]
            if not math.isfinite(high):
                high = closes[cursor]
            if not math.isfinite(low):
                low = closes[cursor]
            if high >= target_price:
                targets_hit += 1
                exit_idx = cursor
                break
            if stop_ticks > 0 and low <= stop_price:
                stops_hit += 1
                exit_idx = cursor
                break
            cursor += 1
        if exit_idx is None:
            open_trades += 1
            hold_lengths.append(price_len - price_idx)
            break
        hold_lengths.append(exit_idx - price_idx + 1)
        next_flat_price_idx = exit_idx + 1
        idx = max(idx + 1, next_flat_price_idx - offset)

    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    return {
        "entries": filled_entries,
        "rows": usable,
        "entry_rate": (filled_entries / usable) if usable else float("nan"),
        "threshold": threshold,
        "target_ticks": target_ticks,
        "stop_ticks": stop_ticks if stop_ticks > 0 else None,
        "tick_size": tick_size,
        "avg_hold_bars": avg_hold,
        "targets_hit": targets_hit,
        "stops_hit": stops_hit,
        "open_trades": open_trades,
    }


def log_final_split_results(
    name: str,
    stats: Optional[Dict[str, Any]],
    acc: float,
    auc: float,
    execution: Optional[Dict[str, Any]] = None,
) -> None:
    if not stats:
        logging.info(
            "Results[%s]: rows=0 win_rate=nan acc=%.3f auc=%.3f",
            name,
            acc,
            auc,
        )
        return
    exec_suffix = ""
    if execution:
        stop_ticks = execution.get("stop_ticks")
        stop_str = (
            f"{stop_ticks:.1f}" if isinstance(stop_ticks, (float, int)) else "none"
        )
        exec_suffix = (
            f" executed_trades={execution['entries']}"
            f" threshold={execution['threshold']:.2f}"
            f" target_ticks={execution['target_ticks']:.1f}"
            f" stop_ticks={stop_str}"
        )
    logging.info(
        "Results[%s]: rows=%d win_rate=%.3f acc=%.3f auc=%.3f%s",
        name,
        stats["rows"],
        stats["hit_rate"],
        acc,
        auc,
        exec_suffix,
    )
    if execution:
        logging.info(
            "Results[%s]: executed_trades=%d over %d rows (rate=%.3f) hits=%d stops=%d open=%d avg_hold=%.2f",
            name,
            execution["entries"],
            execution["rows"],
            execution["entry_rate"],
            execution["targets_hit"],
            execution["stops_hit"],
            execution["open_trades"],
            execution["avg_hold_bars"],
        )
    hourly = stats.get("hourly") if isinstance(stats, dict) else None
    if isinstance(hourly, pd.DataFrame) and not hourly.empty:
        logging.info(
            "Results[%s] hourly breakdown:\n%s",
            name,
            format_hourly_table(hourly),
        )


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------


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


# -------------------------------------------------------
# Main
# -------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    logging.info("Starting Phase 1–4 validation with args: %s", vars(args))

    feature_files = discover_feature_files(
        args.feature_root, args.resolution, args.limit_files
    )
    feature_frames = load_feature_frames(feature_files)
    label_frames = load_labels(args.label_root, [f.stem for f in feature_files])
    dataset = combine_feature_label_frames(feature_frames, label_frames)
    ensure_feature_columns(dataset, FEATURE_COLUMNS)
    ensure_feature_columns(dataset, PRICE_COLUMNS)
    data_quality_checks(dataset, FEATURE_COLUMNS)

    external_test_df: Optional[pd.DataFrame] = None
    if args.feature_root_test is not None:
        label_root_test = args.label_root_test or args.label_root
        test_feature_files = discover_feature_files(
            args.feature_root_test, args.resolution, args.test_limit_files
        )
        test_feature_frames = load_feature_frames(test_feature_files)
        test_label_frames = load_labels(
            label_root_test, [f.stem for f in test_feature_files]
        )
        external_test_df = combine_feature_label_frames(
            test_feature_frames, test_label_frames
        )
        ensure_feature_columns(external_test_df, FEATURE_COLUMNS)
        ensure_feature_columns(external_test_df, PRICE_COLUMNS)
        data_quality_checks(external_test_df, FEATURE_COLUMNS)

    splits = build_splits(
        dataset,
        split_mode=args.split_mode,
        train_ratio=0.7,
        val_ratio=0.15,
        val_days=args.val_days,
        test_days=args.test_days,
        external_test=external_test_df,
    )
    arrays = standardize_splits(splits)
    price_series = {
        "train": extract_price_series(splits.train),
        "val": extract_price_series(splits.val),
        "test": extract_price_series(splits.test),
    }
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

    eval_loaders = tuple(
        DataLoader(dl.dataset, batch_size=args.batch_size, shuffle=False)
        for dl in loaders
    )

    trade_exec: Dict[str, Dict[str, Any]] = {}
    for split_name, loader_eval in zip(("train", "val", "test"), eval_loaders):
        probs, _ = collect_probabilities(model, loader_eval, device)
        trade_exec[split_name] = simulate_trade_entries(
            probs,
            price_series[split_name],
            args.trade_threshold,
            args.trade_target_ticks,
            args.instrument_tick_size,
            args.sequence_len,
            args.trade_stop_ticks,
        )

    logging.info(
        "Final metrics -> train: acc=%.3f auc=%.3f | val: acc=%.3f auc=%.3f | test: acc=%.3f auc=%.3f",
        train_acc,
        train_auc,
        val_acc,
        val_auc,
        test_acc,
        test_auc,
    )
    log_final_split_results(
        "train", splits.stats.get("train"), train_acc, train_auc, trade_exec["train"]
    )
    log_final_split_results(
        "val", splits.stats.get("val"), val_acc, val_auc, trade_exec["val"]
    )
    log_final_split_results(
        "test", splits.stats.get("test"), test_acc, test_auc, trade_exec["test"]
    )


if __name__ == "__main__":
    main()
