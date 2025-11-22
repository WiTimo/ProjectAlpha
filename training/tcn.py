"""Phase 1–5 (Groups A–E) validation script.

Loads feature Parquet files emitted by the Rust preprocessing pipeline, runs the
QA steps defined in IMPLEMENTATION.md, standardizes features using train-only
statistics, and trains a tiny Temporal Convolutional Network on the resulting
sequences. Metrics and sanity checks are logged to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import textwrap
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
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
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

GROUP_D_COLUMNS = [
    "limit_add_bid_volume_rel",
    "limit_add_ask_volume_rel",
    "limit_cancel_bid_volume_rel",
    "limit_cancel_ask_volume_rel",
    "limit_of_imbalance",
]

OFI_COLUMNS = [
    "ofi_bid",
    "ofi_ask",
    "ofi_net_log",
]

AGGRESSOR_COLUMNS = [
    "buy_trade_volume_rel",
    "sell_trade_volume_rel",
    "trade_imbalance_ratio",
    "avg_buy_dist_to_ask",
    "avg_sell_dist_to_bid",
    "has_buy_trade",
    "has_sell_trade",
]

PRESENCE_COLUMNS = [
    "bid_level_2_present",
    "bid_level_3_present",
    "ask_level_2_present",
    "ask_level_3_present",
]

PHASE4_FEATURE_COLUMNS = (
    CORE_FEATURE_COLUMNS
    + GROUP_D_COLUMNS
    + LEVEL_OFFSETS_COLUMNS
    + LEVEL_SIZE_COLUMNS
    + PRESENCE_COLUMNS
)

PHASE5_FEATURE_COLUMNS = PHASE4_FEATURE_COLUMNS + OFI_COLUMNS + AGGRESSOR_COLUMNS

FEATURE_SET_COLUMNS = {
    "phase4": PHASE4_FEATURE_COLUMNS,
    "phase5": PHASE5_FEATURE_COLUMNS,
}

PRICE_COLUMNS = ["mid_close_price", "mid_high_price", "mid_low_price"]

SUPPORTED_RESOLUTIONS = ("fast", "mid", "slow")

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
    "buy_trade_volume_rel",
    "sell_trade_volume_rel",
    "ofi_bid",
    "ofi_ask",
]

WINSOR_LOWER = 0.001
WINSOR_UPPER = 0.999

TARGET_CLASS_VALUES = np.array([-1, 0, 1], dtype=np.int8)
TARGET_VALUE_TO_INDEX = {
    int(value): idx for idx, value in enumerate(TARGET_CLASS_VALUES)
}
NUM_TARGET_CLASSES = int(TARGET_CLASS_VALUES.size)
DOWN_CLASS_INDEX = TARGET_VALUE_TO_INDEX[-1]
FLAT_CLASS_INDEX = TARGET_VALUE_TO_INDEX[0]
UP_CLASS_INDEX = TARGET_VALUE_TO_INDEX[1]

MIN_STD = 1e-6


def resolve_target_column_name(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned.startswith("target_"):
        return cleaned
    if cleaned.startswith("t") and cleaned[1:].isdigit():
        return f"target_{cleaned}"
    if cleaned.isdigit():
        return f"target_t{cleaned}"
    if cleaned.startswith("t"):
        cleaned = cleaned[1:]
    return f"target_{cleaned}"


@dataclass
class TargetConfig:
    column: str
    weight: float = 1.0
    class_weights: torch.Tensor = field(
        default_factory=lambda: torch.ones(NUM_TARGET_CLASSES, dtype=torch.float32)
    )


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    target_columns: List[str]
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class LabelTable:
    frame: pd.DataFrame
    outcome_cols: List[str]


# -------------------------------------------------------
# Dataset + stats helpers
# -------------------------------------------------------


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_len: int):
        if len(data) != len(targets):
            raise ValueError("Feature and target lengths differ")
        if len(data) < seq_len:
            raise ValueError("Not enough samples for the requested sequence length")
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
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


class DilatedTCN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden: int = 64,
        layers: int = 4,
        stacks: int = 1,
        kernel: int = 5,
        dilation_base: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("tcn-layers must be >= 1")
        if stacks < 1:
            raise ValueError("tcn-stacks must be >= 1")
        if kernel < 2:
            raise ValueError("tcn-kernel must be >= 2")
        if dilation_base < 1:
            raise ValueError("tcn-dilation-base must be >= 1")

        blocks: List[nn.Module] = []
        channels_in = num_features
        for _stack in range(stacks):
            for level in range(layers):
                dilation = dilation_base**level
                block = TemporalBlock(channels_in, hidden, kernel, dilation, dropout)
                blocks.append(block)
                channels_in = hidden

        self.tcn = nn.Sequential(*blocks)
        self.project = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        features = self.project(features)
        return self.head(features).squeeze(-1)


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
        "--resolutions",
        type=str,
        nargs="+",
        choices=sorted(SUPPORTED_RESOLUTIONS),
        help=(
            "Optional list of resolutions to combine (first entry is the base alignment);"
            " overrides --resolution when supplied"
        ),
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="phase5",
        choices=sorted(FEATURE_SET_COLUMNS.keys()),
        help="Which feature group to train (phase4 excludes OFI/aggressor additions)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="t40",
        help="Target horizon identifier (e.g. 20, t40, target_t60)",
    )
    primary_group = parser.add_mutually_exclusive_group()
    primary_group.add_argument(
        "--primary-only",
        dest="primary_only",
        action="store_true",
        help="Force training to use only --target even if --targets is supplied",
    )
    primary_group.add_argument(
        "--allow-multi-targets",
        dest="primary_only",
        action="store_false",
        help="Honor the explicit --targets list instead of collapsing to --target",
    )
    parser.set_defaults(primary_only=True)
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
        default=5e-4,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability inside TemporalBlocks",
    )
    parser.add_argument(
        "--tcn-hidden",
        type=int,
        default=48,
        help="Number of hidden channels in each dilated block",
    )
    parser.add_argument(
        "--tcn-layers",
        type=int,
        default=2,
        help="Number of dilated layers per stack",
    )
    parser.add_argument(
        "--tcn-stacks",
        type=int,
        default=1,
        help="Number of stacks (repetitions) of the dilation schedule",
    )
    parser.add_argument(
        "--tcn-kernel",
        type=int,
        default=5,
        help="Kernel size for the causal convolutions",
    )
    parser.add_argument(
        "--tcn-dilation-base",
        type=int,
        default=2,
        help="Base used to increase dilation per layer (e.g. 2 => 1,2,4,8)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
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
    parser.add_argument(
        "--compare-phase4",
        action="store_true",
        help="Train an additional Phase 4 baseline (ignoring OFI/aggressor features) for comparison",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["t20", "t40", "t60", "t100"],
        help="Target horizons to train (subset of outcome_* columns without prefix)",
    )
    parser.add_argument(
        "--target-loss-weights",
        type=float,
        nargs="+",
        help="Optional per-target loss weights matching --targets order",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Exponent applied to inverse frequency class weights (0 disables weighting)",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Clip gradients to this L2 norm each step (0 disables clipping)",
    )
    parser.add_argument(
        "--logistic-threshold-min",
        type=float,
        default=0.5,
        help="Minimum probability threshold evaluated for the logistic baseline",
    )
    parser.add_argument(
        "--logistic-threshold-max",
        type=float,
        default=0.8,
        help="Maximum probability threshold evaluated for the logistic baseline",
    )
    parser.add_argument(
        "--logistic-threshold-steps",
        type=int,
        default=7,
        help="Number of evenly spaced thresholds to sweep for the logistic baseline",
    )
    parser.add_argument(
        "--logistic-threshold-metric",
        choices=["net_ticks", "win_rate", "entries"],
        default="net_ticks",
        help="Metric used to pick the best logistic threshold on the validation split",
    )
    parser.add_argument(
        "--enable-regime-training",
        action="store_true",
        help="Train separate models per volatility regime and route during inference",
    )
    parser.add_argument(
        "--regime-column",
        type=str,
        default="volatility_regime",
        help="Column used to segment the dataset into volatility regimes",
    )
    parser.add_argument(
        "--regime-values",
        type=float,
        nargs="*",
        help="Explicit list of regime values to train (defaults to all unique values in the dataset)",
    )
    parser.add_argument(
        "--min-regime-rows",
        type=int,
        default=5000,
        help="Minimum number of rows required to train a regime-specific model",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("runs") / "regimes",
        help="Directory to store trained regime model bundles",
    )
    parser.add_argument(
        "--model-export-dir",
        type=Path,
        default=Path("runs") / "models",
        help="Directory where trained single-model bundles will be saved",
    )
    return parser.parse_args()


def resolve_requested_targets(
    requested: List[str],
    primary: str,
    available: List[str],
) -> List[str]:
    resolved: List[str] = []
    available_set = set(available)
    for name in requested:
        column = resolve_target_column_name(name)
        if column not in available_set:
            logging.warning("Skipping target %s – not found in dataset", column)
            continue
        if column not in resolved:
            resolved.append(column)
    primary_column = resolve_target_column_name(primary)
    if primary_column not in available_set:
        raise ValueError(
            f"Primary target '{primary_column}' missing from available columns: {available}"
        )
    if primary_column in resolved:
        resolved.remove(primary_column)
    resolved.insert(0, primary_column)
    if not resolved:
        raise RuntimeError("No valid targets were resolved for training")
    return resolved


def resolve_resolution_plan(args: argparse.Namespace) -> List[str]:
    raw_plan = args.resolutions or [args.resolution]
    plan: List[str] = []
    for entry in raw_plan:
        normalized = entry.lower()
        if normalized not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution '{entry}'; choose from {SUPPORTED_RESOLUTIONS}"
            )
        if normalized not in plan:
            plan.append(normalized)
    return plan


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


def load_labels(label_root: Path, stems: Iterable[str]) -> Dict[str, LabelTable]:
    result: Dict[str, LabelTable] = {}
    for stem in stems:
        label_path = label_root / f"{stem}.parquet"
        if not label_path.exists():
            logging.warning("Missing label file for %s", stem)
            continue
        table = pq.read_table(label_path)
        df = table.to_pandas()
        if "timestamp_ns" not in df.columns:
            logging.warning("Label file %s missing timestamp_ns column", label_path)
            continue
        outcome_cols = [col for col in df.columns if col.startswith("outcome_")]
        if not outcome_cols:
            logging.warning("Label file %s has no outcome_* columns", label_path)
            continue
        subset_cols = ["timestamp_ns", *outcome_cols]
        subset = df[subset_cols].copy()
        result[stem] = LabelTable(frame=subset, outcome_cols=outcome_cols)
        logging.info(
            "Loaded labels for %s -> %d events (%s)",
            stem,
            len(subset),
            ", ".join(outcome_cols),
        )
    return result


# -------------------------------------------------------
# Label alignment – updated aggregation (any positive wins)
# -------------------------------------------------------


def _aggregate_bar_label(values: pd.Series) -> float:
    vals = values.to_numpy()
    if np.any(vals == 1):
        return 1.0
    if np.any(vals == -1):
        return -1.0
    return 0.0


def assign_labels_to_bars(
    features: pd.DataFrame, labels: pd.DataFrame, outcome_cols: List[str]
) -> pd.DataFrame:
    """Assign multi-class {-1,0,1} labels per target column to each bar."""

    starts = features["start_timestamp_ns"].to_numpy()
    ends = features["end_timestamp_ns"].to_numpy()
    label_times = labels["timestamp_ns"].to_numpy()

    bar_indices = np.searchsorted(starts, label_times, side="right") - 1
    valid = (
        (bar_indices >= 0)
        & (bar_indices < len(starts))
        & (label_times < ends[bar_indices])
    )

    result: Dict[str, np.ndarray] = {}
    valid_bar_idx = (
        bar_indices[valid].astype(int) if valid.any() else np.array([], dtype=int)
    )

    for col in outcome_cols:
        alias = col.replace("outcome_", "target_", 1)
        bar_labels = np.full(len(features), np.nan, dtype=float)
        if valid.any():
            target_values = labels[col].astype(float, copy=False).to_numpy()
            valid_vals = target_values[valid]
            tmp = pd.DataFrame({"bar_idx": valid_bar_idx, "val": valid_vals})
            agg = tmp.groupby("bar_idx", sort=False)["val"].apply(_aggregate_bar_label)
            bar_labels[agg.index.to_numpy()] = agg.to_numpy(dtype=float)
        result[alias] = bar_labels

    return pd.DataFrame(result, index=features.index)


def combine_feature_label_frames(
    feature_frames: List[pd.DataFrame],
    label_frames: Dict[str, LabelTable],
) -> Tuple[pd.DataFrame, List[str]]:
    combined: List[pd.DataFrame] = []
    resolved_targets: Optional[List[str]] = None
    for frame in feature_frames:
        stem = frame["source_file"].iloc[0]
        label_table = label_frames.get(stem)
        if label_table is None:
            logging.warning("Skipping %s – no labels available", stem)
            continue
        frame = frame.copy()
        targets_df = assign_labels_to_bars(
            frame, label_table.frame, label_table.outcome_cols
        )
        if resolved_targets is None:
            resolved_targets = list(targets_df.columns)
        mask = targets_df.notna().any(axis=1)
        before = len(frame)
        frame = frame.loc[mask].copy()
        targets_df = targets_df.loc[mask]
        for col in targets_df.columns:
            frame[col] = targets_df[col].to_numpy()
        logging.info(
            "Aligned %s -> kept %d/%d bars with labels",
            stem,
            len(frame),
            before,
        )
        if not frame.empty:
            combined.append(frame)
    if not combined:
        raise RuntimeError("No feature rows had matching labels")
    df = pd.concat(combined, ignore_index=True)
    df.sort_values("start_timestamp_ns", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, (resolved_targets or [])


# -------------------------------------------------------
# Target encoding + metadata
# -------------------------------------------------------


def ensure_feature_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing required feature column(s): %s" % ", ".join(sorted(missing))
        )


def encode_target_matrix(df: pd.DataFrame, target_columns: List[str]) -> np.ndarray:
    rows = len(df)
    if not target_columns:
        return np.zeros((rows, 0), dtype=np.int64)
    matrix = np.zeros((rows, len(target_columns)), dtype=np.int64)
    for idx, column in enumerate(target_columns):
        if column not in df.columns:
            raise KeyError(f"Target column '{column}' missing from dataframe")
        values = df[column].to_numpy(dtype=float, copy=False)
        if np.isnan(values).any():
            raise ValueError(f"Target column '{column}' contains NaNs after filtering")
        ints = values.astype(int)
        mask = np.isin(ints, TARGET_CLASS_VALUES)
        if not mask.all():
            invalid = ", ".join(str(v) for v in np.unique(ints[~mask]))
            raise ValueError(
                f"Unexpected label value(s) in {column}: {invalid} (expected -1, 0, 1)"
            )
        encoded = np.empty(len(ints), dtype=np.int64)
        for class_value, class_idx in TARGET_VALUE_TO_INDEX.items():
            encoded[ints == class_value] = class_idx
        matrix[:, idx] = encoded
    return matrix


def primary_positive_mask(target_matrix: np.ndarray) -> np.ndarray:
    if target_matrix.ndim != 2:
        raise ValueError("Target matrix must be 2D")
    if target_matrix.shape[1] == 0:
        return np.zeros(target_matrix.shape[0], dtype=np.int64)
    return (target_matrix[:, 0] == UP_CLASS_INDEX).astype(np.int64)


def encode_class_weights(counts: np.ndarray, power: float) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    if power <= 0.0 or counts.sum() <= 0.0:
        return torch.ones(NUM_TARGET_CLASSES, dtype=torch.float32)
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-8, None)
    weights = (1.0 / probs) ** power
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32))


def compute_target_metadata(
    df: pd.DataFrame,
    target_columns: List[str],
    target_loss_weights: Optional[Iterable[float]],
    class_weight_power: float,
) -> List[TargetConfig]:
    if not target_columns:
        raise ValueError("At least one target column is required")
    if target_loss_weights is None:
        weights = [1.0 for _ in target_columns]
    else:
        weights = list(target_loss_weights)
        if len(weights) != len(target_columns):
            raise ValueError(
                "--target-loss-weights must match number of target columns"
            )
    configs: List[TargetConfig] = []
    for idx, column in enumerate(target_columns):
        series = df[column].astype(int, copy=False)
        class_counts = np.array(
            [(series == value).sum() for value in TARGET_CLASS_VALUES],
            dtype=np.float64,
        )
        class_weights = encode_class_weights(class_counts, class_weight_power)
        logging.info(
            "Target %s distribution -> down=%d flat=%d up=%d weight=%.3f",
            column,
            int(class_counts[DOWN_CLASS_INDEX]),
            int(class_counts[FLAT_CLASS_INDEX]),
            int(class_counts[UP_CLASS_INDEX]),
            float(weights[idx]),
        )
        configs.append(
            TargetConfig(
                column=column, weight=float(weights[idx]), class_weights=class_weights
            )
        )
    return configs


def refresh_target_configs(
    configs: List[TargetConfig], train_df: pd.DataFrame, class_weight_power: float
) -> List[TargetConfig]:
    refreshed: List[TargetConfig] = []
    for cfg in configs:
        if cfg.column not in train_df.columns:
            raise KeyError(f"Target column '{cfg.column}' missing from training split")
        series = train_df[cfg.column].astype(int, copy=False)
        class_counts = np.array(
            [(series == value).sum() for value in TARGET_CLASS_VALUES],
            dtype=np.float64,
        )
        class_weights = encode_class_weights(class_counts, class_weight_power)
        refreshed.append(
            TargetConfig(
                column=cfg.column, weight=cfg.weight, class_weights=class_weights
            )
        )
        logging.info(
            "Refreshed class weights for %s -> counts=%s weights=%s",
            cfg.column,
            ", ".join(str(int(c)) for c in class_counts.tolist()),
            ", ".join(f"{w:.3f}" for w in class_weights.tolist()),
        )
    return refreshed


# -------------------------------------------------------
# QA + standardisation
# -------------------------------------------------------


def data_quality_checks(df: pd.DataFrame, feature_cols: List[str]) -> None:
    logging.info("Running data quality checks on %d rows", len(df))
    ensure_feature_columns(df, feature_cols)
    feature_df = df[feature_cols]

    nan_counts = feature_df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        preview = ", ".join(
            f"{col}={int(count)}"
            for col, count in nan_counts.sort_values(ascending=False).items()
        )
        raise ValueError(f"Feature columns contain NaNs: {preview}")

    values = feature_df.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Feature columns contain non-finite values (inf/nan)")

    key_cols = ["source_file", "start_timestamp_ns"]
    if all(col in df.columns for col in key_cols):
        dupes = int(df.duplicated(subset=key_cols).sum())
        if dupes:
            logging.warning("Detected %d duplicate bar(s) based on %s", dupes, key_cols)


def augment_with_multi_resolution_inputs(
    base_df: pd.DataFrame,
    feature_root: Path,
    extra_resolutions: Iterable[str],
    limit_files: int,
    base_feature_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    if not extra_resolutions:
        return base_df, {}, []

    augmented = base_df.copy()
    column_map: Dict[str, List[str]] = {}
    multi_cols: List[str] = []
    key_cols = ["source_file", "start_timestamp_ns"]
    if not all(col in augmented.columns for col in key_cols):
        raise KeyError(
            f"Dataset missing required key columns {key_cols} for multi-resolution merge"
        )
    stems = set(augmented["source_file"].unique())

    for resolution in extra_resolutions:
        try:
            feature_files = discover_feature_files(
                feature_root, resolution, limit_files
            )
        except FileNotFoundError:
            logging.warning(
                "Skipping resolution %s – no Parquet files found", resolution
            )
            continue
        frames = load_feature_frames(feature_files)
        if not frames:
            logging.warning(
                "Skipping resolution %s – unable to load feature frames", resolution
            )
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged = merged[merged["source_file"].isin(stems)].copy()
        if merged.empty:
            logging.warning(
                "Resolution %s has no overlapping source files with the base dataset",
                resolution,
            )
            continue
        keep_cols = [col for col in base_feature_cols if col in merged.columns]
        if not keep_cols:
            logging.warning(
                "Resolution %s lacks the requested feature columns; skipping",
                resolution,
            )
            continue
        renamed = {col: f"{col}@{resolution}" for col in keep_cols}
        subset = merged[key_cols + keep_cols].copy()
        subset.sort_values(key_cols, inplace=True)
        subset.rename(columns=renamed, inplace=True)
        augmented = augmented.merge(subset, on=key_cols, how="left")
        for base_col, alias in renamed.items():
            column_map.setdefault(base_col, []).append(alias)
            multi_cols.append(alias)

    if multi_cols:
        augmented.sort_values(key_cols, inplace=True)
        augmented.reset_index(drop=True, inplace=True)
        augmented[multi_cols] = augmented.groupby("source_file")[multi_cols].ffill()
        na_mask = augmented[multi_cols].isna().any(axis=1)
        if na_mask.any():
            count = int(na_mask.sum())
            logging.warning(
                "Dropping %d row(s) lacking complete multi-resolution features", count
            )
            augmented = augmented.loc[~na_mask].copy()
        invalid_aliases = [alias for alias in multi_cols if alias.count("@") != 1]
        if invalid_aliases:
            raise ValueError(
                "Unexpected multi-resolution column aliases: %s"
                % ", ".join(sorted(invalid_aliases))
            )
        sample = ", ".join(multi_cols[: min(8, len(multi_cols))])
        logging.info(
            "Augmented %d feature(s) with %d multi-resolution columns. Sample: %s",
            len(column_map),
            len(multi_cols),
            sample,
        )

    return augmented, column_map, multi_cols


def validate_column_integrity(df: pd.DataFrame, context: str) -> None:
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            "Duplicate column(s) detected in %s: %s"
            % (context, ", ".join(sorted(set(duplicate_cols))))
        )
    suspicious = [col for col in df.columns if col.count("@") > 1]
    if suspicious:
        logging.warning(
            "Suspicious column naming patterns observed in %s: %s",
            context,
            ", ".join(sorted(suspicious)[:10]),
        )


def expand_feature_columns(
    base_cols: List[str], multi_map: Dict[str, List[str]]
) -> List[str]:
    expanded = list(base_cols)
    for col in base_cols:
        expanded.extend(multi_map.get(col, []))
    return expanded


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


def compute_split_stats(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "hit_rate": float("nan"),
            "mean_spread": float("nan"),
            "mean_depth": float("nan"),
            "mean_volume": float("nan"),
            "hourly": pd.DataFrame(columns=["hour", "rows", "win_rate"]),
        }

    target_series = df[target_column]
    stats = {
        "rows": int(len(df)),
        "hit_rate": (target_series == 1).mean(),
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
        hourly_df = pd.DataFrame(
            {"hour": hours, "target": (target_series == 1).astype(float)}
        )
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
    pd.Series,
    pd.Series,
]:
    train, val, test = splits.train, splits.val, splits.test

    # Winsorization of heavy–tailed features
    winsorize_features(train, val, test, WINSOR_COLUMNS, WINSOR_LOWER, WINSOR_UPPER)

    cols = splits.feature_cols
    stds = pd.Series({col: float(train[col].std()) for col in cols}, index=cols).fillna(
        0.0
    )
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
    means = pd.Series({col: float(train[col].mean()) for col in cols})
    stds = pd.Series({col: float(train[col].std()) for col in cols}).replace(0.0, 1.0)
    logging.info("Train means:\n%s", means)
    logging.info("Train stds:\n%s", stds)

    for split in (train, val, test):
        split.loc[:, cols] = (split[cols] - means) / stds

    target_cols = splits.target_columns
    train_targets = encode_target_matrix(train, target_cols)
    val_targets = encode_target_matrix(val, target_cols)
    test_targets = encode_target_matrix(test, target_cols)
    return (
        train[cols].to_numpy(),
        val[cols].to_numpy(),
        test[cols].to_numpy(),
        train_targets,
        val_targets,
        test_targets,
        cols,
        means,
        stds,
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


def _format_ns_timestamp(value: int) -> str:
    return pd.to_datetime(int(value), unit="ns", utc=True).strftime("%Y-%m-%d %H:%M:%S")


def verify_split_boundaries(splits: DatasetSplits) -> None:
    required_col = "start_timestamp_ns"
    if required_col not in splits.train.columns:
        logging.warning(
            "Split boundary verification skipped – %s missing", required_col
        )
        return
    spans: Dict[str, Optional[Tuple[int, int]]] = {}
    for name, df in (
        ("train", splits.train),
        ("val", splits.val),
        ("test", splits.test),
    ):
        if df.empty:
            spans[name] = None
            logging.info("%s split is empty", name.capitalize())
            continue
        start = int(df[required_col].iloc[0])
        end = int(df[required_col].iloc[-1])
        spans[name] = (start, end)
        logging.info(
            "%s split spans %s -> %s (%d rows)",
            name.capitalize(),
            _format_ns_timestamp(start),
            _format_ns_timestamp(end),
            len(df),
        )

    def _check(order_a: str, order_b: str) -> None:
        first, second = spans.get(order_a), spans.get(order_b)
        if first is None or second is None:
            return
        if first[1] >= second[0]:
            raise ValueError(
                f"{order_a.capitalize()} and {order_b} splits overlap in time; check split parameters"
            )

    _check("train", "val")
    _check("val", "test")


def _drop_overlap(
    prev: pd.DataFrame,
    curr: pd.DataFrame,
    label: str,
    key: str = "start_timestamp_ns",
) -> pd.DataFrame:
    if prev.empty or curr.empty or key not in prev.columns or key not in curr.columns:
        return curr
    boundary = prev[key].iloc[-1]
    mask = curr[key] > boundary
    if mask.all():
        return curr
    dropped = int((~mask).sum())
    if dropped:
        logging.warning(
            "Removed %d overlapping row(s) from %s split to enforce strict ordering",
            dropped,
            label,
        )
    filtered = curr.loc[mask].copy()
    filtered.reset_index(drop=True, inplace=True)
    if filtered.empty:
        raise ValueError(
            f"Split '{label}' became empty after removing overlapping timestamps; adjust split parameters"
        )
    return filtered


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

    train_df, val_df, test_df = (
        select(train_files),
        select(val_files),
        select(test_files),
    )
    val_df = _drop_overlap(train_df, val_df, "validation")
    test_df = _drop_overlap(val_df, test_df, "test")
    return train_df, val_df, test_df


def build_splits(
    df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    val_days: int,
    test_days: int,
    feature_cols: List[str],
    target_columns: List[str],
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
    if not target_columns:
        raise ValueError("At least one target column is required for training")
    reference_target = target_columns[0]
    splits = DatasetSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_cols=feature_cols,
        target_columns=target_columns,
    )
    splits.stats = {
        "train": compute_split_stats(splits.train, reference_target),
        "val": compute_split_stats(splits.val, reference_target),
        "test": compute_split_stats(splits.test, reference_target),
    }
    return splits


# -------------------------------------------------------
# Dataloaders + baselines
# -------------------------------------------------------


def build_dataloaders(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SequenceDataset(train_x, train_targets, seq_len)
    val_ds = SequenceDataset(val_x, val_targets, seq_len)
    test_ds = SequenceDataset(test_x, test_targets, seq_len)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def build_threshold_grid(min_value: float, max_value: float, steps: int) -> np.ndarray:
    min_value = float(min_value)
    max_value = float(max_value)
    if max_value < min_value:
        min_value, max_value = max_value, min_value
    if steps <= 1 or math.isclose(min_value, max_value):
        return np.array([min_value])
    return np.linspace(min_value, max_value, steps)


def summarize_trade_simulation(
    simulation: Dict[str, Any], target_ticks: float, stop_ticks: Optional[float]
) -> Dict[str, float]:
    stop_ticks = stop_ticks if stop_ticks and stop_ticks > 0 else target_ticks
    closed = simulation["targets_hit"] + simulation["stops_hit"]
    win_rate = simulation["targets_hit"] / closed if closed > 0 else float("nan")
    net_ticks = (
        simulation["targets_hit"] * target_ticks - simulation["stops_hit"] * stop_ticks
    )
    return {
        "entries": int(simulation["entries"]),
        "entry_rate": float(simulation["entry_rate"]),
        "win_rate": float(win_rate),
        "targets_hit": int(simulation["targets_hit"]),
        "stops_hit": int(simulation["stops_hit"]),
        "avg_hold_bars": float(simulation["avg_hold_bars"]),
        "net_ticks": float(net_ticks),
        "rows": int(simulation["rows"]),
    }


def simulate_with_threshold(
    probabilities: np.ndarray,
    price_series: Dict[str, np.ndarray],
    threshold: float,
    trade_params: Dict[str, float],
) -> Dict[str, float]:
    sim = simulate_trade_entries(
        probabilities,
        price_series,
        threshold,
        trade_params["target_ticks"],
        trade_params["tick_size"],
        int(trade_params["seq_len"]),
        trade_params.get("stop_ticks"),
    )
    summary = summarize_trade_simulation(
        sim,
        trade_params["target_ticks"],
        trade_params.get("stop_ticks"),
    )
    summary["threshold"] = float(threshold)
    return summary


def select_best_threshold(
    probabilities: np.ndarray,
    price_series: Dict[str, np.ndarray],
    thresholds: np.ndarray,
    trade_params: Dict[str, float],
    metric: str,
) -> Optional[Dict[str, Any]]:
    if probabilities.size == 0 or thresholds.size == 0:
        return None
    metric_key = {
        "net_ticks": "net_ticks",
        "win_rate": "win_rate",
        "entries": "entries",
    }[metric]
    best: Optional[Dict[str, Any]] = None
    for threshold in thresholds:
        summary = simulate_with_threshold(
            probabilities, price_series, threshold, trade_params
        )
        value = summary[metric_key]
        if isinstance(value, float) and not math.isfinite(value):
            comparator = float("-inf")
        else:
            comparator = float(value)
        if best is None or comparator > best["metric_value"]:
            best = {
                "threshold": float(threshold),
                "metric_value": comparator,
                "summary": summary,
            }
    return best


def run_logistic_baseline(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    price_series: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    trade_params: Optional[Dict[str, float]] = None,
    threshold_grid: Optional[np.ndarray] = None,
    threshold_metric: str = "net_ticks",
) -> Optional[Dict[str, Any]]:
    logging.info("Running logistic-regression baseline for reference")
    if train_targets.shape[1] == 0:
        logging.warning("Baseline skipped – no target columns available")
        return None
    train_y = primary_positive_mask(train_targets).astype(int)
    val_y = primary_positive_mask(val_targets).astype(int)
    test_y = primary_positive_mask(test_targets).astype(int)
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
        return None

    warmup = 0
    if trade_params:
        warmup = max(int(trade_params.get("seq_len", 1)) - 1, 0)

    def apply_warmup(
        probs: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if warmup <= 0:
            return probs, targets
        if len(probs) <= warmup:
            logging.warning(
                "Warmup (%d) exceeds available predictions (%d); skipping split",
                warmup,
                len(probs),
            )
            return np.asarray([]), np.asarray([])
        return probs[warmup:], targets[warmup:]

    classification_metrics: Dict[str, Dict[str, float]] = {}
    probabilities_by_split: Dict[str, np.ndarray] = {}
    for name, features, targets in (
        ("train", train_x, train_y),
        ("val", val_x, val_y),
        ("test", test_x, test_y),
    ):
        probs = clf.predict_proba(features)[:, 1]
        probs = probs.astype(np.float32, copy=False)
        trimmed_probs, trimmed_targets = apply_warmup(probs, targets.astype(int))
        probabilities_by_split[name] = trimmed_probs
        if len(trimmed_probs) == 0:
            classification_metrics[name] = {"acc": float("nan"), "auc": float("nan")}
            logging.warning(
                "Skipping classification metrics for %s split – no rows after warmup",
                name,
            )
            continue
        preds = (trimmed_probs >= 0.5).astype(int)
        acc = accuracy_score(preds, trimmed_targets.astype(int))
        try:
            auc = roc_auc_score(trimmed_targets, trimmed_probs)
        except ValueError:
            auc = float("nan")
        classification_metrics[name] = {"acc": float(acc), "auc": float(auc)}
        logging.info("LogReg %s: acc=%.3f auc=%.3f", name, acc, auc)

    summary: Dict[str, Any] = {"classification": classification_metrics}
    if price_series and trade_params and threshold_grid is not None:
        val_prices = price_series.get("val")
        if val_prices is None:
            logging.warning(
                "Logistic threshold sweep skipped – missing val price series"
            )
        else:
            best = select_best_threshold(
                probabilities_by_split["val"],
                val_prices,
                threshold_grid,
                trade_params,
                threshold_metric,
            )
            if best:
                reports: Dict[str, Dict[str, float]] = {}
                for split_name in ("train", "val", "test"):
                    prices = price_series.get(split_name)
                    probs = probabilities_by_split.get(split_name)
                    if prices is None or probs is None:
                        continue
                    reports[split_name] = simulate_with_threshold(
                        probs,
                        prices,
                        best["threshold"],
                        trade_params,
                    )
                summary["trade_threshold"] = {
                    "metric": threshold_metric,
                    "threshold": best["threshold"],
                    "val_metric": best["metric_value"],
                    "reports": reports,
                }
                val_report = reports.get("val")
                if val_report:
                    logging.info(
                        "LogReg threshold sweep best=%.3f metric=%s val_net=%.2f entries=%d win=%.3f",
                        best["threshold"],
                        threshold_metric,
                        val_report.get("net_ticks", float("nan")),
                        val_report.get("entries", 0),
                        val_report.get("win_rate", float("nan")),
                    )
            else:
                logging.warning(
                    "Logistic threshold sweep skipped – unable to evaluate candidates"
                )

    return summary


# -------------------------------------------------------
# Evaluation + trade sim
# -------------------------------------------------------


def softmax_np(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    sums = np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)
    return exp / sums


def collect_model_outputs(
    model: nn.Module, loader: DataLoader, device: torch.device, num_targets: int
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            logits_batches.append(logits.detach().cpu())
            target_batches.append(batch_y.detach().cpu())
    if not logits_batches:
        empty_logits = np.zeros((0, num_targets, NUM_TARGET_CLASSES), dtype=np.float32)
        empty_targets = np.zeros((0, num_targets), dtype=np.int64)
        return empty_logits, empty_targets
    stacked_logits = torch.cat(logits_batches).numpy()
    stacked_targets = torch.cat(target_batches).numpy()
    reshaped_logits = stacked_logits.reshape(-1, num_targets, NUM_TARGET_CLASSES)
    return reshaped_logits, stacked_targets


def evaluate_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_configs: List[TargetConfig],
) -> Dict[str, Dict[str, float]]:
    num_targets = len(target_configs)
    logits, targets = collect_model_outputs(model, loader, device, num_targets)
    metrics: Dict[str, Dict[str, float]] = {}
    if logits.size == 0:
        return metrics
    probs = softmax_np(logits)
    for idx, cfg in enumerate(target_configs):
        target_vec = targets[:, idx].astype(int, copy=False)
        prob_vec = probs[:, idx, :]
        preds = prob_vec.argmax(axis=1)
        acc = accuracy_score(target_vec, preds) if len(target_vec) else float("nan")
        up_probs = prob_vec[:, UP_CLASS_INDEX]
        up_targets = (target_vec == UP_CLASS_INDEX).astype(int, copy=False)
        try:
            auc = roc_auc_score(up_targets, up_probs)
        except ValueError:
            auc = float("nan")
        metrics[cfg.column] = {"acc": float(acc), "auc": float(auc)}
    return metrics


def collect_primary_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    primary_index: int,
    num_targets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    logits, targets = collect_model_outputs(model, loader, device, num_targets)
    if logits.size == 0:
        return np.asarray([]), np.asarray([])
    probs = softmax_np(logits)
    primary_probs = probs[:, primary_index, UP_CLASS_INDEX]
    binary_targets = (targets[:, primary_index] == UP_CLASS_INDEX).astype(np.float32)
    return primary_probs, binary_targets


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
    per_target: Optional[Dict[str, Dict[str, float]]] = None,
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
    if per_target:
        for target_name, target_metrics in per_target.items():
            logging.info(
                "Results[%s][%s]: acc=%.3f auc=%.3f",
                name,
                target_name,
                target_metrics.get("acc", float("nan")),
                target_metrics.get("auc", float("nan")),
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
    weight_decay: float,
    patience: int,
    min_delta: float,
    model_cfg: Dict[str, Any],
    target_configs: List[TargetConfig],
    max_grad_norm: float,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    train_loader, val_loader, _ = loaders
    num_targets = len(target_configs)
    if num_targets == 0:
        raise ValueError("At least one target configuration is required")
    model_cfg = {**model_cfg, "output_dim": num_targets * NUM_TARGET_CLASSES}
    model = DilatedTCN(num_features=num_features, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    best_auc = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = max(patience, 0)
    patience_left = patience
    weight_tensors = [cfg.class_weights.to(device) for cfg in target_configs]
    primary_col = target_configs[0].column

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            targets = batch_y.to(device)
            logits = logits.view(targets.size(0), num_targets, NUM_TARGET_CLASSES)
            loss_terms: List[torch.Tensor] = []
            for idx, cfg in enumerate(target_configs):
                ce = F.cross_entropy(
                    logits[:, idx, :],
                    targets[:, idx],
                    weight=weight_tensors[idx],
                )
                loss_terms.append(cfg.weight * ce)
            loss = torch.stack(loss_terms).sum()
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        val_metrics = evaluate_targets(model, val_loader, device, target_configs)
        primary_metrics = val_metrics.get(primary_col, {})
        val_acc = primary_metrics.get("acc", float("nan"))
        val_auc = primary_metrics.get("auc", float("nan"))
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


def run_training_for_feature_set(
    run_label: str,
    feature_cols: List[str],
    dataset: pd.DataFrame,
    external_test_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    device: torch.device,
    enable_baseline: bool,
) -> Dict[str, Any]:
    splits = build_splits(
        dataset,
        split_mode=args.split_mode,
        train_ratio=0.7,
        val_ratio=0.15,
        val_days=args.val_days,
        test_days=args.test_days,
        feature_cols=feature_cols,
        target_columns=[cfg.column for cfg in args.target_configs],
        external_test=external_test_df,
    )
    verify_split_boundaries(splits)

    price_series = {
        "train": extract_price_series(splits.train),
        "val": extract_price_series(splits.val),
        "test": extract_price_series(splits.test),
    }
    target_configs = refresh_target_configs(
        args.target_configs, splits.train, args.class_weight_power
    )

    arrays = standardize_splits(splits)
    (
        train_x,
        val_x,
        test_x,
        train_targets,
        val_targets,
        test_targets,
        active_cols,
        means,
        stds,
    ) = arrays
    logging.info(
        "[%s] Active feature columns (%d):\n%s",
        run_label,
        len(active_cols),
        textwrap.fill(", ".join(active_cols), width=120),
    )

    logistic_summary: Optional[Dict[str, Any]] = None
    if enable_baseline and not args.skip_baseline:
        trade_params = {
            "target_ticks": float(args.trade_target_ticks),
            "stop_ticks": (
                None if args.trade_stop_ticks is None else float(args.trade_stop_ticks)
            ),
            "tick_size": float(args.instrument_tick_size),
            "seq_len": int(args.sequence_len),
        }
        threshold_grid = build_threshold_grid(
            args.logistic_threshold_min,
            args.logistic_threshold_max,
            args.logistic_threshold_steps,
        )
        logistic_summary = run_logistic_baseline(
            train_x,
            val_x,
            test_x,
            train_targets,
            val_targets,
            test_targets,
            price_series=price_series,
            trade_params=trade_params,
            threshold_grid=threshold_grid,
            threshold_metric=args.logistic_threshold_metric,
        )

    model_cfg = {
        "hidden": args.tcn_hidden,
        "layers": args.tcn_layers,
        "stacks": args.tcn_stacks,
        "kernel": args.tcn_kernel,
        "dilation_base": args.tcn_dilation_base,
        "dropout": args.dropout,
    }

    loaders = build_dataloaders(
        train_x,
        val_x,
        test_x,
        train_targets,
        val_targets,
        test_targets,
        seq_len=args.sequence_len,
        batch_size=args.batch_size,
    )

    model, _ = train_model(
        loaders,
        len(active_cols),
        args.epochs,
        args.learning_rate,
        device,
        args.weight_decay,
        args.patience,
        args.min_delta,
        model_cfg,
        target_configs,
        args.max_gradient_norm,
    )

    eval_loaders = {
        name: DataLoader(dl.dataset, batch_size=args.batch_size, shuffle=False)
        for name, dl in zip(("train", "val", "test"), loaders)
    }
    split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        name: evaluate_targets(model, loader, device, target_configs)
        for name, loader in eval_loaders.items()
    }

    num_targets = len(target_configs)
    primary_idx = 0
    primary_col = target_configs[primary_idx].column
    trade_exec: Dict[str, Dict[str, Any]] = {}
    for split_name, loader_eval in eval_loaders.items():
        probs, _ = collect_primary_probabilities(
            model,
            loader_eval,
            device,
            primary_idx,
            num_targets,
        )
        trade_exec[split_name] = simulate_trade_entries(
            probs,
            price_series[split_name],
            args.trade_threshold,
            args.trade_target_ticks,
            args.instrument_tick_size,
            args.sequence_len,
            args.trade_stop_ticks,
        )

    def primary_stats(split: str) -> Tuple[float, float]:
        metrics = split_metrics.get(split, {})
        primary = metrics.get(primary_col, {})
        return (
            float(primary.get("acc", float("nan"))),
            float(primary.get("auc", float("nan"))),
        )

    train_acc, train_auc = primary_stats("train")
    val_acc, val_auc = primary_stats("val")
    test_acc, test_auc = primary_stats("test")

    log_final_split_results(
        f"train ({run_label})",
        splits.stats.get("train"),
        train_acc,
        train_auc,
        trade_exec["train"],
        split_metrics.get("train"),
    )
    log_final_split_results(
        f"val ({run_label})",
        splits.stats.get("val"),
        val_acc,
        val_auc,
        trade_exec["val"],
        split_metrics.get("val"),
    )
    log_final_split_results(
        f"test ({run_label})",
        splits.stats.get("test"),
        test_acc,
        test_auc,
        trade_exec["test"],
        split_metrics.get("test"),
    )

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    primary_summary = {
        "train": {"acc": train_acc, "auc": train_auc},
        "val": {"acc": val_acc, "auc": val_auc},
        "test": {"acc": test_acc, "auc": test_auc},
    }

    return {
        "label": run_label,
        "active_cols": active_cols,
        "scaler": {"means": means.to_dict(), "stds": stds.to_dict()},
        "model_state_dict": cpu_state,
        "metrics": {
            split: {
                "acc": primary_summary[split]["acc"],
                "auc": primary_summary[split]["auc"],
                "per_target": split_metrics.get(split, {}),
            }
            for split in ("train", "val", "test")
        },
        "trade_exec": trade_exec,
        "logistic_baseline": logistic_summary,
    }


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif key == "target_configs" and isinstance(value, list):
            serialized[key] = [
                {"column": cfg.column, "weight": cfg.weight}
                for cfg in value
                if isinstance(cfg, TargetConfig)
            ]
        else:
            serialized[key] = value
    return serialized


def format_regime_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def save_regime_bundle(
    result: Dict[str, Any],
    args: argparse.Namespace,
    regime_value: Any,
) -> Path:
    output_dir = Path(args.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regime_str = format_regime_value(regime_value)
    filename = f"{result['label']}_regime_{regime_str}.pt"
    path = output_dir / filename
    bundle = {
        "regime_value": regime_value,
        "label": result["label"],
        "feature_set": args.feature_set,
        "target_column": args.target_column,
        "feature_columns": result["active_cols"],
        "scaler": result["scaler"],
        "state_dict": result["model_state_dict"],
        "metrics": result["metrics"],
        "training": {
            "sequence_len": args.sequence_len,
            "model": {
                "hidden": args.tcn_hidden,
                "layers": args.tcn_layers,
                "stacks": args.tcn_stacks,
                "kernel": args.tcn_kernel,
                "dilation_base": args.tcn_dilation_base,
                "dropout": args.dropout,
            },
            "optimizer": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "epochs": args.epochs,
            },
        },
        "cli_args": _serialize_args(args),
    }
    torch.save(bundle, path)
    logging.info("Saved regime %s bundle -> %s", regime_str, path)
    return path


def save_training_bundle(
    result: Dict[str, Any],
    args: argparse.Namespace,
    export_dir: Path,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result['label']}_model.pt"
    path = export_dir / filename
    bundle = {
        "label": result["label"],
        "feature_set": args.feature_set,
        "target_column": getattr(args, "target_column", None),
        "target_columns": getattr(args, "target_columns", []),
        "feature_columns": result["active_cols"],
        "scaler": result["scaler"],
        "state_dict": result["model_state_dict"],
        "metrics": result["metrics"],
        "trade_exec": result.get("trade_exec"),
        "logistic_baseline": result.get("logistic_baseline"),
        "training": {
            "sequence_len": args.sequence_len,
            "model": {
                "hidden": args.tcn_hidden,
                "layers": args.tcn_layers,
                "stacks": args.tcn_stacks,
                "kernel": args.tcn_kernel,
                "dilation_base": args.tcn_dilation_base,
                "dropout": args.dropout,
            },
            "optimizer": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "epochs": args.epochs,
            },
        },
        "cli_args": _serialize_args(args),
    }
    torch.save(bundle, path)
    logging.info("Saved model bundle -> %s", path)
    return path


def write_regime_manifest(
    regime_runs: List[Dict[str, Any]], args: argparse.Namespace, output_dir: Path
) -> None:
    manifest = {
        "target": args.target_column,
        "feature_set": args.feature_set,
        "regime_column": args.regime_column,
        "models": [],
    }
    for run in regime_runs:
        bundle_path = run.get("bundle_path")
        manifest["models"].append(
            {
                "regime_value": run["regime_value"],
                "bundle": str(Path(bundle_path).name) if bundle_path else None,
                "metrics": run["metrics"].get("test"),
                "features": run["active_cols"],
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote regime manifest -> %s", manifest_path)


def train_multi_regime_models(
    dataset: pd.DataFrame,
    external_test_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    regime_col = args.regime_column
    if regime_col not in dataset.columns:
        raise KeyError(
            f"Regime column '{regime_col}' not found in training dataset columns"
        )

    regimes: List[Any]
    if args.regime_values:
        regimes = list(args.regime_values)
    else:
        regimes = sorted(dataset[regime_col].dropna().unique().tolist())
    if not regimes:
        raise RuntimeError(
            "Unable to infer regime values; ensure the dataset includes non-null entries"
        )

    logging.info(
        "Training multi-regime models using %s values: %s",
        regime_col,
        ", ".join(format_regime_value(v) for v in regimes),
    )

    regime_runs: List[Dict[str, Any]] = []
    multi_map = getattr(args, "multi_resolution_map", {})
    feature_cols = expand_feature_columns(
        FEATURE_SET_COLUMNS[args.feature_set], multi_map
    )
    external_has_regime = (
        external_test_df is not None and regime_col in external_test_df.columns
    )

    for value in regimes:
        mask = dataset[regime_col] == value
        subset = dataset.loc[mask].copy()
        if len(subset) < args.min_regime_rows:
            logging.warning(
                "Skipping regime %s – only %d rows (< min-regime-rows=%d)",
                format_regime_value(value),
                len(subset),
                args.min_regime_rows,
            )
            continue

        ext_subset = None
        if external_has_regime:
            ext_mask = external_test_df[regime_col] == value
            regime_external = external_test_df.loc[ext_mask].copy()
            if not regime_external.empty:
                ext_subset = regime_external

        run_label = f"{args.feature_set}-regime{format_regime_value(value)}"
        logging.info(
            "[%s] rows=%d (test rows=%s)",
            run_label,
            len(subset),
            len(ext_subset) if ext_subset is not None else "n/a",
        )
        result = run_training_for_feature_set(
            run_label,
            feature_cols,
            subset,
            ext_subset,
            args,
            device,
            enable_baseline=False,
        )
        result["regime_value"] = value
        bundle_path = save_regime_bundle(result, args, value)
        result["bundle_path"] = str(bundle_path)
        regime_runs.append(result)

    if not regime_runs:
        raise RuntimeError(
            "No regime-specific models were trained – revise min-regime-rows or data availability"
        )

    output_dir = Path(args.model_output_dir)
    write_regime_manifest(regime_runs, args, output_dir)
    return regime_runs


def log_phase_comparison(baseline: Dict[str, Any], contender: Dict[str, Any]) -> None:
    base_label = baseline["label"]
    new_label = contender["label"]
    for split in ("val", "test"):
        base_metrics = baseline["metrics"].get(split, {})
        new_metrics = contender["metrics"].get(split, {})
        base_auc = base_metrics.get("auc") or float("nan")
        new_auc = new_metrics.get("auc") or float("nan")
        base_acc = base_metrics.get("acc") or float("nan")
        new_acc = new_metrics.get("acc") or float("nan")
        logging.info(
            "%s vs %s on %s: Δauc=%.4f Δacc=%.4f",
            new_label,
            base_label,
            split,
            new_auc - base_auc,
            new_acc - base_acc,
        )


# -------------------------------------------------------
# Main
# -------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    logging.info("Starting Phase 1–5 validation with args: %s", vars(args))

    resolution_plan = resolve_resolution_plan(args)
    setattr(args, "resolution_plan", resolution_plan)
    primary_resolution = resolution_plan[0]
    if len(resolution_plan) > 1:
        logging.info(
            "Using multi-resolution inputs: base=%s extras=%s",
            primary_resolution,
            ", ".join(resolution_plan[1:]),
        )
    else:
        logging.info("Training with single resolution: %s", primary_resolution)

    feature_files = discover_feature_files(
        args.feature_root, primary_resolution, args.limit_files
    )
    feature_frames = load_feature_frames(feature_files)
    label_frames = load_labels(args.label_root, [f.stem for f in feature_files])
    dataset, target_columns = combine_feature_label_frames(feature_frames, label_frames)
    if not target_columns:
        raise RuntimeError("No target columns were generated from the labels")
    primary_target = resolve_target_column_name(args.target)
    requested_targets = args.targets or [args.target]
    resolved_targets = resolve_requested_targets(
        requested_targets, args.target, target_columns
    )
    weight_map: Optional[Dict[str, float]] = None
    if args.target_loss_weights:
        if len(args.target_loss_weights) != len(resolved_targets):
            raise ValueError(
                "--target-loss-weights must match the resolved --targets list"
            )
        weight_map = {
            column: float(weight)
            for column, weight in zip(resolved_targets, args.target_loss_weights)
        }
    training_targets = resolved_targets
    if getattr(args, "primary_only", False):
        if primary_target not in training_targets:
            training_targets.insert(0, primary_target)
        training_targets = [primary_target]
        logging.info(
            "primary-only enabled – restricting training to %s", primary_target
        )
    effective_loss_weights: Optional[List[float]] = None
    if weight_map is not None:
        effective_loss_weights = [weight_map[target] for target in training_targets]
    args.target_loss_weights = effective_loss_weights
    setattr(args, "target_column", primary_target)
    setattr(args, "target_columns", training_targets)
    logging.info(
        "Available targets: %s (selected %s)",
        ", ".join(target_columns),
        ", ".join(training_targets),
    )
    requested_sets = {args.feature_set}
    if args.compare_phase4:
        requested_sets.add("phase4")
    required_columns = sorted(
        {col for name in requested_sets for col in FEATURE_SET_COLUMNS[name]}
    )
    dataset.dropna(subset=training_targets, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    if dataset.empty:
        raise RuntimeError(
            "No feature rows remain after aligning labels for the requested targets"
        )
    target_configs = compute_target_metadata(
        dataset,
        training_targets,
        args.target_loss_weights,
        args.class_weight_power,
    )
    setattr(args, "target_configs", target_configs)
    ensure_feature_columns(dataset, required_columns)
    ensure_feature_columns(dataset, PRICE_COLUMNS)
    validate_column_integrity(dataset, "training dataset (base)")
    data_quality_checks(dataset, FEATURE_SET_COLUMNS[args.feature_set])

    multi_resolution_cols: List[str] = []
    multi_resolution_map: Dict[str, List[str]] = {}
    if len(resolution_plan) > 1:
        dataset, multi_resolution_map, multi_resolution_cols = (
            augment_with_multi_resolution_inputs(
                dataset,
                args.feature_root,
                resolution_plan[1:],
                args.limit_files,
                FEATURE_SET_COLUMNS[args.feature_set],
            )
        )
        validate_column_integrity(dataset, "training dataset (multi-resolution)")
        logging.info(
            "Added %d multi-resolution feature(s)",
            len(multi_resolution_cols),
        )
    setattr(args, "multi_resolution_map", multi_resolution_map)

    external_test_df: Optional[pd.DataFrame] = None
    if args.feature_root_test is not None:
        label_root_test = args.label_root_test or args.label_root
        test_feature_files = discover_feature_files(
            args.feature_root_test, primary_resolution, args.test_limit_files
        )
        test_feature_frames = load_feature_frames(test_feature_files)
        test_label_frames = load_labels(
            label_root_test, [f.stem for f in test_feature_files]
        )
        external_test_df, ext_targets = combine_feature_label_frames(
            test_feature_frames, test_label_frames
        )
        if primary_target not in ext_targets:
            logging.warning(
                "External test set missing target %s (available: %s)",
                primary_target,
                ", ".join(ext_targets),
            )
        missing_targets = [col for col in training_targets if col not in ext_targets]
        if missing_targets:
            logging.warning(
                "External test set missing %d target(s): %s",
                len(missing_targets),
                ", ".join(missing_targets),
            )
        ext_required = [col for col in training_targets if col in ext_targets]
        external_test_df.dropna(subset=ext_required or [primary_target], inplace=True)
        external_test_df.reset_index(drop=True, inplace=True)
        if external_test_df.empty:
            logging.warning(
                "External test set has no rows after filtering by %s",
                ", ".join(ext_required or [primary_target]),
            )
        ensure_feature_columns(external_test_df, required_columns)
        ensure_feature_columns(external_test_df, PRICE_COLUMNS)
        data_quality_checks(external_test_df, FEATURE_SET_COLUMNS[args.feature_set])

        if len(resolution_plan) > 1 and not external_test_df.empty:
            ext_root = args.feature_root_test or args.feature_root
            (
                external_test_df,
                _ext_map,
                ext_multi_cols,
            ) = augment_with_multi_resolution_inputs(
                external_test_df,
                ext_root,
                resolution_plan[1:],
                args.test_limit_files or args.limit_files,
                FEATURE_SET_COLUMNS[args.feature_set],
            )
            validate_column_integrity(
                external_test_df, "external test dataset (multi-resolution)"
            )
            missing_ext = [
                col for col in multi_resolution_cols if col not in ext_multi_cols
            ]
            if missing_ext:
                logging.warning(
                    "External test set missing %d multi-resolution column(s): %s",
                    len(missing_ext),
                    ", ".join(missing_ext),
                )
                dataset.drop(
                    columns=[c for c in missing_ext if c in dataset], inplace=True
                )
                multi_resolution_cols = [
                    col for col in multi_resolution_cols if col not in missing_ext
                ]
                for base_col, pref_list in list(multi_resolution_map.items()):
                    updated = [c for c in pref_list if c not in missing_ext]
                    if updated:
                        multi_resolution_map[base_col] = updated
                    else:
                        multi_resolution_map.pop(base_col, None)

    device = torch.device(args.device)

    active_feature_cols = expand_feature_columns(
        FEATURE_SET_COLUMNS[args.feature_set], multi_resolution_map
    )

    if args.enable_regime_training:
        if args.compare_phase4:
            logging.warning(
                "compare-phase4 is ignored while enable-regime-training is active"
            )
        train_multi_regime_models(dataset, external_test_df, args, device)
        return

    baseline_run: Optional[Dict[str, Any]] = None
    if args.compare_phase4 and args.feature_set != "phase4":
        logging.info("Running Phase 4 baseline for comparison")
        baseline_feature_cols = expand_feature_columns(
            FEATURE_SET_COLUMNS["phase4"], multi_resolution_map
        )
        baseline_run = run_training_for_feature_set(
            "phase4",
            baseline_feature_cols,
            dataset,
            external_test_df,
            args,
            device,
            enable_baseline=False,
        )
    elif args.compare_phase4:
        logging.info(
            "compare-phase4 requested but feature-set already phase4; skipping duplicate run"
        )

    primary_run = run_training_for_feature_set(
        args.feature_set,
        active_feature_cols,
        dataset,
        external_test_df,
        args,
        device,
        enable_baseline=True,
    )

    export_dir = Path(args.model_export_dir)
    save_training_bundle(primary_run, args, export_dir)
    if baseline_run is not None:
        save_training_bundle(baseline_run, args, export_dir)

    if baseline_run is not None and args.feature_set != "phase4":
        log_phase_comparison(baseline_run, primary_run)


if __name__ == "__main__":
    main()
