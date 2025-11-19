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

# Optional presence flags – level 1 is always 1.0, so we only use 2/3.
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
    "buy_trade_volume_rel",
    "sell_trade_volume_rel",
    "ofi_bid",
    "ofi_ask",
]

WINSOR_LOWER = 0.001
WINSOR_UPPER = 0.999


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
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_column: str = "target_t40"


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
            nn.Linear(hidden, 1),
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
        "--tcn-hidden",
        type=int,
        default=64,
        help="Number of hidden channels in each dilated block",
    )
    parser.add_argument(
        "--tcn-layers",
        type=int,
        default=4,
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
    parser.add_argument(
        "--compare-phase4",
        action="store_true",
        help="Train an additional Phase 4 baseline (ignoring OFI/aggressor features) for comparison",
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


def assign_labels_to_bars(
    features: pd.DataFrame, labels: pd.DataFrame, outcome_cols: List[str]
) -> pd.DataFrame:
    """Assign binary labels per target column to each bar."""

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
            target_values = labels[col].map(LABEL_MAP).to_numpy()
            valid_vals = target_values[valid].astype(float, copy=False)
            tmp = pd.DataFrame({"bar_idx": valid_bar_idx, "val": valid_vals})
            agg = tmp.groupby("bar_idx", sort=False)["val"].max()
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


def load_resolution_feature_table(
    feature_root: Path, resolution: str, limit_files: int
) -> pd.DataFrame:
    feature_files = discover_feature_files(feature_root, resolution, limit_files)
    frames = load_feature_frames(feature_files)
    if not frames:
        raise RuntimeError(
            f"No feature frames were loaded for resolution '{resolution}'"
        )
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError(f"Resolution '{resolution}' produced an empty feature table")
    df.sort_values(["source_file", "start_timestamp_ns"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def augment_with_multi_resolution_inputs(
    dataset: pd.DataFrame,
    feature_root: Path,
    resolutions: List[str],
    limit_files: int,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    if not resolutions:
        return dataset, {}, []

    if "source_file" not in dataset.columns:
        raise KeyError("source_file column required for multi-resolution joins")

    augmented = dataset.copy()
    augmented.sort_values(["source_file", "start_timestamp_ns"], inplace=True)
    augmented.reset_index(drop=True, inplace=True)
    available_sources = set(augmented["source_file"].unique())
    multi_cols: List[str] = []
    column_map: Dict[str, List[str]] = {col: [] for col in feature_cols}

    for resolution in resolutions:
        other_df = load_resolution_feature_table(feature_root, resolution, limit_files)
        other_df = other_df[other_df["source_file"].isin(available_sources)].copy()
        if other_df.empty:
            raise RuntimeError(
                f"Resolution '{resolution}' has no overlapping source files with the base dataset"
            )

        ensure_feature_columns(other_df, feature_cols)
        for meta_col in ("start_timestamp_ns", "end_timestamp_ns"):
            if meta_col not in other_df.columns:
                raise KeyError(
                    f"Resolution '{resolution}' missing required column {meta_col}"
                )

        prefix = resolution
        rename_map = {col: f"{prefix}__{col}" for col in feature_cols}
        subset = other_df[
            ["source_file", "start_timestamp_ns", "end_timestamp_ns", *feature_cols]
        ].rename(
            columns={
                "start_timestamp_ns": f"{prefix}__start_timestamp_ns",
                "end_timestamp_ns": f"{prefix}__end_timestamp_ns",
                **rename_map,
            }
        )

        start_col = f"{prefix}__start_timestamp_ns"
        end_col = f"{prefix}__end_timestamp_ns"
        subset.sort_values(["source_file", start_col], inplace=True)
        merged = pd.merge_asof(
            augmented,
            subset,
            left_on="start_timestamp_ns",
            right_on=start_col,
            by="source_file",
            direction="backward",
        )

        coverage_mask = (
            merged[end_col].notna()
            & (merged["start_timestamp_ns"] >= merged[start_col])
            & (merged["start_timestamp_ns"] < merged[end_col])
        )
        missing_rows = int((~coverage_mask).sum())
        prefixed_cols = list(rename_map.values())
        for base_col, pref_col in rename_map.items():
            column_map.setdefault(base_col, []).append(pref_col)
        if missing_rows:
            logging.warning(
                "[%s] Multi-resolution coverage missing for %d rows; leaving NaNs",
                resolution,
                missing_rows,
            )
            merged.loc[~coverage_mask, prefixed_cols] = np.nan
        else:
            logging.info(
                "[%s] Multi-resolution coverage aligned for all rows", resolution
            )

        merged.drop(columns=[start_col, end_col], inplace=True)
        augmented = merged
        multi_cols.extend(prefixed_cols)

    if multi_cols:
        augmented.sort_values(["source_file", "start_timestamp_ns"], inplace=True)
        for col in multi_cols:
            augmented[col] = augmented.groupby("source_file")[col].ffill()
        na_mask = augmented[multi_cols].isna().any(axis=1)
        if na_mask.any():
            count = int(na_mask.sum())
            logging.warning(
                "Dropping %d row(s) lacking complete multi-resolution features", count
            )
            augmented = augmented.loc[~na_mask].copy()
        augmented.reset_index(drop=True, inplace=True)

    return augmented, column_map, multi_cols


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

    stats = {
        "rows": int(len(df)),
        "hit_rate": df[target_column].mean(),
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
        hourly_df = pd.DataFrame({"hour": hours, "target": df[target_column]})
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

    target_col = splits.target_column
    return (
        train[cols].to_numpy(),
        val[cols].to_numpy(),
        test[cols].to_numpy(),
        train[target_col].to_numpy(),
        val[target_col].to_numpy(),
        test[target_col].to_numpy(),
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
    feature_cols: List[str],
    target_column: str,
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
        feature_cols=feature_cols,
        target_column=target_column,
    )
    splits.stats = {
        "train": summarize_split("train", splits.train, target_column),
        "val": summarize_split("val", splits.val, target_column),
        "test": summarize_split("test", splits.test, target_column),
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
    weight_decay: float,
    patience: int,
    min_delta: float,
    model_cfg: Dict[str, Any],
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    train_loader, val_loader, _ = loaders
    model = DilatedTCN(num_features=num_features, **model_cfg).to(device)
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
        target_column=args.target_column,
        external_test=external_test_df,
    )

    price_series = {
        "train": extract_price_series(splits.train),
        "val": extract_price_series(splits.val),
        "test": extract_price_series(splits.test),
    }

    arrays = standardize_splits(splits)
    (
        train_x,
        val_x,
        test_x,
        train_y,
        val_y,
        test_y,
        active_cols,
        means,
        stds,
    ) = arrays
    logging.info(
        "[%s] Active feature columns (%d): %s",
        run_label,
        len(active_cols),
        ", ".join(active_cols),
    )

    if enable_baseline and not args.skip_baseline:
        run_logistic_baseline(train_x, val_x, test_x, train_y, val_y, test_y)

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
        train_y,
        val_y,
        test_y,
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

    log_final_split_results(
        f"train ({run_label})",
        splits.stats.get("train"),
        train_acc,
        train_auc,
        trade_exec["train"],
    )
    log_final_split_results(
        f"val ({run_label})",
        splits.stats.get("val"),
        val_acc,
        val_auc,
        trade_exec["val"],
    )
    log_final_split_results(
        f"test ({run_label})",
        splits.stats.get("test"),
        test_acc,
        test_auc,
        trade_exec["test"],
    )

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "label": run_label,
        "active_cols": active_cols,
        "scaler": {"means": means.to_dict(), "stds": stds.to_dict()},
        "model_state_dict": cpu_state,
        "metrics": {
            "train": {"acc": train_acc, "auc": train_auc},
            "val": {"acc": val_acc, "auc": val_auc},
            "test": {"acc": test_acc, "auc": test_auc},
        },
        "trade_exec": trade_exec,
    }


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
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
    target_column = resolve_target_column_name(args.target)
    if target_column not in target_columns:
        raise ValueError(
            f"Requested target '{target_column}' not available; choose from {target_columns}"
        )
    setattr(args, "target_column", target_column)
    logging.info(
        "Available targets: %s (selected %s)",
        ", ".join(target_columns),
        target_column,
    )
    requested_sets = {args.feature_set}
    if args.compare_phase4:
        requested_sets.add("phase4")
    required_columns = sorted(
        {col for name in requested_sets for col in FEATURE_SET_COLUMNS[name]}
    )
    dataset.dropna(subset=[target_column], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    if dataset.empty:
        raise RuntimeError(
            f"No feature rows remain after aligning labels for {target_column}"
        )
    ensure_feature_columns(dataset, required_columns)
    ensure_feature_columns(dataset, PRICE_COLUMNS)
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
        if target_column not in ext_targets:
            logging.warning(
                "External test set missing target %s (available: %s)",
                target_column,
                ", ".join(ext_targets),
            )
        external_test_df.dropna(subset=[target_column], inplace=True)
        external_test_df.reset_index(drop=True, inplace=True)
        if external_test_df.empty:
            logging.warning(
                "External test set has no rows after filtering by %s",
                target_column,
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

    if baseline_run is not None and args.feature_set != "phase4":
        log_phase_comparison(baseline_run, primary_run)


if __name__ == "__main__":
    main()
