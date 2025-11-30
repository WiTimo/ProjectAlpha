import logging
import gc
import heapq
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from src.definitions import MIN_STD, PRICE_COLUMNS, NUM_TARGET_CLASSES, CROSS_RES_COLUMNS

@dataclass
class StreamingFileEntry:
    stem: str
    rows: int
    start_ts: Optional[int]
    feature_paths: Dict[str, Path]
    targets_path: Path
    target_mask_path: Path
    exit_index_path: Path
    timestamp_path: Path
    price_path: Path
    feature_sums: Dict[str, np.ndarray]
    feature_sumsq: Dict[str, np.ndarray]
    target_counts: np.ndarray

def discover_feature_files(feature_root: Path, resolution: str, limit: int) -> List[Path]:
    resolution_dir = feature_root / resolution
    files = sorted(resolution_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files found under {resolution_dir}")
    return files[:limit] if limit > 0 else files

def load_streaming_frame(
    stem: str,
    feature_root: Path,
    resolution: str,
    feature_cols: List[str],
) -> Optional[pd.DataFrame]:
    feature_path = feature_root / resolution / f"{stem}.parquet"
    if not feature_path.exists():
        logging.warning(f"Missing feature file: {feature_path}")
        return None

    required_cols = ["start_timestamp_ns", *PRICE_COLUMNS, *feature_cols]
    # Remove duplicates while preserving order
    seen = set()
    ordered_cols = []
    for col in required_cols:
        if col in seen:
            continue
        ordered_cols.append(col)
        seen.add(col)

    try:
        table = pq.read_table(feature_path, columns=ordered_cols)
        frame = table.to_pandas()
    except Exception as e:
        logging.warning(f"Error reading feature file {feature_path}: {e}")
        return None

    frame.sort_values("start_timestamp_ns", inplace=True)
    missing_prices = [col for col in PRICE_COLUMNS if col not in frame.columns]
    if missing_prices:
        logging.warning(f"File {feature_path} missing price columns: {missing_prices}")
        return None
    return frame.reset_index(drop=True)

def load_resolution_slice(stem: str, root: Path, res: str, cols: List[str]) -> Optional[pd.DataFrame]:
    path = root / res / f"{stem}.parquet"
    if not path.exists():
        return None
    try:
        df = pq.read_table(path, columns=["start_timestamp_ns", *cols]).to_pandas()
        df.sort_values("start_timestamp_ns", inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Error reading slice {path}: {e}")
        return None


def derive_triclass_targets(
    frame: pd.DataFrame,
    tick_size: float,
    target_ticks: float,
    stop_ticks: float,
    lookahead_bars: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tri-class (down/flat/up) labels by walking forward until +/- target_ticks within a bounded horizon."""
    closes = frame["mid_close_price"].to_numpy(dtype=np.float64)
    highs = frame["mid_high_price"].to_numpy(dtype=np.float64)
    lows = frame["mid_low_price"].to_numpy(dtype=np.float64)

    n = len(frame)
    labels = np.full(n, 1, dtype=np.int8)  # default flat class
    valid_mask = np.zeros(n, dtype=bool)
    exit_indices = np.full(n, -1, dtype=np.int32)

    if n == 0:
        return labels, valid_mask, exit_indices

    up_delta = tick_size * target_ticks
    down_delta = tick_size * stop_ticks
    pending_up: List[Tuple[float, int]] = []
    pending_down: List[Tuple[float, int]] = []
    next_up = np.full(n, -1, dtype=np.int32)
    next_down = np.full(n, -1, dtype=np.int32)
    lookahead = n if not lookahead_bars or lookahead_bars <= 0 else lookahead_bars
    expirations: list[list[int]] = [[] for _ in range(n)]
    expired = np.zeros(n, dtype=bool)
    expiry_for_anchor = np.full(n, n - 1, dtype=np.int32)

    for idx in range(n):
        high = highs[idx]
        low = lows[idx]

        while pending_up and pending_up[0][0] <= high:
            _, anchor = heapq.heappop(pending_up)
            if expired[anchor] or next_up[anchor] != -1:
                continue
            next_up[anchor] = idx

        neg_low = -low
        while pending_down and pending_down[0][0] <= neg_low:
            _, anchor = heapq.heappop(pending_down)
            if expired[anchor] or next_down[anchor] != -1:
                continue
            next_down[anchor] = idx

        anchor_price = closes[idx]
        heapq.heappush(pending_up, (anchor_price + up_delta, idx))
        heapq.heappush(pending_down, (-(anchor_price - down_delta), idx))

        expiry_idx = min(n - 1, idx + lookahead)
        expiry_for_anchor[idx] = expiry_idx
        expirations[expiry_idx].append(idx)
        for anchor in expirations[idx]:
            expired[anchor] = True

    for idx in range(n):
        up_hit = next_up[idx]
        down_hit = next_down[idx]

        if up_hit == -1 and down_hit == -1:
            exit_indices[idx] = int(expiry_for_anchor[idx])
            valid_mask[idx] = True
            continue

        if up_hit == -1:
            labels[idx] = 0  # down
            exit_indices[idx] = down_hit
        elif down_hit == -1:
            labels[idx] = 2  # up
            exit_indices[idx] = up_hit
        else:
            if up_hit == down_hit:
                labels[idx] = 1  # flat tie
                exit_indices[idx] = up_hit
            elif up_hit < down_hit:
                labels[idx] = 2
                exit_indices[idx] = up_hit
            else:
                labels[idx] = 0
                exit_indices[idx] = down_hit

        if exit_indices[idx] != -1:
            valid_mask[idx] = True

    return labels, valid_mask, exit_indices

def cache_streaming_files(
    stems: List[str],
    config: Dict[str, Any],
    resolution_plan: List[str],
    feature_cols_map: Dict[str, List[str]],
    lookahead_bars: int | None = None,
    store_cols_map: Dict[str, List[str]] | None = None,
) -> List[StreamingFileEntry]:
    
    feature_root = Path(config['paths']['feature_root'])
    cache_dir = Path(config['paths']['cache_dir']) / "stream"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    for r in resolution_plan:
        (cache_dir / r).mkdir(parents=True, exist_ok=True)

    if store_cols_map is None:
        store_cols_map = feature_cols_map
        
    entries = []
    primary_res = resolution_plan[0]
    tick_size = float(config['trade_simulation']['tick_size'])
    target_ticks = float(config['trade_simulation']['target_ticks'])
    stop_ticks = float(config['trade_simulation']['stop_ticks'])
    
    # Enable cross-resolution aggregates only for Phase 8 feature set.
    feature_set_name = str(config.get("data", {}).get("feature_set", "") or "")
    use_cross_resolution = feature_set_name == "phase8"

    # Progress bar added here!
    for stem in tqdm(stems, desc="Caching Feature Files", unit="file"):
        base_frame = load_streaming_frame(
            stem, feature_root, primary_res, feature_cols_map[primary_res]
        )
        
        if base_frame is None:
            continue
            
        align_df = base_frame[["start_timestamp_ns"]].copy()
        resolution_tables = {primary_res: base_frame[feature_cols_map[primary_res]].copy()}
        raw_slices: Dict[str, pd.DataFrame] = {}
        
        valid_mask = np.ones(len(base_frame), dtype=bool)
        
        # Load other resolutions
        failed = False
        for res in resolution_plan[1:]:
            slice_df = load_resolution_slice(stem, feature_root, res, feature_cols_map[res])
            if slice_df is None:
                failed = True
                break

            if use_cross_resolution:
                raw_slices[res] = slice_df[["start_timestamp_ns", *feature_cols_map[res]]].copy()
                
            merged = align_df.merge(slice_df, on="start_timestamp_ns", how="left")
            res_data = merged[feature_cols_map[res]].copy().ffill().bfill()
            
            valid = res_data.notna().all(axis=1).to_numpy()
            valid_mask &= valid
            resolution_tables[res] = res_data
            
        if failed or not valid_mask.any():
            continue
        
        valid_indices = np.where(valid_mask)[0]
        base_frame = base_frame.iloc[valid_indices].reset_index(drop=True)
        for res in list(resolution_tables.keys()):
            resolution_tables[res] = resolution_tables[res].iloc[valid_indices].reset_index(drop=True)
        # We filtered base_frame; adjust raw_slices to the same primary timeline if needed later.

        labels, valid_targets, exit_indices = derive_triclass_targets(
            base_frame,
            tick_size,
            target_ticks,
            stop_ticks,
            lookahead_bars=lookahead_bars,
        )

        if not valid_targets.any():
            logging.warning(f"File {stem}: no valid binary targets after walk-forward labeling")
            continue
        
        # Save to disk
        num_rows = len(base_frame)
        targets = np.zeros((num_rows, 1), dtype=np.int8)
        targets[:, 0] = np.where(valid_targets, labels, 1)

        mask_path = cache_dir / f"{stem}_target_mask.npy"
        np.save(mask_path, valid_targets.astype(np.bool_))

        exit_path = cache_dir / f"{stem}_exit_idx.npy"
        np.save(exit_path, exit_indices.astype(np.int32))

        timestamps = base_frame["start_timestamp_ns"].to_numpy(dtype=np.int64)
        timestamp_path = cache_dir / f"{stem}_timestamps.npy"
        np.save(timestamp_path, timestamps)

        price_matrix = base_frame[PRICE_COLUMNS].to_numpy(dtype=np.float32)
        price_path = cache_dir / f"{stem}_prices.npy"
        np.save(price_path, price_matrix)
            
        # Optionally compute cross-resolution aggregates for the primary resolution.
        if use_cross_resolution:
            mid_slice = raw_slices.get("mid")
            slow_slice = raw_slices.get("slow")
            if mid_slice is not None:
                # Restrict slices to the same time range as base_frame, if necessary.
                # We rely on timestamps for grouping; duplicate entries are harmless.
                cross_df = _compute_cross_resolution_features(
                    base_frame,
                    mid_slice,
                    slow_slice,
                    tick_size,
                )
                # Ensure all required columns exist, even if zero-filled.
                for col in CROSS_RES_COLUMNS:
                    if col not in cross_df.columns:
                        cross_df[col] = 0.0
                resolution_tables[primary_res] = pd.concat(
                    [resolution_tables[primary_res].reset_index(drop=True), cross_df.reset_index(drop=True)],
                    axis=1,
                )

        feature_paths = {}
        feature_sums = {}
        feature_sumsq = {}
        
        save_fail = False
        for res, df in resolution_tables.items():
            cols = store_cols_map.get(res, feature_cols_map[res])
            df_use = df[cols].copy()
            data = df_use.to_numpy(dtype=np.float32)
            if data.size == 0:
                save_fail = True
                break
                
            path = cache_dir / res / f"{stem}_{res}.npy"
            np.save(path, data)
            feature_paths[res] = path
            feature_sums[res] = data.sum(axis=0, dtype=np.float64)
            feature_sumsq[res] = np.square(data, dtype=np.float64).sum(axis=0)
            
        if save_fail:
            continue
            
        t_path = cache_dir / f"{stem}_targets.npy"
        np.save(t_path, targets)
        
        # Count classes
        counts = np.zeros((targets.shape[1], NUM_TARGET_CLASSES), dtype=np.int64)
        valid_labels = labels[valid_targets]
        if valid_labels.size:
            counts[0] = np.bincount(valid_labels, minlength=NUM_TARGET_CLASSES)

        entries.append(StreamingFileEntry(
            stem=stem,
            rows=num_rows,
            start_ts=int(base_frame["start_timestamp_ns"].iloc[0]),
            feature_paths=feature_paths,
            targets_path=t_path,
            target_mask_path=mask_path,
            exit_index_path=exit_path,
            timestamp_path=timestamp_path,
            price_path=price_path,
            feature_sums=feature_sums,
            feature_sumsq=feature_sumsq,
            target_counts=counts
        ))
        
        del base_frame, resolution_tables, align_df, targets
        gc.collect()
        
    return entries

def compute_stats(entries: List[StreamingFileEntry], resolution_plan: List[str], col_map: Dict):
    stats = {}
    for res in resolution_plan:
        cols = col_map[res]
        if not cols: continue
        
        sum_vec = np.zeros(len(cols), dtype=np.float64)
        sumsq_vec = np.zeros(len(cols), dtype=np.float64)
        total = 0
        
        for e in entries:
            if res in e.feature_sums:
                total += e.rows
                sum_vec += e.feature_sums[res]
                sumsq_vec += e.feature_sumsq[res]
                
        if total < 2:
            stats[res] = (np.zeros_like(sum_vec), np.ones_like(sum_vec))
        else:
            means = sum_vec / total
            var = np.maximum((sumsq_vec / total) - means**2, MIN_STD**2)
            stats[res] = (means.astype(np.float32), np.sqrt(var).astype(np.float32))
    return stats


def _compute_cross_resolution_features(
    base_frame: pd.DataFrame,
    mid_slice: Optional[pd.DataFrame],
    slow_slice: Optional[pd.DataFrame],
    tick_size: float,
) -> pd.DataFrame:
    """Compute cross-resolution aggregates on the primary (fast) grid.

    H1 (fast+mid): avg_fast_spread_abs, sum_fast_trade_volume, sum_fast_ofi_net
    H2 (mid+slow): avg_mid_spread_abs, sum_mid_trade_volume, sum_mid_ofi_net

    Features are defined per mid/slow bar but broadcast back to the fast grid so
    that each fast row carries the aggregates for its containing mid/slow bar.
    """
    n = len(base_frame)
    out = pd.DataFrame(index=base_frame.index)
    # Initialize with zeros; we fill where we have valid aggregates.
    out["avg_fast_spread_abs"] = 0.0
    out["sum_fast_trade_volume"] = 0.0
    out["sum_fast_ofi_net"] = 0.0
    out["avg_mid_spread_abs"] = 0.0
    out["sum_mid_trade_volume"] = 0.0
    out["sum_mid_ofi_net"] = 0.0

    if mid_slice is not None and not mid_slice.empty:
        # Map each fast row to a mid bar via start_timestamp_ns.
        fast_ts = base_frame["start_timestamp_ns"].to_numpy(dtype=np.int64)
        mid_ts = mid_slice["start_timestamp_ns"].to_numpy(dtype=np.int64)
        if mid_ts.size > 0:
            mid_idx_for_fast = np.searchsorted(mid_ts, fast_ts, side="right") - 1
            valid_mid = (mid_idx_for_fast >= 0) & (mid_idx_for_fast < mid_ts.size)

            spread_ticks_fast = base_frame["spread_ticks"].to_numpy(dtype=np.float64)
            spread_abs_fast = spread_ticks_fast * float(tick_size)
            vol_fast = base_frame["trade_volume_sum_rel"].to_numpy(dtype=np.float64)
            ofi_fast = base_frame["ofi_net_log"].to_numpy(dtype=np.float64)

            # Aggregate per mid bar.
            avg_fast_spread_abs_mid = np.zeros(mid_ts.size, dtype=np.float64)
            sum_fast_trade_mid = np.zeros(mid_ts.size, dtype=np.float64)
            sum_fast_ofi_mid = np.zeros(mid_ts.size, dtype=np.float64)

            for j in range(mid_ts.size):
                mask = mid_idx_for_fast == j
                if not np.any(mask):
                    continue
                avg_fast_spread_abs_mid[j] = spread_abs_fast[mask].mean()
                sum_fast_trade_mid[j] = vol_fast[mask].sum()
                sum_fast_ofi_mid[j] = ofi_fast[mask].sum()

            out.loc[valid_mid, "avg_fast_spread_abs"] = avg_fast_spread_abs_mid[
                mid_idx_for_fast[valid_mid]
            ]
            out.loc[valid_mid, "sum_fast_trade_volume"] = sum_fast_trade_mid[
                mid_idx_for_fast[valid_mid]
            ]
            out.loc[valid_mid, "sum_fast_ofi_net"] = sum_fast_ofi_mid[
                mid_idx_for_fast[valid_mid]
            ]

            # H2: aggregate mid into slow, then broadcast to fast via mid.
            if slow_slice is not None and not slow_slice.empty:
                slow_ts = slow_slice["start_timestamp_ns"].to_numpy(dtype=np.int64)
                if slow_ts.size > 0:
                    # Map each mid bar to a slow bar.
                    slow_idx_for_mid = (
                        np.searchsorted(slow_ts, mid_ts, side="right") - 1
                    )
                    valid_slow_mid = (slow_idx_for_mid >= 0) & (
                        slow_idx_for_mid < slow_ts.size
                    )

                    spread_abs_mid = (
                        mid_slice["spread_ticks"].to_numpy(dtype=np.float64)
                        * float(tick_size)
                    )
                    vol_mid = mid_slice["trade_volume_sum_rel"].to_numpy(
                        dtype=np.float64
                    )
                    ofi_mid = mid_slice["ofi_net_log"].to_numpy(dtype=np.float64)

                    avg_mid_spread_abs_slow = np.zeros(slow_ts.size, dtype=np.float64)
                    sum_mid_trade_slow = np.zeros(slow_ts.size, dtype=np.float64)
                    sum_mid_ofi_slow = np.zeros(slow_ts.size, dtype=np.float64)

                    for j in range(slow_ts.size):
                        mask = slow_idx_for_mid == j
                        if not np.any(mask):
                            continue
                        avg_mid_spread_abs_slow[j] = spread_abs_mid[mask].mean()
                        sum_mid_trade_slow[j] = vol_mid[mask].sum()
                        sum_mid_ofi_slow[j] = ofi_mid[mask].sum()

                    # Map each fast row -> mid -> slow.
                    slow_idx_for_fast = np.full(n, -1, dtype=np.int64)
                    slow_idx_for_fast[valid_mid] = slow_idx_for_mid[
                        mid_idx_for_fast[valid_mid]
                    ]
                    valid_slow = (slow_idx_for_fast >= 0) & (
                        slow_idx_for_fast < slow_ts.size
                    )

                    out.loc[valid_slow, "avg_mid_spread_abs"] = avg_mid_spread_abs_slow[
                        slow_idx_for_fast[valid_slow]
                    ]
                    out.loc[valid_slow, "sum_mid_trade_volume"] = sum_mid_trade_slow[
                        slow_idx_for_fast[valid_slow]
                    ]
                    out.loc[valid_slow, "sum_mid_ofi_net"] = sum_mid_ofi_slow[
                        slow_idx_for_fast[valid_slow]
                    ]

    return out

def standardize_entries(
    entries: List[StreamingFileEntry],
    stats: Dict,
    clip_value: float | None = None,
) -> None:
    """Standardize cached features in-place, with optional clipping for robustness.

    Clipping limits extreme z-scores that can otherwise propagate through the
    TCN and produce numerically unstable logits on a small subset of samples.
    """
    logging.info("Standardizing cache files in-place...")
    for entry in tqdm(entries, desc="Standardizing"):
        for res, (mean, std) in stats.items():
            if res in entry.feature_paths:
                path = entry.feature_paths[res]
                data = np.load(path, mmap_mode="r+")
                # z-score in-place
                data[:] = (data - mean) / std
                # Replace NaN/Inf that might originate from bad inputs/std
                np.nan_to_num(data, copy=False)
                # Optionally clamp extreme values to keep the model numerically stable
                if clip_value is not None and clip_value > 0:
                    np.clip(data, -clip_value, clip_value, out=data)
                data.flush()
                del data
