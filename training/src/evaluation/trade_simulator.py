from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data.loader import StreamingFileEntry
from src.definitions import DOWN_CLASS_INDEX, FLAT_CLASS_INDEX, UP_CLASS_INDEX


@dataclass
class PredictionRecord:
    entry_idx: int
    target_idx: int
    label: int
    move_prob: float
    up_prob: float
    down_prob: float
    logistic_move_prob: Optional[float] = None


class TradeSimulator:
    def __init__(
        self,
        entries: Sequence[StreamingFileEntry],
        threshold: float,
        target_ticks: float,
        stop_ticks: float,
        tick_size: float,
        lookahead_bars: int | None = None,
    ) -> None:
        self.entries = list(entries)
        self.threshold = threshold
        self.target_ticks = target_ticks
        self.stop_ticks = stop_ticks
        self.tick_size = tick_size
        self.lookahead_bars = lookahead_bars if lookahead_bars and lookahead_bars > 0 else None
        self._timestamp_handles: Dict[int, np.ndarray] = {}
        self._price_handles: Dict[int, np.ndarray] = {}
        self._exit_idx_handles: Dict[int, np.ndarray] = {}
        self._target_handles: Dict[int, np.ndarray] = {}
        self.entry_offsets: List[int] = []

        cumulative = 0
        for entry in self.entries:
            self.entry_offsets.append(cumulative)
            cumulative += entry.rows

    def simulate(
        self,
        records: Sequence[PredictionRecord],
        use_logistic: bool = False,
        threshold: float | None = None,
    ) -> Dict[str, float]:
        cutoff = self.threshold if threshold is None else threshold
        events = []
        for rec in records:
            # For TCN-based simulation, gate on the *directional* probability,
            # mirroring realtime behaviour (max(up_prob, down_prob)).
            # For logistic baselines, use the provided move probability.
            if use_logistic:
                prob = rec.logistic_move_prob
            else:
                prob = max(rec.up_prob, rec.down_prob)
            if prob is None:
                continue
            timestamps = self._open_timestamps(rec.entry_idx)
            if rec.target_idx >= len(timestamps):
                continue
            ts = int(timestamps[rec.target_idx])
            abs_idx = self._absolute_index(rec.entry_idx, rec.target_idx)
            events.append((ts, abs_idx, rec, float(prob)))

        # Sort by (timestamp, absolute index) to mirror realtime/offline replay ordering.
        events.sort(key=lambda item: (item[0], item[1]))

        # Simulated trade state, aligned with deployment/offline-eval:
        # - at most one open trade at a time
        # - 2 minute cooldown between *entry timestamps* (in nanoseconds)
        # We approximate "open trade" using cached exit indices and their
        # timestamps, so that the lifetime matches label generation.
        current_trade_exit_ts_ns: Optional[int] = None
        last_entry_timestamp_ns: Optional[int] = None
        cooldown_ns = int(120 * 60 * 1e9)

        wins = losses = 0
        blocked = skipped = 0
        mismatches = 0
        mismatch_examples = 0
        trade_durations: List[int] = []
        for ts_ns, abs_idx, rec, prob in events:
            if prob <= cutoff:
                continue

            # Enforce one-at-a-time trades and 2 minute cooldown, using
            # timestamps to mirror deployment/offline-eval semantics.
            can_open = True
            # If there is an existing trade whose labeled exit timestamp is
            # in the future relative to this candidate, we are still "open".
            if current_trade_exit_ts_ns is not None and ts_ns < current_trade_exit_ts_ns:
                can_open = False
            # Additionally, require at least 2 minutes since the last entry.
            if can_open and ts_ns > 0 and last_entry_timestamp_ns is not None:
                if ts_ns - last_entry_timestamp_ns < cooldown_ns:
                    can_open = False

            if not can_open:
                blocked += 1
                continue

            exit_idx, realized_label = self._resolve_trade(rec.entry_idx, rec.target_idx)
            if exit_idx < 0:
                skipped += 1
                continue

            predicted_dir = (
                UP_CLASS_INDEX if rec.up_prob >= rec.down_prob else DOWN_CLASS_INDEX
            )
            if realized_label == FLAT_CLASS_INDEX:
                direction_win = False
            else:
                direction_win = realized_label == predicted_dir

            exit_abs_idx = self._absolute_index(rec.entry_idx, exit_idx)
            trade_durations.append(max(exit_abs_idx - abs_idx, 1))

            # Use labeled exit timestamp as the "close" time, and the
            # current event timestamp as the entry time for cooldown.
            exit_ts_arr = self._open_timestamps(rec.entry_idx)
            if 0 <= exit_idx < len(exit_ts_arr):
                current_trade_exit_ts_ns = int(exit_ts_arr[exit_idx])
            else:
                current_trade_exit_ts_ns = ts_ns
            last_entry_timestamp_ns = ts_ns

            if direction_win:
                wins += 1
            else:
                losses += 1

            if realized_label != FLAT_CLASS_INDEX and realized_label != predicted_dir:
                mismatches += 1
                if mismatch_examples < 5:
                    mismatch_examples += 1
                    # Minimal inline debug; caller can correlate entry_idx/target_idx in cache if needed.
                    print(
                        f"[TradeSim] mismatch entry={rec.entry_idx} target={rec.target_idx} realized={realized_label} predicted={predicted_dir}"
                    )

        total_trades = wins + losses
        if total_trades:
            expectancy = (wins * self.target_ticks - losses * self.stop_ticks) / total_trades
            avg_duration = float(np.mean(trade_durations)) if trade_durations else 0.0
        else:
            expectancy = 0.0
            avg_duration = 0.0

        entry_rate = (total_trades / max(len(events), 1)) * 100.0
        net_ticks = (wins * self.target_ticks) - (losses * self.stop_ticks)

        return {
            "entries": float(total_trades),
            "wins": float(wins),
            "losses": float(losses),
            "blocked": float(blocked),
            "skipped": float(skipped),
            "price_mismatches": float(mismatches),
            "entry_rate": entry_rate,
            "net_ticks": net_ticks,
            "expectancy": expectancy,
            "avg_duration_bars": avg_duration,
            "population": float(len(events)),
            "mismatch_rate": (mismatches / max(total_trades, 1)) * 100.0 if total_trades else 0.0,
        }

    def _open_timestamps(self, entry_idx: int) -> np.ndarray:
        if entry_idx not in self._timestamp_handles:
            path = self.entries[entry_idx].timestamp_path
            self._timestamp_handles[entry_idx] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._timestamp_handles[entry_idx]

    def _absolute_index(self, entry_idx: int, local_idx: int) -> int:
        return self.entry_offsets[entry_idx] + local_idx

    def _open_prices(self, entry_idx: int) -> np.ndarray:
        if entry_idx not in self._price_handles:
            path = self.entries[entry_idx].price_path
            self._price_handles[entry_idx] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._price_handles[entry_idx]

    def _open_exit_indices(self, entry_idx: int) -> np.ndarray | None:
        if entry_idx not in self._exit_idx_handles:
            path = self.entries[entry_idx].exit_index_path
            try:
                self._exit_idx_handles[entry_idx] = np.load(path, mmap_mode="r", allow_pickle=False)
            except OSError:
                self._exit_idx_handles[entry_idx] = None
        return self._exit_idx_handles[entry_idx]

    def _open_targets(self, entry_idx: int) -> np.ndarray | None:
        if entry_idx not in self._target_handles:
            path = self.entries[entry_idx].targets_path
            try:
                self._target_handles[entry_idx] = np.load(path, mmap_mode="r", allow_pickle=False)
            except OSError:
                self._target_handles[entry_idx] = None
        return self._target_handles[entry_idx]

    def _resolve_trade(self, entry_idx: int, start_idx: int) -> tuple[int, int]:
        # Logic MUST be identical to label generation. We now exclusively use cached data.
        exit_idx_arr = self._open_exit_indices(entry_idx)
        targets_arr = self._open_targets(entry_idx)

        # If cached data is missing or index is invalid, we cannot resolve the trade.
        # Fallback to re-derivation is risky and has been proven to be inconsistent.
        if (exit_idx_arr is None 
            or targets_arr is None
            or start_idx >= len(exit_idx_arr) 
            or start_idx >= len(targets_arr)):
            return -1, FLAT_CLASS_INDEX

        exit_idx = int(exit_idx_arr[start_idx])
        # A negative exit_idx means no valid exit was found during labeling.
        if exit_idx < 0:
            return -1, FLAT_CLASS_INDEX

        direction = int(targets_arr[start_idx, 0])
        return exit_idx, direction
