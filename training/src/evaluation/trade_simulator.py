from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.data.loader import StreamingFileEntry


@dataclass
class PredictionRecord:
    entry_idx: int
    target_idx: int
    label: int
    tcn_prob: float
    logistic_prob: Optional[float] = None


class TradeSimulator:
    def __init__(
        self,
        entries: Sequence[StreamingFileEntry],
        threshold: float,
        target_ticks: float,
        stop_ticks: float,
        tick_size: float,
    ) -> None:
        self.entries = list(entries)
        self.threshold = threshold
        self.target_ticks = target_ticks
        self.stop_ticks = stop_ticks
        self.tick_size = tick_size
        self._timestamp_handles: Dict[int, np.ndarray] = {}
        self._price_handles: Dict[int, np.ndarray] = {}
        self.entry_offsets: List[int] = []

        cumulative = 0
        for entry in self.entries:
            self.entry_offsets.append(cumulative)
            cumulative += entry.rows

    def simulate(
        self,
        records: Sequence[PredictionRecord],
        prob_field: str,
        threshold: float | None = None,
    ) -> Dict[str, float]:
        cutoff = self.threshold if threshold is None else threshold
        events = []
        for rec in records:
            prob = getattr(rec, prob_field, None)
            if prob is None:
                continue
            timestamps = self._open_timestamps(rec.entry_idx)
            if rec.target_idx >= len(timestamps):
                continue
            ts = int(timestamps[rec.target_idx])
            abs_idx = self._absolute_index(rec.entry_idx, rec.target_idx)
            events.append((ts, abs_idx, rec, float(prob)))

        events.sort(key=lambda item: (item[0], item[1]))

        open_until = -1
        wins = losses = 0
        blocked = skipped = 0
        mismatches = 0
        trade_durations: List[int] = []
        for _, abs_idx, rec, prob in events:
            if prob <= cutoff:
                continue
            if abs_idx <= open_until:
                blocked += 1
                continue

            exit_idx, direction = self._resolve_trade(rec.entry_idx, rec.target_idx)
            if exit_idx < 0:
                skipped += 1
                continue

            exit_abs_idx = self._absolute_index(rec.entry_idx, exit_idx)
            open_until = exit_abs_idx
            trade_durations.append(max(exit_abs_idx - abs_idx, 1))

            if direction == 1:
                wins += 1
            else:
                losses += 1

            if direction != rec.label:
                mismatches += 1

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

    def _resolve_trade(self, entry_idx: int, start_idx: int) -> tuple[int, int]:
        prices = self._open_prices(entry_idx)
        if start_idx >= len(prices) - 1:
            return -1, 0

        anchor = prices[start_idx, 0]
        up_level = anchor + self.tick_size * self.target_ticks
        down_level = anchor - self.tick_size * self.stop_ticks

        for idx in range(start_idx + 1, len(prices)):
            high = prices[idx, 1]
            low = prices[idx, 2]
            hit_up = high >= up_level
            hit_down = low <= down_level
            if hit_up and hit_down:
                return -1, 0
            if hit_up:
                return idx, 1
            if hit_down:
                return idx, 0

        return -1, 0
