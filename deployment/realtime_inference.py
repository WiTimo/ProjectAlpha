"""Realtime inference service for Project Alpha."""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.model.network import DilatedTCN

NUM_TARGET_CLASSES = 3


def load_model(bundle_path: Path) -> Dict[str, Any]:
    import joblib

    # Joblib-based logistic regression bundle (legacy/simple baseline)
    if str(bundle_path).endswith(".pkl"):
        bundle = joblib.load(bundle_path)
        model = bundle["model"]
        feature_columns = bundle.get("feature_columns", [])
        threshold = bundle.get("threshold", 0.5)
        return {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "type": "logistic_regression",
        }

    # TCN bundle exported by training/tcn.py
    bundle = torch.load(bundle_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise KeyError(
            "Model file does not contain a metadata bundle. "
            "Please re-run training with the updated exporter."
        )

    model_state = bundle.get("model_state_dict") or bundle.get("state_dict")
    if model_state is None:
        raise KeyError(
            "Model bundle missing 'model_state_dict'/'state_dict'. Re-export training bundle."
        )

    feature_columns = bundle.get("feature_columns", [])
    if not feature_columns:
        raise KeyError(
            "Model bundle missing feature column metadata; re-export the training bundle."
        )

    scaler = bundle.get("scaler", {})
    target_columns = bundle.get("target_columns") or (
        [bundle.get("target_column")] if bundle.get("target_column") else []
    )
    num_targets = max(1, len(target_columns) or 1)

    training_cfg = bundle.get("training", {})
    model_meta = dict(training_cfg.get("model", {}))
    hidden_dim = int(model_meta.get("hidden_dim", 32))
    layers = int(model_meta.get("layers", 2))
    kernel_size = int(model_meta.get("kernel_size", 5))
    dropout = float(model_meta.get("dropout", 0.0))
    dilation_base = int(model_meta.get("dilation_base", 2))
    num_channels = [hidden_dim] * layers
    sequence_len = max(1, int(training_cfg.get("sequence_len") or 1))

    model = DilatedTCN(
        num_inputs=len(feature_columns),
        num_classes=NUM_TARGET_CLASSES,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        dilation_base=dilation_base,
    )
    model.load_state_dict(model_state)
    model.eval()

    means = np.array(
        [scaler.get("means", {}).get(col, 0.0) for col in feature_columns], dtype=np.float32
    )
    stds = np.array(
        [scaler.get("stds", {}).get(col, 1.0) for col in feature_columns], dtype=np.float32
    )
    stds = np.where(stds == 0, 1.0, stds)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "target_columns": target_columns or ["t40"],
        "means": means,
        "stds": stds,
        "sequence_len": sequence_len,
        "num_targets": num_targets,
        "type": "tcn",
    }


def standardize(features: Dict[str, float], feature_columns: Iterable[str], means, stds) -> np.ndarray:
    """Standardize a single feature row using training-style z-score + clipping.

    Mirrors the training pipeline:
      - (x - mean) / std
      - replace NaN/Inf with finite values
      - clamp extreme z-scores to a fixed range for numerical stability
    """
    vector = np.array([features.get(col, 0.0) for col in feature_columns], dtype=np.float32)
    vector = (vector - means) / stds
    # Guard against NaN/Inf originating from bad inputs or zero std
    vector = np.nan_to_num(vector, copy=False)
    # Clip using the same default as training (`standardize_clip`); override via env if needed
    clip_raw = os.getenv("ALPHA_STANDARDIZE_CLIP", "10.0")
    try:
        clip_value = float(clip_raw)
    except ValueError:
        clip_value = 10.0
    if clip_value > 0:
        np.clip(vector, -clip_value, clip_value, out=vector)
    return vector


class HotkeyEmitter:
    """Minimal Windows hotkey sender for trade triggers."""

    _BASE_KEYCODES = {
        "CTRL": 0x11,
        "CONTROL": 0x11,
        "SHIFT": 0x10,
        "ALT": 0x12,
        "WIN": 0x5B,
        "ENTER": 0x0D,
        "RETURN": 0x0D,
        "TAB": 0x09,
        "SPACE": 0x20,
    }

    def __init__(self) -> None:
        self.available = platform.system() == "Windows" and hasattr(ctypes, "windll")
        self._user32 = ctypes.windll.user32 if self.available else None

    def press(self, combo: str) -> bool:
        if not self.available or not self._user32:
            logging.warning("Hotkey emission unavailable on this platform; requested %s", combo)
            return False

        tokens = [token.strip().upper() for token in combo.split("+") if token.strip()]
        if not tokens:
            logging.warning("Ignoring empty hotkey combo")
            return False

        codes: List[int] = []
        for token in tokens:
            code = self._vk_code(token)
            if code is None:
                logging.warning("Unsupported hotkey token '%s'", token)
                return False
            codes.append(code)

        for code in codes:
            self._user32.keybd_event(code, 0, 0, 0)
        for code in reversed(codes):
            self._user32.keybd_event(code, 0, 0x0002, 0)
        return True

    def _vk_code(self, token: str) -> Optional[int]:
        if token in self._BASE_KEYCODES:
            return self._BASE_KEYCODES[token]
        if token.startswith("F") and token[1:].isdigit():
            fn_idx = int(token[1:])
            if 1 <= fn_idx <= 24:
                return 0x6F + fn_idx  # F1 starts at 0x70
        if len(token) == 1 and token.isalnum():
            return ord(token)
        return None


class FeatureTail:
    def __init__(self, path: Path, start_at_end: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._file = self.path.open("r", encoding="utf-8", errors="ignore")
        if start_at_end:
            self._file.seek(0, os.SEEK_END)
        self._position = self._file.tell()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def _reopen(self) -> None:
        self.close()
        self._file = self.path.open("r", encoding="utf-8", errors="ignore")
        self._position = 0

    def read_new_lines(self) -> List[str]:
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            time.sleep(0.25)
            return []
        if current_size < self._position:
            self._reopen()
        self._file.seek(self._position)
        lines = self._file.readlines()
        self._position = self._file.tell()
        return [line.strip() for line in lines if line.strip()]


def parse_feature_line(
    line: str, expected_resolution: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        logging.debug("Skipping non-JSON line: %s", line)
        return None, "non-json"

    resolution = str(payload.get("resolution", "")).lower()
    # Require all three resolutions to be present in the stream
    if resolution not in ["fast", "mid", "slow"]:
        logging.debug(
            "Dropping row due to invalid resolution (got=%s)",
            resolution,
        )
        return None, "resolution-mismatch"

    features = payload.get("features")
    if not isinstance(features, dict):
        logging.debug("JSON line missing 'features' dictionary: %s", line)
        return None, "missing-features"

    return payload, None


def is_in_pause_window(timestamp_ns: int, start_min: Optional[int], end_min: Optional[int]) -> bool:
    """Return True if the given data timestamp falls inside the configured pause window.

    The window is defined in minutes-from-midnight (0-1440), based on the *data* time
    carried by the feature stream (UTC). A wrapped window (e.g. 1320-60 for 22:00-01:00)
    is supported.
    """
    if timestamp_ns <= 0 or start_min is None or end_min is None:
        return False
    try:
        dt = datetime.utcfromtimestamp(timestamp_ns / 1e9)
    except (OverflowError, OSError, ValueError):
        return False
    minute_of_day = dt.hour * 60 + dt.minute

    if start_min == end_min:
        # Empty window when equal; treat as no pause to avoid surprising behaviour.
        return False

    if 0 <= start_min < 1440 and 0 <= end_min < 1440:
        if start_min < end_min:
            return start_min <= minute_of_day < end_min
        # Wrapped window over midnight.
        return minute_of_day >= start_min or minute_of_day < end_min

    return False


def aggregate_trades_by_time(
    trades: List[Dict[str, Any]], window_seconds: float = 120.0
) -> List[Dict[str, Any]]:
    """Aggregate sequential completed trades into time-based clusters.

    Trades whose *entry* timestamps are within `window_seconds` of the previous
    trade's entry are merged into a single logical trade. PnL is summed and the
    result label is recomputed from the aggregated pips.
    """
    if not trades:
        return []

    sorted_trades = sorted(
        trades, key=lambda t: int(t.get("entry_timestamp_ns") or 0)
    )
    window_ns = int(window_seconds * 1e9)

    aggregated: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for trade in sorted_trades:
        entry_ns = int(trade.get("entry_timestamp_ns") or 0)
        if current is None:
            current = dict(trade)
            continue

        prev_entry_ns = int(current.get("entry_timestamp_ns") or 0)
        if entry_ns > 0 and prev_entry_ns > 0 and entry_ns - prev_entry_ns <= window_ns:
            # Merge into current cluster: extend exit and accumulate PnL/duration.
            current["exit_timestamp_ns"] = trade.get("exit_timestamp_ns")
            current["exit_price"] = trade.get("exit_price")
            current["pips_move"] = float(current.get("pips_move", 0.0)) + float(
                trade.get("pips_move", 0.0)
            )
            current["duration_rows"] = int(current.get("duration_rows", 0)) + int(
                trade.get("duration_rows", 0)
            )
        else:
            aggregated.append(current)
            current = dict(trade)

    if current is not None:
        aggregated.append(current)

    # Recompute win/loss label from aggregated pips.
    for t in aggregated:
        pips = float(t.get("pips_move", 0.0))
        if pips > 0:
            t["result"] = "win"
        elif pips < 0:
            t["result"] = "loss"
        else:
            t["result"] = "flat"

    return aggregated


def inference_loop(args: argparse.Namespace) -> None:
    bundle = load_model(Path(args.model_bundle))
    model_type = bundle.get("type", "tcn")
    features_path = Path(args.features)
    tail = FeatureTail(features_path, start_at_end=not args.replay_existing)
    logging.info("Watching %s for feature rows", features_path)
    last_idle_log = time.time()

    total_rows: Optional[int] = None
    progress_bar = None

    eval_log_file = None
    trigger_log_file = None
    eval_summary_path: Optional[Path] = getattr(args, "eval_summary_json", None)
    threshold_sweep: List[float] = list(getattr(args, "eval_threshold_sweep", []) or [])
    if args.trade_threshold is not None and args.trade_threshold not in threshold_sweep:
        threshold_sweep.append(args.trade_threshold)

    eval_stats: Dict[str, Any] = {
        "rows_total": 0,
        "rows_parsed": 0,
        "rows_per_resolution": Counter(),
        "threshold_sweep": {
            float(t): {
                "up_triggers": 0,
                "down_triggers": 0,
                "either_triggers": 0,
            }
            for t in threshold_sweep
        },
        "hotkey_trades": {"up": 0, "down": 0},
        "first_start_timestamp_ns": None,
        "last_end_timestamp_ns": None,
        "config": {
            "trade_threshold": args.trade_threshold,
            "trigger_target": args.trigger_target,
            "up_hotkey": args.up_hotkey,
            "down_hotkey": args.down_hotkey,
            "entry_min_price_move": args.entry_min_price_move,
            "entry_price_feature": args.entry_price_feature,
            "idle_log_seconds": args.idle_log_seconds,
            "poll_interval": args.poll_interval,
            "pause_min_start": getattr(args, "pause_min_start", None),
            "pause_min_end": getattr(args, "pause_min_end", None),
        },
    }

    # Optional per-trigger log file (used mainly for live deployment).
    trigger_log_path: Optional[Path] = getattr(args, "trigger_log_file", None)
    if trigger_log_path is not None:
        try:
            trigger_log_path.parent.mkdir(parents=True, exist_ok=True)
            trigger_log_file = trigger_log_path.open("a", encoding="utf-8")
            logging.info("Trigger log will be written to %s", trigger_log_path)
        except Exception as exc:  # pragma: no cover - log-only
            logging.warning("Failed to open trigger log file %s: %s", trigger_log_path, exc)
            trigger_log_file = None

    if args.replay_existing:
        try:
            with features_path.open("r", encoding="utf-8", errors="ignore") as f:
                total_rows = sum(1 for _ in f)
        except FileNotFoundError:
            total_rows = None

        if total_rows is not None and total_rows > 0:
            logging.info(
                "Offline replay mode: %d feature rows found in %s",
                total_rows,
                features_path,
            )
            if getattr(args, "eval_log_jsonl", None) is not None:
                try:
                    eval_log_path = Path(args.eval_log_jsonl)
                    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
                    eval_log_file = eval_log_path.open("w", encoding="utf-8")
                    logging.info("Offline eval log will be written to %s", eval_log_path)
                except Exception as exc:  # pragma: no cover - log-only
                    logging.warning("Failed to open eval log file %s: %s", args.eval_log_jsonl, exc)
                    eval_log_file = None
        else:
            logging.info(
                "Offline replay mode: no feature rows found in %s (nothing to process yet)",
                features_path,
            )

    if model_type == "logistic_regression":
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        threshold = bundle["threshold"]
        try:
            while True:
                lines = tail.read_new_lines()
                if not lines:
                    now = time.time()
                    if now - last_idle_log >= args.idle_log_seconds:
                        logging.debug("Idle %.1fs", now - last_idle_log)
                        last_idle_log = now
                    time.sleep(args.poll_interval)
                    continue
                for line in lines:
                    parsed, drop_reason = parse_feature_line(line, args.resolution)
                    if not parsed:
                        continue
                    payload = parsed
                    vector = np.array([payload["features"].get(col, 0.0) for col in feature_columns], dtype=np.float32)
                    prob = model.predict_proba(vector.reshape(1, -1))[0, 1]
                    logging.info(f"LogisticRegression: up={prob:.3f} (threshold={threshold:.3f})")
                    # Add hotkey logic if needed
        except KeyboardInterrupt:
            logging.info("Stopping realtime inference")
        finally:
            tail.close()
        return

    # TCN logic (unchanged)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    means = bundle["means"]
    stds = bundle["stds"]
    sequence_len = bundle["sequence_len"]
    num_targets = bundle["num_targets"]
    target_columns = bundle["target_columns"] or [f"target_{idx}" for idx in range(num_targets)]
    window: deque[np.ndarray] = deque(maxlen=sequence_len)
    warmed_up = sequence_len == 1
    rows_seen = 0
    rows_read = 0
    drop_counts: Counter[str] = Counter()

    # Offline trade simulation state (replay_existing mode)
    open_trades: List[Dict[str, Any]] = []
    completed_trades: List[Dict[str, Any]] = []
    next_trade_id = 1

    trigger_idx: Optional[int] = None
    trigger_label: Optional[str] = None
    simulate_trades = args.trade_threshold is not None and num_targets > 0
    enable_triggers = simulate_trades and not args.replay_existing
    if simulate_trades:
        trigger_label = args.trigger_target or target_columns[0]
        if trigger_label is None:
            simulate_trades = False
        elif trigger_label not in target_columns:
            logging.warning("Trigger target '%s' not found; disabling hotkeys", trigger_label)
            simulate_trades = False
        else:
            trigger_idx = target_columns.index(trigger_label)
    hotkey_emitter = HotkeyEmitter() if enable_triggers else None
    if enable_triggers and hotkey_emitter and not hotkey_emitter.available:
        logging.warning("Hotkey emission unavailable on this platform; disabling triggers")
        enable_triggers = False
        hotkey_emitter = None

    # Wall-clock trigger timestamps for live mode (seconds)
    last_trigger_at_wall = {"up": 0.0, "down": 0.0}
    # Data-time trigger timestamps for offline mode (nanoseconds)
    last_trigger_at_data: Dict[str, Optional[int]] = {"up": None, "down": None}

    last_entry_price: Dict[str, Optional[float]] = {"up": None, "down": None}
    last_entry_timestamp_ns: Optional[int] = None

    logging.info(
        "Model loaded (features=%d, sequence_len=%d, targets=%s)",
        len(feature_columns),
        sequence_len,
        ", ".join(target_columns) or "n/a",
    )

    if args.replay_existing and total_rows and total_rows > 0:
        try:
            from tqdm import tqdm  # type: ignore

            progress_bar = tqdm(
                total=total_rows,
                desc="Offline eval",
                unit="row",
                ncols=100,
                leave=False,
            )
        except Exception:
            progress_bar = None

    try:
        while True:
            lines = tail.read_new_lines()
            if not lines:
                if args.replay_existing and total_rows and total_rows > 0 and rows_read >= total_rows:
                    break
                now = time.time()
                if now - last_idle_log >= args.idle_log_seconds:
                    drop_summary = ", ".join(
                        f"{reason}={count}" for reason, count in drop_counts.items()
                    ) or "none"
                    logging.debug(
                        "Idle %.1fs (buffer=%d/%d, rows=%d, drops=%s)",
                        now - last_idle_log,
                        len(window),
                        sequence_len,
                        rows_seen,
                        drop_summary,
                    )
                    last_idle_log = now
                time.sleep(args.poll_interval)
                continue

            for line in lines:
                rows_read += 1
                eval_stats["rows_total"] += 1

                parsed, drop_reason = parse_feature_line(line, args.resolution)
                if drop_reason:
                    drop_counts[drop_reason] += 1
                if not parsed:
                    if args.replay_existing and total_rows and total_rows > 0 and progress_bar is not None:
                        progress_bar.update(1)
                    continue

                payload = parsed
                eval_stats["rows_parsed"] += 1

                resolution = str(payload.get("resolution", ""))
                eval_stats["rows_per_resolution"][resolution] += 1

                start_ns = int(payload.get("start_timestamp_ns", 0))
                end_ns = int(payload.get("end_timestamp_ns", 0))
                if eval_stats["first_start_timestamp_ns"] is None and start_ns:
                    eval_stats["first_start_timestamp_ns"] = start_ns
                if end_ns:
                    eval_stats["last_end_timestamp_ns"] = end_ns

                rows_seen += 1
                vector = standardize(payload["features"], feature_columns, means, stds)
                window.append(vector)

                logging.debug(
                    "Buffered rows=%d buffer=%d/%d",
                    rows_seen,
                    len(window),
                    sequence_len,
                )

                if len(window) < sequence_len:
                    continue

                if not warmed_up:
                    logging.debug("Sequence buffer primed (%d steps)", sequence_len)
                    warmed_up = True

                stacked = np.stack(list(window), axis=0)
                channels_first = np.ascontiguousarray(stacked.T)
                tensor = torch.from_numpy(channels_first).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                    # Support both single-target (N, C) and flattened multi-target (N, T*C)
                    if logits.dim() == 2 and logits.shape[1] == NUM_TARGET_CLASSES:
                        logits = logits.view(1, 1, NUM_TARGET_CLASSES)
                        effective_targets = 1
                    elif logits.dim() == 2 and logits.shape[1] == num_targets * NUM_TARGET_CLASSES:
                        logits = logits.view(1, num_targets, NUM_TARGET_CLASSES)
                        effective_targets = num_targets
                    else:
                        raise RuntimeError(
                            f"Unexpected logits shape {tuple(logits.shape)} for "
                            f"num_targets={num_targets} and NUM_TARGET_CLASSES={NUM_TARGET_CLASSES}"
                        )
                    probs = torch.softmax(logits, dim=-1)

                summary_parts: List[str] = []
                per_target: Dict[str, Dict[str, float]] = {}
                for idx, name in enumerate(target_columns[:effective_targets]):
                    target_prob = probs[0, idx]
                    down_prob = float(target_prob[0].item())
                    up_prob = float(target_prob[2].item())
                    summary_parts.append(f"{name}:down={down_prob:.3f},up={up_prob:.3f}")
                    per_target[name] = {
                        "down": down_prob,
                        "up": up_prob,
                        "flat": float(target_prob[1].item()),
                    }

                # Threshold sweep stats (offline eval only; no labels)
                if threshold_sweep:
                    max_prob = 0.0
                    max_dir: Optional[str] = None
                    for name, probs_dict in per_target.items():
                        if probs_dict["up"] >= max_prob:
                            max_prob = probs_dict["up"]
                            max_dir = "up"
                        if probs_dict["down"] >= max_prob:
                            max_prob = probs_dict["down"]
                            max_dir = "down"
                    if max_dir is not None:
                        for thr in threshold_sweep:
                            if max_prob >= thr:
                                sweep_entry = eval_stats["threshold_sweep"][float(thr)]
                                sweep_entry["either_triggers"] += 1
                                if max_dir == "up":
                                    sweep_entry["up_triggers"] += 1
                                else:
                                    sweep_entry["down_triggers"] += 1

                # Actual trade triggering (threshold-based) and optional hotkey emission
                price_val = float(payload["features"].get(args.entry_price_feature, float("nan")))
                if simulate_trades and trigger_idx is not None:
                    probs_for_trigger = per_target.get(trigger_label or target_columns[0])

                    def price_ok(direction: str) -> bool:
                        if not np.isfinite(price_val) or args.entry_min_price_move <= 0.0:
                            return True
                        last_price = last_entry_price.get(direction)
                        if last_price is None:
                            return True
                        return abs(price_val - last_price) >= args.entry_min_price_move

                    if probs_for_trigger is not None and np.isfinite(price_val):
                        up_prob_trigger = probs_for_trigger["up"]
                        down_prob_trigger = probs_for_trigger["down"]

                        # Compute data-time timestamp (ns) for pause-window checks and offline stats.
                        current_ts_ns = end_ns or start_ns or 0
                        paused = is_in_pause_window(
                            current_ts_ns,
                            getattr(args, "pause_min_start", None),
                            getattr(args, "pause_min_end", None),
                        )

                        def log_trigger(direction: str, prob: float) -> None:
                            if trigger_log_file is None:
                                return
                            record: Dict[str, Any] = {
                                "event": "trade_trigger",
                                "direction": direction,
                                "local_time": datetime.now().isoformat(),
                                "trade_timestamp_ns": int(current_ts_ns) if current_ts_ns else None,
                                "trade_time_utc": (
                                    datetime.utcfromtimestamp(current_ts_ns / 1e9).isoformat()
                                    if current_ts_ns
                                    else None
                                ),
                                "probability": float(prob),
                                "threshold": float(args.trade_threshold)
                                if args.trade_threshold is not None
                                else None,
                                "resolution": resolution,
                                "entry_price_feature": args.entry_price_feature,
                                "entry_price": price_val if np.isfinite(price_val) else None,
                                "row_index": rows_seen,
                                "paused": paused,
                            }
                            try:
                                trigger_log_file.write(json.dumps(record) + "\n")
                                trigger_log_file.flush()
                            except Exception as exc:  # pragma: no cover - log-only
                                logging.debug("Failed to write trigger log record: %s", exc)

                        # Up trade trigger
                        if up_prob_trigger >= args.trade_threshold and price_ok("up"):
                            log_trigger("up", up_prob_trigger)
                            if not paused and enable_triggers and hotkey_emitter:
                                if hotkey_emitter.press(args.up_hotkey):
                                    logging.info(
                                        "Hotkey %s emitted for %s up=%.3f (>= %.3f) at price=%.2f",
                                        args.up_hotkey,
                                        trigger_label or target_columns[0],
                                        up_prob_trigger,
                                        args.trade_threshold,
                                        price_val,
                                    )
                            if args.replay_existing and current_ts_ns > 0:
                                last_trigger_at_data["up"] = current_ts_ns
                            else:
                                last_trigger_at_wall["up"] = time.time()
                            last_entry_price["up"] = (
                                price_val if np.isfinite(price_val) else last_entry_price["up"]
                            )
                            eval_stats["hotkey_trades"]["up"] += 1

                            # Simulated trade state: only allow opening a new trade if
                            # there is no open trade and at least 2 minutes
                            # have passed since the previous entry.
                            if np.isfinite(price_val):
                                can_open = not open_trades
                                if can_open and current_ts_ns > 0 and last_entry_timestamp_ns is not None:
                                    if current_ts_ns - last_entry_timestamp_ns < int(120 * 60 * 1e9):
                                        can_open = False
                                if can_open:
                                    trade = {
                                        "id": next_trade_id,
                                        "direction": "up",
                                        "entry_price": price_val,
                                        "entry_row_index": rows_seen,
                                        "entry_timestamp_ns": current_ts_ns,
                                        "threshold": float(args.trade_threshold),
                                        "resolution": resolution,
                                        "bar_index": int(payload.get("bar_index", -1)),
                                    }
                                    next_trade_id += 1
                                    open_trades.append(trade)
                                    last_entry_timestamp_ns = current_ts_ns
                                    if eval_log_file is not None:
                                        trade_evt = {
                                            "event": "trade_open",
                                            "trade_id": trade["id"],
                                            "direction": trade["direction"],
                                            "entry_price": trade["entry_price"],
                                            "entry_row_index": trade["entry_row_index"],
                                            "entry_timestamp_ns": trade["entry_timestamp_ns"],
                                        }
                                        try:
                                            eval_log_file.write(json.dumps(trade_evt) + "\n")
                                        except Exception as exc:  # pragma: no cover - log-only
                                            logging.debug("Failed to write trade_open event: %s", exc)

                        # Down trade trigger
                        if down_prob_trigger >= args.trade_threshold and price_ok("down"):
                            log_trigger("down", down_prob_trigger)
                            if not paused and enable_triggers and hotkey_emitter:
                                if hotkey_emitter.press(args.down_hotkey):
                                    logging.info(
                                        "Hotkey %s emitted for %s down=%.3f (>= %.3f) at price=%.2f",
                                        args.down_hotkey,
                                        trigger_label or target_columns[0],
                                        down_prob_trigger,
                                        args.trade_threshold,
                                        price_val,
                                    )
                            if args.replay_existing and current_ts_ns > 0:
                                last_trigger_at_data["down"] = current_ts_ns
                            else:
                                last_trigger_at_wall["down"] = time.time()
                            last_entry_price["down"] = (
                                price_val if np.isfinite(price_val) else last_entry_price["down"]
                            )
                            eval_stats["hotkey_trades"]["down"] += 1

                            # Simulated trade state: one-at-a-time trades with 2 minute cooldown.
                            if np.isfinite(price_val):
                                can_open = not open_trades
                                if can_open and current_ts_ns > 0 and last_entry_timestamp_ns is not None:
                                    if current_ts_ns - last_entry_timestamp_ns < int(120 * 60 * 1e9):
                                        can_open = False
                                if can_open:
                                    trade = {
                                        "id": next_trade_id,
                                        "direction": "down",
                                        "entry_price": price_val,
                                        "entry_row_index": rows_seen,
                                        "entry_timestamp_ns": current_ts_ns,
                                        "threshold": float(args.trade_threshold),
                                        "resolution": resolution,
                                        "bar_index": int(payload.get("bar_index", -1)),
                                    }
                                    next_trade_id += 1
                                    open_trades.append(trade)
                                    last_entry_timestamp_ns = current_ts_ns
                                    if eval_log_file is not None:
                                        trade_evt = {
                                            "event": "trade_open",
                                            "trade_id": trade["id"],
                                            "direction": trade["direction"],
                                            "entry_price": trade["entry_price"],
                                            "entry_row_index": trade["entry_row_index"],
                                            "entry_timestamp_ns": trade["entry_timestamp_ns"],
                                        }
                                        try:
                                            eval_log_file.write(json.dumps(trade_evt) + "\n")
                                        except Exception as exc:  # pragma: no cover - log-only
                                            logging.debug("Failed to write trade_open event: %s", exc)

                # Update any open trades with the latest price
                if open_trades and np.isfinite(price_val):
                    tp = float(getattr(args, "eval_take_profit", 10.0))
                    sl = float(getattr(args, "eval_stop_loss", 10.0))
                    tp = tp if tp > 0 else 10.0
                    sl = sl if sl > 0 else 10.0
                    for trade in list(open_trades):
                        move = price_val - trade["entry_price"]
                        result: Optional[str] = None
                        if trade["direction"] == "up":
                            if move >= tp:
                                result = "win"
                            elif move <= -sl:
                                result = "loss"
                        else:
                            # down trade: profit when price moves down
                            if -move >= tp:
                                result = "win"
                            elif -move <= -sl:
                                result = "loss"
                        if result is None:
                            continue

                        trade["exit_price"] = price_val
                        trade["exit_row_index"] = rows_seen
                        trade["exit_timestamp_ns"] = end_ns or start_ns
                        trade["result"] = result
                        trade["pips_move"] = (
                            price_val - trade["entry_price"]
                            if trade["direction"] == "up"
                            else trade["entry_price"] - price_val
                        )
                        trade["duration_rows"] = trade["exit_row_index"] - trade["entry_row_index"]
                        completed_trades.append(trade)
                        open_trades.remove(trade)

                        if eval_log_file is not None:
                            trade_evt = {
                                "event": "trade_close",
                                "trade_id": trade["id"],
                                "direction": trade["direction"],
                                "result": trade["result"],
                                "entry_price": trade["entry_price"],
                                "exit_price": trade["exit_price"],
                                "pips_move": trade["pips_move"],
                                "entry_row_index": trade["entry_row_index"],
                                "exit_row_index": trade["exit_row_index"],
                                "entry_timestamp_ns": trade["entry_timestamp_ns"],
                                "exit_timestamp_ns": trade["exit_timestamp_ns"],
                            }
                            try:
                                eval_log_file.write(json.dumps(trade_evt) + "\n")
                            except Exception as exc:  # pragma: no cover - log-only
                                logging.debug("Failed to write trade_close event: %s", exc)

                if args.replay_existing and total_rows and total_rows > 0 and progress_bar is not None:
                    progress_bar.update(1)
                    if summary_parts:
                        progress_bar.set_postfix_str(", ".join(summary_parts))
                else:
                    logging.info("%s", ", ".join(summary_parts))

                if eval_log_file is not None and args.replay_existing:
                    record: Dict[str, Any] = {
                        "event": "tick",
                        "row_index": rows_seen,
                        "bar_index": int(payload.get("bar_index", -1)),
                        "resolution": resolution,
                        "start_timestamp_ns": start_ns,
                        "end_timestamp_ns": end_ns,
                        "probs": per_target,
                        "hotkey_trades": dict(eval_stats["hotkey_trades"]),
                        "open_trades": len(open_trades),
                    }
                    try:
                        eval_log_file.write(json.dumps(record) + "\n")
                    except Exception as exc:  # pragma: no cover - log-only
                        logging.debug("Failed to write eval log record: %s", exc)

                last_idle_log = time.time()
    except KeyboardInterrupt:
        logging.info("Stopping realtime inference")
    finally:
        tail.close()
        if progress_bar is not None:
            try:
                progress_bar.close()
            except Exception:
                pass
        if eval_log_file is not None:
            try:
                eval_log_file.close()
            except Exception:
                pass
        if trigger_log_file is not None:
            try:
                trigger_log_file.close()
            except Exception:
                pass
        # Build aggregated trade statistics for offline summaries or optional JSONL logs.
        aggregated_trades = aggregate_trades_by_time(completed_trades, window_seconds=120.0)

        if eval_summary_path is not None and args.replay_existing and total_rows:
            try:
                # Build a JSON-serialisable summary structure.
                serializable_stats: Dict[str, Any] = dict(eval_stats)
                serializable_stats["rows_per_resolution"] = dict(eval_stats["rows_per_resolution"])
                serializable_stats["threshold_sweep"] = {
                    str(thr): data for thr, data in eval_stats["threshold_sweep"].items()
                }

                total_trades = len(aggregated_trades)
                wins = sum(1 for t in aggregated_trades if t.get("result") == "win")
                losses = sum(1 for t in aggregated_trades if t.get("result") == "loss")
                up_trades = sum(1 for t in aggregated_trades if t.get("direction") == "up")
                down_trades = sum(1 for t in aggregated_trades if t.get("direction") == "down")
                net_pips = float(sum(float(t.get("pips_move", 0.0)) for t in aggregated_trades))
                total_win_pips = float(
                    sum(float(t.get("pips_move", 0.0)) for t in aggregated_trades if t.get("result") == "win")
                )
                total_loss_pips = float(
                    sum(float(t.get("pips_move", 0.0)) for t in aggregated_trades if t.get("result") == "loss")
                )
                avg_pips = net_pips / total_trades if total_trades else 0.0
                win_rate = (wins / total_trades) * 100.0 if total_trades else 0.0

                # Per-half-hour win rates based on entry timestamp
                trades_per_half_hour: Dict[str, Dict[str, Any]] = {}
                for t in aggregated_trades:
                    ts_ns = int(t.get("entry_timestamp_ns") or 0)
                    if ts_ns <= 0:
                        continue
                    try:
                        dt = datetime.utcfromtimestamp(ts_ns / 1e9)
                        slot_minute = 0 if dt.minute < 30 else 30
                        slot_key = f"{dt.hour:02d}:{slot_minute:02d}"
                    except Exception:
                        slot_key = "unknown"
                    bucket = trades_per_half_hour.setdefault(
                        slot_key,
                        {"entries": 0, "wins": 0, "losses": 0, "net_pips": 0.0},
                    )
                    bucket["entries"] += 1
                    if t.get("result") == "win":
                        bucket["wins"] += 1
                    elif t.get("result") == "loss":
                        bucket["losses"] += 1
                    bucket["net_pips"] += float(t.get("pips_move", 0.0))

                serializable_stats["trades"] = {
                    "total": float(total_trades),
                    "wins": float(wins),
                    "losses": float(losses),
                    "up_trades": float(up_trades),
                    "down_trades": float(down_trades),
                    "win_rate_pct": win_rate,
                    "net_pips": net_pips,
                    "avg_pips": avg_pips,
                }
                serializable_stats["trades"]["total_win_pips"] = total_win_pips
                serializable_stats["trades"]["total_loss_pips"] = total_loss_pips
                serializable_stats["trades_per_half_hour"] = trades_per_half_hour
                serializable_stats["open_trades_remaining"] = len(open_trades)

                # Persist evaluation parameters (thresholds, TP, SL) alongside stats.
                eval_params: Dict[str, Any] = {
                    "trade_threshold": args.trade_threshold,
                    "eval_threshold_sweep": list(getattr(args, "eval_threshold_sweep", []) or []),
                    "take_profit": float(getattr(args, "eval_take_profit", 0.0)),
                    "stop_loss": float(getattr(args, "eval_stop_loss", 0.0)),
                }
                serializable_stats["eval_params"] = eval_params

                eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
                with eval_summary_path.open("w", encoding="utf-8") as f:
                    json.dump(serializable_stats, f, indent=2)
                logging.info("Offline eval summary written to %s", eval_summary_path)
            except Exception as exc:  # pragma: no cover - log-only
                logging.warning("Failed to write eval summary to %s: %s", eval_summary_path, exc)

        # Optionally write per-trade summary JSONL with aggregated trades (offline or live).
        trades_path = getattr(args, "eval_trades_jsonl", None)
        if trades_path is not None and aggregated_trades:
            try:
                trades_path = Path(trades_path)
                trades_path.parent.mkdir(parents=True, exist_ok=True)
                with trades_path.open("w", encoding="utf-8") as f_trades:
                    for t in aggregated_trades:
                        entry_ts_ns = int(t.get("entry_timestamp_ns") or 0)
                        exit_ts_ns = int(t.get("exit_timestamp_ns") or 0)
                        entry_time = (
                            datetime.utcfromtimestamp(entry_ts_ns / 1e9).isoformat()
                            if entry_ts_ns > 0
                            else None
                        )
                        exit_time = (
                            datetime.utcfromtimestamp(exit_ts_ns / 1e9).isoformat()
                            if exit_ts_ns > 0
                            else None
                        )
                        trade_record: Dict[str, Any] = {
                            "direction": t.get("direction"),
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": t.get("entry_price"),
                            "exit_price": t.get("exit_price"),
                        }
                        f_trades.write(json.dumps(trade_record) + "\n")
            except Exception as exc:  # pragma: no cover - log-only
                logging.warning("Failed to write eval trades JSONL %s: %s", trades_path, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime inference runner")
    parser.add_argument(
        "--model-bundle",
        type=Path,
        required=True,
        help="Path to exported model bundle (e.g., runs/models/phase5_model.pt)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("runs/realtime/features.jsonl"),
        help="Path to the realtime features JSONL file emitted by the Rust preprocessor",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for stdout logging",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        nargs="+",
        default=["fast", "mid", "slow"],
        help="List of resolutions to require in the feature stream. Must be: fast mid slow.",
    )
    parser.add_argument(
        "--idle-log-seconds",
        type=float,
        default=30.0,
        help="Emit a DEBUG heartbeat if no new bars arrive for this many seconds",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Seconds to wait between file checks when idle",
    )
    parser.add_argument(
        "--replay-existing",
        action="store_true",
        help="Read the feature file from the beginning instead of tailing new rows",
    )
    parser.add_argument(
        "--trade-threshold",
        type=float,
        default=None,
        help="Probability threshold for emitting trade hotkeys; disabled if omitted",
    )
    parser.add_argument(
        "--trigger-target",
        type=str,
        default=None,
        help="Target column name to watch for trade triggers (defaults to first target)",
    )
    parser.add_argument(
        "--up-hotkey",
        type=str,
        default=None,
        help="Hotkey combo for up entries (default CTRL+F11)",
    )
    parser.add_argument(
        "--down-hotkey",
        type=str,
        default=None,
        help="Hotkey combo for down entries (default CTRL+F12)",
    )
    parser.add_argument(
        "--trigger-hotkey",
        type=str,
        default=None,
        help="Deprecated alias that sets both --up-hotkey and --down-hotkey",
    )
    parser.add_argument(
        "--trigger-cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between repeated hotkey emissions per direction",
    )
    parser.add_argument(
        "--entry-min-price-move",
        type=float,
        default=0.0,
        help=(
            "Minimum absolute mid-price move (in price units, e.g. $) since the last entry "
            "in a given direction before allowing a new hotkey. 0 disables price-based gating."
        ),
    )
    parser.add_argument(
        "--entry-price-feature",
        type=str,
        default="mid_close_price",
        help=(
            "Feature name to use as the reference price for entry gating "
            "(defaults to 'mid_close_price')."
        ),
    )
    parser.add_argument(
        "--eval-trades-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path where each completed simulated trade is logged during "
            "offline evaluation (used with --replay-existing)."
        ),
    )
    parser.add_argument(
          "--eval-log-jsonl",
          type=Path,
          default=None,
          help="Optional JSONL path for per-row offline evaluation logs (used with --replay-existing).",
      )
    parser.add_argument(
        "--eval-summary-json",
        type=Path,
        default=None,
        help="Optional JSON path for aggregated offline evaluation summary (used with --replay-existing).",
    )
    parser.add_argument(
        "--eval-threshold-sweep",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional list of probability thresholds to sweep when computing offline evaluation stats; "
            "combined with --trade-threshold when provided."
        ),
    )
    parser.add_argument(
        "--eval-take-profit",
        type=float,
        default=10.0,
        help=(
            "Take-profit distance (in units of entry_price_feature) for offline-eval simulated trades "
            "when replaying existing feature files."
        ),
    )
    parser.add_argument(
        "--eval-stop-loss",
        type=float,
        default=10.0,
        help=(
            "Stop-loss distance (in units of entry_price_feature) for offline-eval simulated trades "
            "when replaying existing feature files."
        ),
    )
    parser.add_argument(
        "--trigger-log-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path where each qualifying trade trigger is logged with local and data timestamps."
        ),
    )
    parser.add_argument(
        "--pause-min-start",
        type=int,
        default=None,
        help=(
            "Optional pause window start in minutes from 00:00 (data time). "
            "When set together with --pause-min-end, hotkeying is disabled inside this window."
        ),
    )
    parser.add_argument(
        "--pause-min-end",
        type=int,
        default=None,
        help=(
            "Optional pause window end in minutes from 00:00 (data time). "
            "Supports wrapping windows when end < start (e.g. 1320-60 for 22:00-01:00)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.up_hotkey = args.up_hotkey or "CTRL+F11"
    args.down_hotkey = args.down_hotkey or "CTRL+F12"
    if args.trigger_hotkey:
        args.up_hotkey = args.trigger_hotkey
        args.down_hotkey = args.trigger_hotkey
    # Resolution filtering is currently unused by the feature parser; keep the
    # attribute for backward compatibility but default to no filtering.
    args.resolution = None
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    inference_loop(args)


if __name__ == "__main__":
    main()
