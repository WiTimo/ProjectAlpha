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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_TARGET_CLASSES = 3


def load_model(bundle_path: Path) -> Dict[str, Any]:
    bundle = torch.load(bundle_path, map_location="cpu")
    model_state = bundle.get("model_state_dict") or bundle.get("state_dict")
    if model_state is None:
        raise KeyError(
            "Model bundle missing 'model_state_dict'/'state_dict'. Re-export training bundle."
        )
    scaler = bundle.get("scaler", {})
    feature_columns = bundle.get("feature_columns", [])
    if not feature_columns:
        raise KeyError(
            "Model bundle missing feature column metadata; re-export the training bundle."
        )
    target_columns = bundle.get("target_columns") or (
        [bundle.get("target_column")] if bundle.get("target_column") else []
    )
    num_targets = max(1, len(target_columns) or 1)

    from training.tcn import DilatedTCN

    training_cfg = bundle.get("training", {})
    model_cfg = dict(training_cfg.get("model", {}))
    model_cfg.setdefault("output_dim", num_targets * NUM_TARGET_CLASSES)
    sequence_len = max(1, int(training_cfg.get("sequence_len") or 1))
    model = DilatedTCN(num_features=len(feature_columns), **model_cfg)
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
        "target_columns": target_columns,
        "means": means,
        "stds": stds,
        "sequence_len": sequence_len,
        "num_targets": num_targets,
    }


def standardize(features: Dict[str, float], feature_columns: Iterable[str], means, stds) -> np.ndarray:
    vector = np.array([features.get(col, 0.0) for col in feature_columns], dtype=np.float32)
    vector = (vector - means) / stds
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
    if expected_resolution and resolution != expected_resolution:
        logging.debug(
            "Dropping row due to resolution mismatch (got=%s expected=%s)",
            resolution,
            expected_resolution,
        )
        return None, "resolution-mismatch"

    features = payload.get("features")
    if not isinstance(features, dict):
        logging.debug("JSON line missing 'features' dictionary: %s", line)
        return None, "missing-features"

    return payload, None


def inference_loop(args: argparse.Namespace) -> None:
    bundle = load_model(Path(args.model_bundle))
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
    drop_counts: Counter[str] = Counter()

    trigger_idx: Optional[int] = None
    trigger_label: Optional[str] = None
    enable_triggers = args.trade_threshold is not None and num_targets > 0
    if enable_triggers:
        trigger_label = args.trigger_target or target_columns[0]
        if trigger_label is None:
            enable_triggers = False
        elif trigger_label not in target_columns:
            logging.warning("Trigger target '%s' not found; disabling hotkeys", trigger_label)
            enable_triggers = False
        else:
            trigger_idx = target_columns.index(trigger_label)
    hotkey_emitter = HotkeyEmitter() if enable_triggers else None
    if enable_triggers and hotkey_emitter and not hotkey_emitter.available:
        logging.warning("Hotkey emission unavailable on this platform; disabling triggers")
        enable_triggers = False
        hotkey_emitter = None
    last_trigger_at = {"up": 0.0, "down": 0.0}

    logging.info(
        "Model loaded (features=%d, sequence_len=%d, targets=%s)",
        len(feature_columns),
        sequence_len,
        ", ".join(target_columns) or "n/a",
    )

    features_path = Path(args.features)
    tail = FeatureTail(features_path, start_at_end=not args.replay_existing)
    logging.info("Watching %s for feature rows", features_path)
    last_idle_log = time.time()

    try:
        while True:
            lines = tail.read_new_lines()
            if not lines:
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
                parsed, drop_reason = parse_feature_line(line, args.resolution)
                if drop_reason:
                    drop_counts[drop_reason] += 1
                if not parsed:
                    continue

                payload = parsed
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
                    logits = logits.view(1, num_targets, NUM_TARGET_CLASSES)
                    probs = torch.softmax(logits, dim=-1)

                summary_parts: List[str] = []
                for idx, name in enumerate(target_columns):
                    target_prob = probs[0, idx]
                    down_prob = float(target_prob[0].item())
                    up_prob = float(target_prob[2].item())
                    summary_parts.append(f"{name}:down={down_prob:.3f},up={up_prob:.3f}")

                    if enable_triggers and hotkey_emitter and idx == trigger_idx:
                        now = time.time()
                        if up_prob >= args.trade_threshold and now - last_trigger_at["up"] >= args.trigger_cooldown:
                            if hotkey_emitter.press(args.up_hotkey):
                                logging.info(
                                    "Hotkey %s emitted for %s up=%.3f (>= %.3f)",
                                    args.up_hotkey,
                                    name,
                                    up_prob,
                                    args.trade_threshold,
                                )
                            last_trigger_at["up"] = now
                        if down_prob >= args.trade_threshold and now - last_trigger_at["down"] >= args.trigger_cooldown:
                            if hotkey_emitter.press(args.down_hotkey):
                                logging.info(
                                    "Hotkey %s emitted for %s down=%.3f (>= %.3f)",
                                    args.down_hotkey,
                                    name,
                                    down_prob,
                                    args.trade_threshold,
                                )
                            last_trigger_at["down"] = now

                logging.info("%s", ", ".join(summary_parts))
                last_idle_log = time.time()
    except KeyboardInterrupt:
        logging.info("Stopping realtime inference")
    finally:
        tail.close()


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
        "--resolution",
        type=str,
        default="fast",
        help="Filter incoming rows to the expected resolution (fast/mid/slow)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.up_hotkey = args.up_hotkey or "CTRL+F11"
    args.down_hotkey = args.down_hotkey or "CTRL+F12"
    if args.trigger_hotkey:
        args.up_hotkey = args.trigger_hotkey
        args.down_hotkey = args.trigger_hotkey
    resolution_filter = args.resolution.lower().strip() if args.resolution else None
    if resolution_filter == "all":
        resolution_filter = None
    args.resolution = resolution_filter
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    inference_loop(args)


if __name__ == "__main__":
    main()
