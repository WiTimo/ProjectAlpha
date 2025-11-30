Simple deployment that uses raw tri-class predictions (down/flat/up) from the
TCN model to trigger hotkeys, without the probability-based trade simulator.

It reads the same realtime feature stream as the main deployment, but on each
new inference step it:

- Runs the model to obtain per-target class probabilities.
- Picks the most likely class for the primary target.
- If the best class is "up" or "down" and at least 10 minutes have passed since
  the last entry, it emits the corresponding hotkey once.
- If the best class is "flat" or the cooldown has not expired, it does nothing.

Usage example (assuming `npm run realtime` is already running):

```bash
cd deployment-simple
python simple_realtime_inference.py \
  --model-bundle ../runs/models/best_model.pt \
  --features ../runs/realtime/features.jsonl \
  --trigger-hotkey CTRL+F11 \
  --trigger-cooldown-global 600
```

The `npm run deployment:simple` script is wired to a reasonable default command.
*** Add File: deployment-simple/simple_realtime_inference.py
"""Simple realtime deployment using raw tri-class predictions.

This variant does not simulate trades or use probability thresholds. It:

- Streams Phase-8 features from the Rust realtime preprocessor.
- Runs the TCN model on a rolling window.
- Looks at the argmax class for the primary target:
  - 0 => "down"
  - 1 => "flat"
  - 2 => "up"
- Emits a single hotkey for "up" or "down" decisions subject to a global
  cooldown (default 10 minutes). Within that cooldown window, additional
  entries are suppressed to avoid repeated triggers.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.src.model.network import DilatedTCN

NUM_TARGET_CLASSES = 3


def load_model(bundle_path: Path) -> Dict[str, Any]:
    """Load the TCN model bundle exported by training/tcn.py."""
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
    }


def standardize(features: Dict[str, float], feature_columns: Iterable[str], means, stds) -> np.ndarray:
    """Standardize a single feature row using training-style z-score + clipping."""
    vector = np.array([features.get(col, 0.0) for col in feature_columns], dtype=np.float32)
    vector = (vector - means) / stds
    vector = np.nan_to_num(vector, copy=False)
    # Use same default clip as training (10.0) for stability.
    np.clip(vector, -10.0, 10.0, out=vector)
    return vector


class HotkeyEmitter:
    """Very small Windows hotkey sender for trade triggers."""

    def __init__(self) -> None:
        self.available = False
        self._user32 = None
        if platform.system() != "Windows":
            logging.warning("HotkeyEmitter is only supported on Windows.")
            return
        try:
            self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            self.available = True
        except Exception as exc:  # pragma: no cover - platform dependent
            logging.warning("Failed to initialize HotkeyEmitter: %s", exc)

    def press(self, combo: str) -> bool:
        """Send a key combination like 'CTRL+F11' using keybd_event."""
        if not self.available or not self._user32 or not combo:
            return False

        parts = [p.strip().upper() for p in combo.split("+") if p.strip()]
        if not parts:
            logging.warning("Ignoring empty hotkey combo")
            return False

        vk_map = {
            "CTRL": 0x11,
            "SHIFT": 0x10,
            "ALT": 0x12,
            "F1": 0x70,
            "F2": 0x71,
            "F3": 0x72,
            "F4": 0x73,
            "F5": 0x74,
            "F6": 0x75,
            "F7": 0x76,
            "F8": 0x77,
            "F9": 0x78,
            "F10": 0x79,
            "F11": 0x7A,
            "F12": 0x7B,
        }

        modifiers = []
        main_key = None
        for token in parts:
            if token in ("CTRL", "SHIFT", "ALT"):
                modifiers.append(vk_map[token])
            else:
                if token not in vk_map:
                    logging.warning("Unsupported hotkey token '%s'", token)
                    return False
                main_key = vk_map[token]

        if main_key is None:
            logging.warning("Hotkey combo '%s' missing main key", combo)
            return False

        KEYEVENTF_KEYUP = 0x0002

        for vk in modifiers:
            self._user32.keybd_event(vk, 0, 0, 0)
        self._user32.keybd_event(main_key, 0, 0, 0)
        self._user32.keybd_event(main_key, 0, KEYEVENTF_KEYUP, 0)
        for vk in reversed(modifiers):
            self._user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple realtime deployment using raw tri-class predictions.")
    parser.add_argument(
        "--model-bundle",
        type=Path,
        required=True,
        help="Path to the TCN model bundle exported by training/tcn.py",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to the realtime JSONL feature stream produced by the Rust preprocessor.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["fast", "mid", "slow"],
        help="List of resolutions used during training (for logging only).",
    )
    parser.add_argument(
        "--up-hotkey",
        type=str,
        default=None,
        help="Hotkey combo for up entries (default CTRL+F11).",
    )
    parser.add_argument(
        "--down-hotkey",
        type=str,
        default=None,
        help="Hotkey combo for down entries (default CTRL+F12).",
    )
    parser.add_argument(
        "--trigger-hotkey",
        type=str,
        default=None,
        help="Convenience alias that sets both --up-hotkey and --down-hotkey to the same combo.",
    )
    parser.add_argument(
        "--trigger-cooldown-global",
        type=float,
        default=600.0,
        help="Global cooldown in seconds between any two hotkey entries (default 600 = 10 minutes).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--trigger-log-file",
        type=Path,
        default=None,
        help="Optional JSONL path where each emitted trigger is logged.",
    )
    return parser.parse_args()


def inference_loop(args: argparse.Namespace) -> None:
    bundle = load_model(args.model_bundle)
    model: torch.nn.Module = bundle["model"]
    feature_columns: List[str] = bundle["feature_columns"]
    means = bundle["means"]
    stds = bundle["stds"]
    sequence_len: int = int(bundle["sequence_len"])
    num_targets: int = int(bundle["num_targets"])
    target_columns: List[str] = bundle["target_columns"]

    device = torch.device("cpu")
    model.to(device)

    up_hotkey = args.up_hotkey or "CTRL+F11"
    down_hotkey = args.down_hotkey or "CTRL+F12"
    if args.trigger_hotkey:
        up_hotkey = args.trigger_hotkey
        down_hotkey = args.trigger_hotkey

    hotkey_emitter = HotkeyEmitter()
    if not hotkey_emitter.available:
        logging.warning("Hotkey emission unavailable; running in no-op mode.")

    trigger_log_file = None
    if args.trigger_log_file is not None:
        trigger_log_file = args.trigger_log_file.open("a", encoding="utf-8")

    last_any_trigger_wall: float = 0.0
    cooldown_global = float(args.trigger_cooldown_global)

    window: deque[np.ndarray] = deque(maxlen=sequence_len)
    rows_seen = 0

    logging.info("Starting simple realtime inference.")
    logging.info("Model targets: %s", ", ".join(target_columns))
    logging.info("Using global cooldown: %.1fs", cooldown_global)

    # Stream features forever
    with args.features.open("r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.05)
                f.seek(pos)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logging.debug("Skipping invalid JSON line")
                continue

            if not isinstance(payload, dict):
                continue

            features = payload.get("features") or {}
            if not isinstance(features, dict):
                continue

            rows_seen += 1
            vector = standardize(features, feature_columns, means, stds)
            window.append(vector)

            if len(window) < sequence_len:
                continue

            stacked = np.stack(list(window), axis=0)
            channels_first = np.ascontiguousarray(stacked.T)
            tensor = torch.from_numpy(channels_first).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                if logits.dim() == 2 and logits.shape[1] == NUM_TARGET_CLASSES:
                    logits = logits.view(1, 1, NUM_TARGET_CLASSES)
                    effective_targets = 1
                elif logits.dim() == 2 and logits.shape[1] == num_targets * NUM_TARGET_CLASSES:
                    logits = logits.view(1, num_targets, NUM_TARGET_CLASSES)
                    effective_targets = num_targets
                else:
                    logging.warning("Unexpected logits shape: %s", tuple(logits.shape))
                    continue
                probs = torch.softmax(logits, dim=-1)

            # Use the first target as primary decision source.
            target_idx = 0
            target_name = target_columns[target_idx] if target_idx < len(target_columns) else "t0"
            target_prob = probs[0, target_idx]
            class_probs = target_prob.cpu().numpy()
            class_idx = int(class_probs.argmax())

            # 0 = down, 1 = flat, 2 = up
            class_label = ["down", "flat", "up"][class_idx]
            logging.info(
                "Row %d | target=%s probs=[down=%.3f, flat=%.3f, up=%.3f] -> %s",
                rows_seen,
                target_name,
                class_probs[0],
                class_probs[1],
                class_probs[2],
                class_label.upper(),
            )

            if class_label == "flat":
                continue

            now_wall = time.time()
            if now_wall - last_any_trigger_wall < cooldown_global:
                # Still in global cooldown window; skip.
                continue

            hotkey = up_hotkey if class_label == "up" else down_hotkey

            if hotkey_emitter.available and hotkey_emitter.press(hotkey):
                last_any_trigger_wall = now_wall
                logging.info(
                    "Emitted %s hotkey for class=%s (cooldown %.1fs)",
                    hotkey,
                    class_label.upper(),
                    cooldown_global,
                )

                if trigger_log_file is not None:
                    record: Dict[str, Any] = {
                        "event": "simple_trigger",
                        "class": class_label,
                        "hotkey": hotkey,
                        "local_time": datetime.now().isoformat(),
                        "row_index": rows_seen,
                    }
                    try:
                        trigger_log_file.write(json.dumps(record) + "\n")
                        trigger_log_file.flush()
                    except Exception as exc:  # pragma: no cover - log-only
                        logging.debug("Failed to write trigger log record: %s", exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    inference_loop(args)


if __name__ == "__main__":
    main()
