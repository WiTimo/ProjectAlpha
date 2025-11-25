"""Evaluate classification metrics on the same days used for offline trade eval.

This script:

- takes the raw eval CSVs under `data/eval/*.csv`,
- runs the Rust preprocessing pipeline (if needed) to build per-day parquet features,
- builds a streaming cache + labels for **those days only** (no training),
- evaluates the trained TCN on them using the SAME feature ordering and scaler
  that deployment uses, and reports:

  - per-day tri-class accuracy
  - per-day binary move-vs-flat AUC
  - global aggregates across all eval days

Run from repository root, e.g.:

    python evaluation/offline_class_eval.py --model-bundle runs/models/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `src.*` imports (from training) available.
TRAIN_ROOT = REPO_ROOT / "training"
if str(TRAIN_ROOT) not in map(str, map(Path, __import__("sys").path)):
    import sys as _sys

    _sys.path.insert(0, str(TRAIN_ROOT))
    del _sys

from src.utils import load_config, setup_logging
from src.data.dataset import MultiResolutionSequenceDataset
from src.data.loader import cache_streaming_files
from src.definitions import (
    NUM_TARGET_CLASSES,
    DOWN_CLASS_INDEX,
    FLAT_CLASS_INDEX,
    UP_CLASS_INDEX,
    FEATURE_SET_COLUMNS,
)
from src.model.network import DilatedTCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute classification accuracy/AUC on eval days (data/eval/*.csv) "
            "using the trained TCN and the same scaler as deployment."
        )
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=REPO_ROOT / "runs" / "models" / "best_model.pt",
        help="Path to trained TCN bundle (exported by training/tcn.py).",
    )
    parser.add_argument(
        "--eval-csv-dir",
        type=Path,
        default=REPO_ROOT / "data" / "eval",
        help="Directory with eval CSVs (filenames' stems define the days).",
    )
    parser.add_argument(
        "--eval-feature-root",
        type=Path,
        default=REPO_ROOT / "data" / "preprocessed_eval",
        help=(
            "Directory where Rust preprocessing will write parquet features for eval days "
            "(separate from training feature_root)."
        ),
    )
    parser.add_argument(
        "--eval-cache-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "cache_eval",
        help="Cache directory for eval StreamingFileEntries (separate from training cache_dir).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Eval batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. 'cpu' or 'cuda'. Defaults to training config/system or CUDA if available.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_bundle(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = torch.load(model_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise RuntimeError(
            f"Model file {model_path} does not contain a metadata bundle. "
            "Re-run training with the updated exporter."
        )
    required_keys = ["model_state_dict", "feature_columns", "training", "scaler"]
    for key in required_keys:
        if key not in bundle:
            raise KeyError(f"Model bundle missing '{key}'. Re-export the training bundle.")
    return bundle


def build_model_from_bundle(
    bundle: Dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, int, List[str]]:
    feature_columns: List[str] = list(bundle["feature_columns"])
    training_cfg: Dict[str, Any] = dict(bundle.get("training", {}))
    model_meta: Dict[str, Any] = dict(training_cfg.get("model", {}))

    hidden_dim = int(model_meta.get("hidden_dim", 32))
    layers = int(model_meta.get("layers", 2))
    kernel_size = int(model_meta.get("kernel_size", 5))
    dropout = float(model_meta.get("dropout", 0.0))
    dilation_base = int(model_meta.get("dilation_base", 2))
    num_channels = [hidden_dim] * layers

    seq_len = int(training_cfg.get("sequence_len", 1))

    model = DilatedTCN(
        num_inputs=len(feature_columns),
        num_classes=NUM_TARGET_CLASSES,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        dilation_base=dilation_base,
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)
    model.eval()

    return model, seq_len, feature_columns


def discover_eval_stems(eval_csv_dir: Path) -> List[str]:
    if not eval_csv_dir.exists():
        raise FileNotFoundError(f"Eval CSV directory not found: {eval_csv_dir}")
    csv_paths = sorted(eval_csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {eval_csv_dir}")
    return [p.stem for p in csv_paths]


def ensure_eval_preprocessed(eval_csv_dir: Path, feature_root: Path) -> None:
    """Run the Rust preprocessing pipeline on data/eval if parquet features are absent.

    This does NOT train the model; it only computes features + targets for the eval
    days using the same feature engineering and labeling definitions as training,
    but writes them into a separate feature_root.
    """
    fast_dir = feature_root / "fast"
    if fast_dir.exists() and any(fast_dir.glob("*.parquet")):
        logging.info("Eval feature root %s already contains parquet files; skipping preprocessing.", feature_root)
        return

    preproc_root = REPO_ROOT / "preprocessing"
    base_cfg_path = preproc_root / "config.yaml"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Preprocessing config not found at {base_cfg_path}")

    with base_cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    io_cfg = cfg.get("io", {}) or {}
    io_cfg["input_path"] = str(eval_csv_dir.resolve())
    io_cfg["feature_output_path"] = str(feature_root.resolve())
    io_cfg["checkpoint_path"] = str((feature_root / ".checkpoints").resolve())
    cfg["io"] = io_cfg

    tmp_cfg_path = preproc_root / "config_eval_auto.yaml"
    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    cmd = ["cargo", "run", "--bin", "preprocessing", "--", "--config", str(tmp_cfg_path.name)]
    logging.info("Running Rust preprocessing for eval data: %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(preproc_root))
    if completed.returncode != 0:
        raise RuntimeError(f"Rust preprocessing failed with exit code {completed.returncode}")


def build_eval_entries(
    stems: List[str],
    base_config: Dict[str, Any],
    feature_root: Path,
    cache_dir: Path,
) -> tuple[List[Any], List[str]]:
    """Build StreamingFileEntry objects for eval stems using cache_streaming_files."""
    if not stems:
        return [], []

    cfg: Dict[str, Any] = {}
    for k, v in base_config.items():
        if isinstance(v, dict):
            cfg[k] = dict(v)
        else:
            cfg[k] = v

    cfg.setdefault("paths", {})
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["feature_root"] = str(feature_root)
    cfg["paths"]["cache_dir"] = str(cache_dir)

    data_cfg = cfg.setdefault("data", {})
    resolutions: List[str] = list(data_cfg.get("resolutions") or ["fast"])
    feature_set = data_cfg.get("feature_set", "phase5")

    base_cols = FEATURE_SET_COLUMNS[feature_set]
    col_map: Dict[str, List[str]] = {res: base_cols for res in resolutions}

    lookahead_bars = int(cfg.get("training", {}).get("label_lookahead_bars", 0))

    entries = cache_streaming_files(
        stems,
        cfg,
        resolutions,
        col_map,
        lookahead_bars=lookahead_bars if lookahead_bars > 0 else None,
    )

    if not entries:
        logging.error(
            "No eval cache entries could be built for stems=%s under feature_root=%s cache_dir=%s",
            stems,
            feature_root,
            cache_dir,
        )
        return [], []

    entry_names = [e.stem for e in entries]
    return entries, entry_names


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channel_means: torch.Tensor,
    channel_stds: torch.Tensor,
    entry_names: List[str],
) -> None:
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    per_day_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_day_targets: Dict[str, List[np.ndarray]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
                meta = None

            x = x.to(device)

            # Apply the same per-channel standardization as deployment
            # (using training-time scaler) before feeding the model.
            x = (x - channel_means.view(1, -1, 1)) / channel_stds.view(1, -1, 1)

            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            targets = y[:, 0].cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

            if meta is not None:
                meta_np = meta.cpu().numpy()
                file_indices = meta_np[:, 0].astype(int)
                for i, f_idx in enumerate(file_indices):
                    if 0 <= f_idx < len(entry_names):
                        day = entry_names[f_idx]
                    else:
                        day = f"file_{f_idx}"
                    per_day_probs[day].append(probs[i : i + 1])
                    per_day_targets[day].append(targets[i : i + 1])

    if not all_targets:
        logging.error("No validation samples found in loader; nothing to evaluate.")
        return

    probs_all = np.concatenate(all_probs, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)

    # Tri-class accuracy
    preds_all = probs_all.argmax(axis=1)
    global_acc = accuracy_score(targets_all, preds_all)

    # Binary move-vs-flat AUC (same definition as training)
    move_probs_all = probs_all[:, DOWN_CLASS_INDEX] + probs_all[:, UP_CLASS_INDEX]
    move_targets_all = (targets_all != FLAT_CLASS_INDEX).astype(int)
    try:
        global_auc = roc_auc_score(move_targets_all, move_probs_all)
    except ValueError:
        global_auc = float("nan")

    logging.info("===== Global Eval over %d samples =====", targets_all.shape[0])
    logging.info("Global tri-class accuracy: %.4f", global_acc)
    logging.info("Global move-vs-flat AUC:   %.4f", global_auc)

    logging.info("===== Per-day metrics =====")
    for day in sorted(per_day_targets.keys()):
        probs_day = np.concatenate(per_day_probs[day], axis=0)
        targets_day = np.concatenate(per_day_targets[day], axis=0)
        preds_day = probs_day.argmax(axis=1)
        acc_day = accuracy_score(targets_day, preds_day)
        move_probs_day = probs_day[:, DOWN_CLASS_INDEX] + probs_day[:, UP_CLASS_INDEX]
        move_targets_day = (targets_day != FLAT_CLASS_INDEX).astype(int)
        try:
            auc_day = roc_auc_score(move_targets_day, move_probs_day)
        except ValueError:
            auc_day = float("nan")
        logging.info(
            "Day %s | samples=%d | accuracy=%.4f | move AUC=%.4f",
            day,
            targets_day.shape[0],
            acc_day,
            auc_day,
        )


def main() -> None:
    args = parse_args()
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = load_config()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        configured = config.get("system", {}).get("device", "cuda")
        if configured == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            logging.warning("CUDA requested in config but not available; falling back to CPU.")
        else:
            device = torch.device(configured)

    bundle = load_bundle(args.model_bundle)
    model, seq_len, feature_columns = build_model_from_bundle(bundle, device)

    scaler_cfg = bundle.get("scaler", {})
    means_by_name: Dict[str, float] = scaler_cfg.get("means", {}) or {}
    stds_by_name: Dict[str, float] = scaler_cfg.get("stds", {}) or {}

    means_vec = np.array([means_by_name.get(col, 0.0) for col in feature_columns], dtype=np.float32)
    stds_vec = np.array([stds_by_name.get(col, 1.0) for col in feature_columns], dtype=np.float32)
    stds_vec = np.where(stds_vec == 0, 1.0, stds_vec)

    channel_means = torch.from_numpy(means_vec).to(device)
    channel_stds = torch.from_numpy(stds_vec).to(device)

    training_cfg = bundle.get("training", {})
    resolutions = list(training_cfg.get("data", {}).get("resolutions", ["fast"]))

    logging.info("Loaded model from %s", args.model_bundle)
    logging.info(
        "Model input_channels=%d, seq_len=%d, resolutions=%s",
        len(feature_columns),
        seq_len,
        resolutions,
    )

    stems = discover_eval_stems(args.eval_csv_dir)
    logging.info("Eval days (from %s): %s", args.eval_csv_dir, ", ".join(stems))

    # 1) Ensure eval parquet features exist (Rust preprocessing).
    args.eval_feature_root.mkdir(parents=True, exist_ok=True)
    ensure_eval_preprocessed(args.eval_csv_dir, args.eval_feature_root)

    # 2) Build streaming cache + labels for eval days only (Python cache).
    args.eval_cache_dir.mkdir(parents=True, exist_ok=True)
    entries, entry_names = build_eval_entries(stems, config, args.eval_feature_root, args.eval_cache_dir)
    if not entries:
        logging.error("Aborting: no cached entries found for eval days.")
        return

    dataset = MultiResolutionSequenceDataset(entries, seq_len, resolutions)
    if len(dataset) == 0:
        logging.error("Dataset built from cached eval entries is empty; nothing to evaluate.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config.get("system", {}).get("num_workers", 0)),
    )

    evaluate(model, loader, device, channel_means, channel_stds, entry_names)


if __name__ == "__main__":
    main()

