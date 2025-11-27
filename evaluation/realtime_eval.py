"""Offline evaluation runner that mirrors realtime deployment.

It reuses the Rust realtime preprocessor and Python inference pipeline,
feeding historical raw CSV data into the same stack but at accelerated speed.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical raw CSVs through realtime stack")
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=REPO_ROOT / "runs" / "models" / "best_model.pt",
        help="Path to exported model bundle (same as deployment)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "eval",
        help="Directory with NinjaTrader-style raw CSV files to replay",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "evaluation",
        help="Directory where evaluation artefacts (features + logs) are stored",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        nargs="+",
        default=["fast", "mid", "slow"],
        help="Resolutions to emit and consume (must match training/deployment)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=20.0,
        help="Approximate realtime speedup factor for log playback (reserved for future fine-tuning)",
    )
    parser.add_argument(
        "--norm-state-dir",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "preprocessed"
        / ".checkpoints"
        / "normalization"
        / "latest",
        help="Directory with normalization snapshots (fast.json/mid.json/slow.json)",
    )
    parser.add_argument(
        "--keep-features",
        action="store_true",
        help="Keep generated features.jsonl for inspection instead of deleting it at the end",
    )
    parser.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help=(
            "If a features_eval.jsonl file already exists in the output-dir, reuse it "
            "instead of regenerating features. If the file does not exist, features "
            "will be generated as usual."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for evaluation runner logs",
    )
    parser.add_argument(
        "--sweep-trade-thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional list of trade thresholds to sweep. If provided together with "
            "--sweep-tp-grid, every combination of threshold x (tp, sl) is evaluated. "
            "Deprecated when using --sweep-config-yaml, which allows explicit combos."
        ),
    )
    parser.add_argument(
        "--sweep-tp-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional grid of take-profit / stop-loss distances (in price units). "
            "If provided together with --sweep-trade-thresholds, every (tp, sl) pair "
            "from this grid is evaluated for each threshold. Deprecated when using "
            "--sweep-config-yaml, which allows explicit combos."
        ),
    )
    parser.add_argument(
        "--sweep-config-yaml",
        type=Path,
        default=None,
        help=(
            "Optional YAML file specifying explicit combinations of trade thresholds "
            "and TP/SL distances. If provided, this takes precedence over "
            "--sweep-trade-thresholds and --sweep-tp-grid. Expected format:\n\n"
            "  combinations:\n"
            "    - threshold: 0.50\n"
            "      take_profit: 10.0\n"
            "      stop_loss: 10.0\n"
            "    - threshold: 0.55\n"
            "      tp: 30.0\n"
            "      sl: 20.0\n"
        ),
    )
    parser.add_argument(
        "--inference-extra-args",
        type=str,
        nargs=argparse.REMAINDER,
        help="Additional args passed through to deployment/realtime_inference.py",
    )
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw eval directory not found: {args.raw_dir}")
    if not any(args.raw_dir.glob("*.csv")):
        raise FileNotFoundError(f"No CSV files found in {args.raw_dir}")
    if not args.model_bundle.exists():
        raise FileNotFoundError(f"Model bundle not found: {args.model_bundle}")
    if not args.norm_state_dir.exists():
        logging.warning("Normalization state dir not found: %s", args.norm_state_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    logging.debug("Running: %s", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def build_concatenated_log(raw_dir: Path, tmp_dir: Path) -> Path:
    """Create a single replay log file by concatenating eval CSVs.

    The Rust realtime binary already interprets timestamps from each line,
    so we can just stream lines quickly to simulate faster markets.
    """
    output = tmp_dir / "eval_replay.csv"
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    total_files = len(csv_paths)
    logging.info(
        "Step 1/3: Building replay log from %d CSV files in %s",
        total_files,
        raw_dir,
    )

    with output.open("w", encoding="utf-8", newline="") as out_f:
        for csv_path in tqdm(csv_paths, desc="Eval: building replay log", unit="file", ncols=100):
            with csv_path.open("r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line)

    try:
        size_bytes = output.stat().st_size
        size_mb = size_bytes / (1024 * 1024) if size_bytes > 0 else 0.0
    except OSError:
        size_mb = 0.0

    logging.info("Step 1/3 done: replay log built at %s (%.1f MB)", output, size_mb)
    return output


def run_realtime_preprocessor(replay_log: Path, features_path: Path, args: argparse.Namespace) -> None:
    """Run the existing Rust realtime binary on the concatenated log.

    This mirrors `npm run realtime`, only with a different --source/--emit.
    """
    resolutions = ",".join(args.resolutions)
    preprocessing_root = REPO_ROOT / "preprocessing"

    # Prefer running the existing compiled binary directly to avoid rebuild
    # conflicts on Windows when the executable is in use.
    bin_path = preprocessing_root / "target" / "debug" / ("realtime.exe" if sys.platform == "win32" else "realtime")
    if not bin_path.exists():
        build_cmd = ["cargo", "build", "--bin", "realtime"]
        run_cmd(build_cmd, cwd=preprocessing_root)

    realtime_cmd = [
        str(bin_path),
        "--resolutions",
        resolutions,
        "--source",
        str(replay_log),
        "--emit",
        str(features_path),
        "--from-start",
        "--poll-ms",
        "1",
        "--norm-state-dir",
        str(args.norm_state_dir),
    ]

    logging.info("Step 2/3: Starting realtime preprocessor to generate features (this can take a while)")
    proc = subprocess.Popen(realtime_cmd, cwd=preprocessing_root)
    last_log = time.time()
    last_size = -1
    try:
        while True:
            ret = proc.poll()
            try:
                size_bytes = features_path.stat().st_size if features_path.exists() else 0
            except OSError:
                size_bytes = 0

            now = time.time()
            if size_bytes != last_size or (now - last_log) >= 10.0:
                mb = size_bytes / (1024 * 1024) if size_bytes > 0 else 0.0
                logging.info(
                    "Step 2/3: Generating features... current features file size ~ %.1f MB",
                    mb,
                )
                last_size = size_bytes
                last_log = now

            if ret is not None:
                if ret != 0:
                    raise RuntimeError(
                        f"Realtime preprocessor exited with code {ret}: "
                        + " ".join(realtime_cmd)
                    )
                break
            time.sleep(5.0)
    finally:
        # Best-effort wait to avoid zombies; ignore errors here.
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass


def _load_sweep_config(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep config YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}

    if isinstance(data, dict) and "combinations" in data:
        raw_combos = data["combinations"]
    elif isinstance(data, list):
        raw_combos = data
    else:
        raise ValueError(
            "Sweep config YAML must be either a list of combinations or a mapping "
            "with a top-level 'combinations' key."
        )

    combos: List[Dict[str, float]] = []
    for idx, entry in enumerate(raw_combos):
        if not isinstance(entry, dict):
            raise ValueError(f"Combination at index {idx} is not a mapping: {entry!r}")

        if "threshold" not in entry:
            raise ValueError(f"Combination at index {idx} is missing 'threshold': {entry!r}")

        # Accept both verbose and short keys for TP/SL.
        tp_val = entry.get("take_profit", entry.get("tp"))
        sl_val = entry.get("stop_loss", entry.get("sl"))
        if tp_val is None or sl_val is None:
            raise ValueError(
                f"Combination at index {idx} must define 'take_profit'/'tp' and "
                f"'stop_loss'/'sl': {entry!r}"
            )

        thr = float(entry["threshold"])
        tp = float(tp_val)
        sl = float(sl_val)
        combos.append({"threshold": thr, "take_profit": tp, "stop_loss": sl})

    if not combos:
        raise ValueError(f"No valid combinations found in sweep config: {path}")

    return combos


def run_inference(features_path: Path, args: argparse.Namespace) -> None:
    """Run the realtime inference watcher on the generated features.

    If sweep_trade_thresholds and sweep_tp_grid are provided, this will
    evaluate every combination of:

        threshold in sweep_trade_thresholds
        tp in sweep_tp_grid
        sl in sweep_tp_grid

    and write a separate eval_log + eval_summary for each combination.
    Otherwise it falls back to a single evaluation run (legacy mode).
    """
    inference_script = REPO_ROOT / "deployment" / "realtime_inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found at {inference_script}")

    base_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Common, fixed part of the command for all runs.
    base_cmd_prefix: list[str] = [
        sys.executable,
        str(inference_script),
        "--model-bundle",
        str(args.model_bundle),
        "--features",
        str(features_path),
        "--resolutions",
        *args.resolutions,
        "--replay-existing",
    ]

    extra = args.inference_extra_args or []

    thresholds = args.sweep_trade_thresholds or []
    tp_grid = args.sweep_tp_grid or []

    # YAML-configured sweep mode: explicit (threshold, tp, sl) combinations.
    if args.sweep_config_yaml is not None:
        combos = _load_sweep_config(args.sweep_config_yaml)
        index_entries: List[Dict[str, Any]] = []

        for combo in combos:
            thr = float(combo["threshold"])
            tp = float(combo["take_profit"])
            sl = float(combo["stop_loss"])

            thr_str = f"{thr:.2f}".replace(".", "p")
            tp_str = f"{tp:.1f}".replace(".", "p")
            sl_str = f"{sl:.1f}".replace(".", "p")

            log_name = f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}.jsonl"
            summary_name = (
                f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}_summary.json"
            )
            trades_name = (
                f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}_trades.jsonl"
            )

            eval_log_jsonl = args.output_dir / log_name
            eval_summary_json = args.output_dir / summary_name
            eval_trades_jsonl = args.output_dir / trades_name

            cmd: list[str] = [
                *base_cmd_prefix,
                "--eval-log-jsonl",
                str(eval_log_jsonl),
                "--eval-summary-json",
                str(eval_summary_json),
                "--eval-trades-jsonl",
                str(eval_trades_jsonl),
            ]

            cmd += list(extra)
            cmd += [
                "--trade-threshold",
                str(thr),
                "--eval-take-profit",
                str(tp),
                "--eval-stop-loss",
                str(sl),
            ]

            logging.info(
                "Running YAML sweep combo: threshold=%.3f, tp=%.3f, sl=%.3f -> %s",
                thr,
                tp,
                sl,
                eval_summary_json,
            )
            run_cmd(cmd, cwd=REPO_ROOT / "deployment")

            index_entries.append(
                {
                    "threshold": thr,
                    "take_profit": tp,
                    "stop_loss": sl,
                    "log_path": str(eval_log_jsonl),
                    "summary_path": str(eval_summary_json),
                    "trades_path": str(eval_trades_jsonl),
                }
            )

        try:
            import json

            index_path = args.output_dir / f"offline_eval_{base_timestamp}_sweep_index.json"
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(index_entries, f, indent=2)
            logging.info("YAML sweep index written to %s", index_path)
        except Exception as exc:  # pragma: no cover - log-only
            logging.warning("Failed to write YAML sweep index: %s", exc)
        return

    # Multi-parameter sweep mode: thresholds x (tp, sl) grid.
    if thresholds and tp_grid:
        index_entries = []
        for thr in thresholds:
            for tp in tp_grid:
                for sl in tp_grid:
                    thr_str = f"{thr:.2f}".replace(".", "p")
                    tp_str = f"{tp:.1f}".replace(".", "p")
                    sl_str = f"{sl:.1f}".replace(".", "p")

                    log_name = f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}.jsonl"
                    summary_name = (
                        f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}_summary.json"
                    )
                    trades_name = (
                        f"offline_eval_{base_timestamp}_thr{thr_str}_tp{tp_str}_sl{sl_str}_trades.jsonl"
                    )

                    eval_log_jsonl = args.output_dir / log_name
                    eval_summary_json = args.output_dir / summary_name
                    eval_trades_jsonl = args.output_dir / trades_name

                    cmd: list[str] = [
                        *base_cmd_prefix,
                        "--eval-log-jsonl",
                        str(eval_log_jsonl),
                        "--eval-summary-json",
                        str(eval_summary_json),
                        "--eval-trades-jsonl",
                        str(eval_trades_jsonl),
                    ]

                    # Let caller still provide extra args, but ensure that our
                    # trade-threshold / TP / SL are appended last so they win.
                    cmd += list(extra)
                    cmd += [
                        "--trade-threshold",
                        str(thr),
                        "--eval-take-profit",
                        str(tp),
                        "--eval-stop-loss",
                        str(sl),
                    ]

                    logging.info(
                        "Running sweep combo: threshold=%.3f, tp=%.3f, sl=%.3f -> %s",
                        thr,
                        tp,
                        sl,
                        eval_summary_json,
                    )
                    run_cmd(cmd, cwd=REPO_ROOT / "deployment")

                    index_entries.append(
                        {
                            "threshold": thr,
                            "take_profit": tp,
                            "stop_loss": sl,
                            "log_path": str(eval_log_jsonl),
                            "summary_path": str(eval_summary_json),
                            "trades_path": str(eval_trades_jsonl),
                        }
                    )

        # Write an index file that lists all combinations and their artefacts.
        try:
            import json

            index_path = args.output_dir / f"offline_eval_{base_timestamp}_grid_index.json"
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(index_entries, f, indent=2)
            logging.info("Multi-parameter sweep index written to %s", index_path)
        except Exception as exc:  # pragma: no cover - log-only
            logging.warning("Failed to write sweep index: %s", exc)
        return

    # Legacy single-run mode: behave exactly as before.
    eval_log_jsonl = args.output_dir / f"offline_eval_{base_timestamp}.jsonl"
    eval_summary_json = args.output_dir / f"offline_eval_{base_timestamp}_summary.json"
    eval_trades_jsonl = args.output_dir / f"offline_eval_{base_timestamp}_trades.jsonl"

    cmd = [
        *base_cmd_prefix,
        "--eval-log-jsonl",
        str(eval_log_jsonl),
        "--eval-summary-json",
        str(eval_summary_json),
        "--eval-trades-jsonl",
        str(eval_trades_jsonl),
        "--eval-threshold-sweep",
        "0.50",
        "0.60",
        "0.70",
        "0.80",
        "0.90",
        *extra,
    ]

    logging.info("Offline eval logs will be written to %s", eval_log_jsonl)
    logging.info("Offline eval summary will be written to %s", eval_summary_json)

    run_cmd(cmd, cwd=REPO_ROOT / "deployment")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_paths(args)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        replay_log = build_concatenated_log(args.raw_dir, tmp_dir)

        features_path = args.output_dir / "features_eval.jsonl"
        generate_features = True

        if args.reuse_existing_features and features_path.exists():
            logging.info(
                "Reusing existing features file at %s (skipping preprocessing)", features_path
            )
            generate_features = False
        elif features_path.exists():
            logging.info("Removing existing features file at %s", features_path)
            features_path.unlink()

        if generate_features:
            logging.info("Step 2/3: Running realtime preprocessor on replay log")
            run_realtime_preprocessor(replay_log, features_path, args)
        else:
            logging.info("Step 2/3: Skipped realtime preprocessor because features already exist")

        logging.info("Step 3/3: Running offline inference on features")
        run_inference(features_path, args)

        if generate_features and not args.keep_features:
            try:
                features_path.unlink()
                logging.info("Temporary features file removed: %s", features_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
