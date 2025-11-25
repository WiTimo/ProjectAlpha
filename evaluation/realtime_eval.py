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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for evaluation runner logs",
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


def run_inference(features_path: Path, args: argparse.Namespace) -> None:
    """Run the realtime inference watcher on the generated features.

    We reuse deployment/realtime_inference.py and ask it to replay the
    entire file from the beginning instead of tailing new lines.
    """
    inference_script = REPO_ROOT / "deployment" / "realtime_inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found at {inference_script}")

    # Derive evaluation log paths under the configured output directory.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_log_jsonl = args.output_dir / f"offline_eval_{timestamp}.jsonl"
    eval_summary_json = args.output_dir / f"offline_eval_{timestamp}_summary.json"

    base_cmd: list[str] = [
        sys.executable,
        str(inference_script),
        "--model-bundle",
        str(args.model_bundle),
        "--features",
        str(features_path),
        "--resolutions",
        *args.resolutions,
        "--replay-existing",
        "--eval-log-jsonl",
        str(eval_log_jsonl),
        "--eval-summary-json",
        str(eval_summary_json),
        "--eval-threshold-sweep",
        "0.50",
        "0.60",
        "0.70",
        "0.80",
        "0.90",
    ]

    extra = args.inference_extra_args or []
    cmd = base_cmd + extra

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
        if features_path.exists():
            logging.info("Removing existing features file at %s", features_path)
            features_path.unlink()

        logging.info("Step 2/3: Running realtime preprocessor on replay log")
        run_realtime_preprocessor(replay_log, features_path, args)

        logging.info("Step 3/3: Running offline inference on generated features")
        run_inference(features_path, args)

        if not args.keep_features:
            try:
                features_path.unlink()
                logging.info("Temporary features file removed: %s", features_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
