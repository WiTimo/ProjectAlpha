#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime


def filter_file(input_path: Path, output_path: Path, start_min: float, end_min: float) -> None:
    """
    Reads a CSV (semicolon-separated, no header) and keeps only rows where
    the timestamp (column 3, format YYYYMMDDHHMMSS) is inside the given
    time-of-day window, specified in minutes from midnight.

    Supports both:
      - Normal interval: start_min <= end_min (same day)
      - Wrapped interval over midnight: start_min > end_min
        (e.g. 1020 -> 930 means 17:00 to 15:30 next day)
    """

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return

    output_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split(";")
        if len(parts) < 3:
            continue

        ts_str = parts[2]

        # Parse timestamp
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except ValueError:
            # Skip invalid timestamps
            continue

        # Convert time-of-day to minutes since midnight
        minutes_since_midnight = ts.hour * 60 + ts.minute + ts.second / 60.0

        # Determine if within interval, with midnight wrap handling
        if start_min <= end_min:
            # Normal interval (same day), e.g. 600 -> 900
            in_range = start_min <= minutes_since_midnight <= end_min
        else:
            # Wrapped interval over midnight, e.g. 1020 -> 930
            # Means: [start_min, 1440) U [0, end_min]
            in_range = (
                minutes_since_midnight >= start_min
                or minutes_since_midnight <= end_min
            )

        if in_range:
            output_lines.append(line)

    # Skip writing if nothing matches
    if not output_lines:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(output_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter all .csv files by absolute time-of-day "
            "(in minutes from midnight). Supports windows crossing midnight."
        )
    )
    parser.add_argument("--input-dir", "-i", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--start-min", "-s", type=float, required=True)
    parser.add_argument("--end-min", "-e", type=float, required=True)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    for csv_path in sorted(input_dir.glob("*.csv")):
        out_path = output_dir / csv_path.name
        print(f"Processing {csv_path.name} -> {out_path.name}")
        filter_file(csv_path, out_path, args.start_min, args.end_min)


if __name__ == "__main__":
    main()
