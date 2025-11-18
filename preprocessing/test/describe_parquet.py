#!/usr/bin/env python3
"""
parquet_report.py

Read a .parquet file and write a detailed text report about its rows and columns.

Usage:
    python parquet_report.py --input data.parquet --output report.txt
"""

import argparse
import sys
from pathlib import Path
from textwrap import indent
from typing import Any, cast

import pandas as pd


def format_number(x):
    """Format numbers nicely for the report."""
    if pd.isna(x):
        return "NaN"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
        return f"{int(x):,}"
    # floats
    return f"{x:,.6g}"


def summarize_basic(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== BASIC DATASET INFO ===")
    lines.append(f"Rows:    {df.shape[0]:,}")
    lines.append(f"Columns: {df.shape[1]:,}")
    lines.append("")
    return "\n".join(lines)


def summarize_columns_overview(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== COLUMN OVERVIEW ===")
    lines.append(f"{'Name':30} {'Dtype':15} {'Nulls':>12} {'Null%':>8} {'Unique':>12}")

    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        nulls = s.isna().sum()
        null_pct = (nulls / total_rows * 100) if total_rows > 0 else 0.0
        nunique = s.nunique(dropna=True)
        lines.append(
            f"{str(col)[:30]:30} "
            f"{str(s.dtype)[:15]:15} "
            f"{nulls:12,d} "
            f"{null_pct:7.2f}% "
            f"{nunique:12,d}"
        )

    lines.append("")
    return "\n".join(lines)


def summarize_numeric_column(s: pd.Series) -> str:
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    lines = []
    lines.append(f"Non-null count: {desc.get('count', 0):,}")
    if "mean" in desc:
        lines.append(f"Mean:          {format_number(desc['mean'])}")
    if "std" in desc:
        lines.append(f"Std dev:       {format_number(desc['std'])}")
    for key in ["min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"]:
        if key in desc:
            label = key.rstrip("%").ljust(3) if "%" in key else key
            lines.append(f"{label:>4}:          {format_number(desc[key])}")
    return "\n".join(lines)


def summarize_non_numeric_column(s: pd.Series, max_values: int = 10) -> str:
    lines = []
    non_null = s.dropna()
    lines.append(f"Non-null count: {len(non_null):,}")
    lines.append(f"Unique values:  {non_null.nunique():,}")
    lines.append("Top values (value: count):")

    vc = non_null.value_counts().head(max_values)
    if vc.empty:
        lines.append("  [no non-null values]")
    else:
        for val, cnt in vc.items():
            v_str = str(val)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            lines.append(f"  {v_str!r}: {cnt:,}")
    return "\n".join(lines)


def summarize_small_cardinality_distribution(s: pd.Series) -> str:
    """
    For columns with <= 5 unique non-null values, output a full distribution,
    including NaN, with counts and percentages.
    """
    lines = []
    total = len(s)
    lines.append("Value distribution (including NaN):")
    vc = s.value_counts(dropna=False)
    if vc.empty:
        lines.append("  [no values]")
        return "\n".join(lines)

    for val, cnt in vc.items():
        pct = (cnt / total * 100) if total > 0 else 0.0
        if pd.isna(cast(Any, val)):
            val_repr = "NaN"
        else:
            val_repr = repr(val)
        lines.append(f"  {val_repr}: {cnt:,} ({pct:.2f}%)")
    return "\n".join(lines)
    return "\n".join(lines)


def summarize_columns_detailed(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== DETAILED COLUMN STATS ===")

    for col in df.columns:
        s = df[col]
        lines.append("")
        lines.append(f"--- Column: {col} ---")
        lines.append(f"Dtype: {s.dtype}")
        lines.append(f"Total rows: {len(s):,}")
        nulls = s.isna().sum()
        null_pct = (nulls / len(s) * 100) if len(s) > 0 else 0.0
        lines.append(f"Nulls: {nulls:,} ({null_pct:.2f}%)")

        unique_non_null = s.nunique(dropna=True)
        lines.append(f"Unique non-null values: {unique_non_null:,}")

        # If small cardinality (<=5), print full distribution
        if unique_non_null <= 5:
            dist_text = summarize_small_cardinality_distribution(s)
            lines.append(indent(dist_text, "  "))

        # Then add the usual numeric/non-numeric stats
        if pd.api.types.is_numeric_dtype(s):
            detail = summarize_numeric_column(s)
        else:
            detail = summarize_non_numeric_column(s)

        lines.append(indent(detail, "  "))

    lines.append("")
    return "\n".join(lines)


def sample_rows(df: pd.DataFrame, n: int = 5) -> str:
    lines = []
    lines.append("=== SAMPLE ROWS ===")
    if len(df) == 0:
        lines.append("[DataFrame is empty]")
        return "\n".join(lines)

    n = min(n, len(df))
    sample = df.head(n)
    lines.append(sample.to_string(max_cols=20, max_colwidth=40))
    lines.append("")
    return "\n".join(lines)


def build_report(df: pd.DataFrame) -> str:
    parts = [
        summarize_basic(df),
        summarize_columns_overview(df),
        summarize_columns_detailed(df),
        sample_rows(df),
    ]
    return "\n".join(parts)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a detailed report from a .parquet file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input .parquet file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output .txt report file",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    report = build_report(df)

    try:
        output_path.write_text(report, encoding="utf-8")
    except Exception as e:
        print(f"Error writing report file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
