import pandas as pd
from pathlib import Path

stem = "20240506"  # <-- use a stem that exists in data/preprocessed_eval/fast

# 1) Offline preprocessed parquet (eval)
parq_path = Path(f"data/preprocessed_testing_output/fast/{stem}.parquet")
df_parq = pd.read_parquet(parq_path)

# 2) Realtime preprocessed features JSONL
feat_path = Path("runs/realtime/features.jsonl")
df_rt = pd.read_json(feat_path, lines=True)

# Keep only fast resolution rows
df_rt_fast = df_rt[df_rt["resolution"] == "fast"].copy()

# If timestamp lives at top-level (e.g. 'start_timestamp_ns'):
# adjust this to match the actual column name you saw in step 2
ts_col = "start_timestamp_ns"
if ts_col not in df_rt_fast.columns and "timestamps" in df_rt_fast.columns:
    # Example if timestamps are nested; adapt as needed
    df_ts = pd.json_normalize(df_rt_fast["timestamps"])
    df_rt_fast[ts_col] = df_ts["start_timestamp_ns"]

# Expand feature dict into columns
feat_cols = pd.json_normalize(df_rt_fast["features"])
df_rt_fast = pd.concat([df_rt_fast[[ts_col]].reset_index(drop=True),
                        feat_cols.reset_index(drop=True)], axis=1)

# Align on timestamp
df_parq = df_parq.rename(columns={"start_timestamp_ns": ts_col})
merged = df_parq.merge(df_rt_fast, on=ts_col, suffixes=("_parq", "_rt"))

print("Merged rows:", len(merged), "parquet rows:", len(df_parq), "realtime rows:", len(df_rt_fast))

# Compare a few key columns
for col in ["mid_close_price", "mid_high_price", "mid_low_price"]:
    col_parq = f"{col}_parq"
    col_rt = f"{col}_rt"
    if col_parq in merged.columns and col_rt in merged.columns:
        diff = (merged[col_parq] - merged[col_rt]).abs()
        print(f"\nColumn {col}:")
        print(diff.describe())
