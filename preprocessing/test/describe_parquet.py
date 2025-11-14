import pandas as pd

print(pd.read_parquet("../data/preprocessed/training/fast/20231219.parquet").describe())
