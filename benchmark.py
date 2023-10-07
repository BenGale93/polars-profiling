"""Benchmarking the summary process."""
# %%
import time
from pathlib import Path

import polars as pl

from polars_profiling import run_profile

# %%
taxi_df = pl.read_parquet("data/yellow_tripdata_2023-01.parquet")

# %%
start = time.time()
profile = run_profile(taxi_df)
duration = time.time() - start

Path("pl.html").write_text(profile.to_html())

print(f"{duration=}")
