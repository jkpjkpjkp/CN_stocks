import polars as pl
import os
import subprocess

for i in range(0, 25000, 500):
    if os.path.exists(f'/dev/shm/{i}_{i+500}.pq'):
        pl.scan_parquet(f'/dev/shm/{i}_{i+500}.pq').drop('num_trades').sink_parquet(f'/dev/shm/{i:06d}_{i+500:06d}.pq')
        subprocess.run(['rm', f'/dev/shm/{i}_{i+500}.pq'])
