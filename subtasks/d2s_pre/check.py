import polars as pl
df = pl.read_parquet('/dev/shm/pre.pq')

pl.Config.set_tbl_cols(-1)
print(df)