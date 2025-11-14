import polars as pl


for i in range(0, 25000, 500):
    pl.scan_parquet('/dev/shm/s2.pq').filter(pl.col.id < i + 500).filter(pl.col.id >= i).sort('id', 'datetime').sink_parquet(f'/dev/shm/{i}_{i+500}.pq')
