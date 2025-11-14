import polars as pl

pl.scan_parquet(
    '/dev/shm/600000_600200.pq'
).select(
    'id', 'datetime', 'close', 'a1'
).sort('id', 'datetime'
).sink_parquet('/dev/shm/0_2h_sorted.pq')


select(
    pl.col.close - pl.col.close.shift(1).over('id')
).select(
    pl.when(pl.col.close.abs() < 64)
        .then(pl.col.close + 64)
        .otherwise(pl.lit(0))
        .cast(pl.UInt8)
).collect()[
    'close'
].to_arrow().buffers()[1]

with open('0_500.bin', 'wb') as file:
    file.write(bytes(series))