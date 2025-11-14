import polars as pl

series = pl.scan_parquet(
    '/dev/shm/0_500.pq'
).select(
    'id', 'datetime', 'close',
).select(
    pl.col.close - pl.col.close.shift(1).over('id')
).select(
    pl.when(pl.col.close.abs() < 128)
        .then(pl.col.close + 128)
        .otherwise(pl.lit(0))
        .cast(pl.UInt8)
).collect()[
    'close'
].to_arrow().buffers()[1]

with open('0_500.bin', 'wb') as file:
    file.write(bytes(series))