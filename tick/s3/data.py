import polars as pl
from datetime import time

pl.scan_parquet(
    '/data/share/data_raw/tick/'
).select(
    # pl.col.num_trades.cast(pl.Int32),
    pl.col.datetime.cast(pl.Datetime('ms')),
    close=pl.when(pl.col.datetime.dt.time() < time(9, 2))
    id = pl.col.order_book_id.str.slice(0, 6).cast(pl.Int32),
).with_columns(
        pl.when(pl.col.id < 300000).then(pl.col.id                 ).otherwise(
        pl.when(pl.col.id < 600000).then(pl.col.id - 300000 + 6000 ).otherwise(
        pl.when(pl.col.id < 688000).then(pl.col.id - 600000 + 12000).otherwise(
                                         pl.col.id - 688000 + 18000)
            )
        ).cast(pl.Int16),
    (pl.col.last * 100 + 0.1).cast(pl.Int32)
).sink_parquet('/dev/shm/s3.pq')

