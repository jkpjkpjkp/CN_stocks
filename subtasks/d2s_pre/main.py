import polars as pl
from datetime import time
df = pl.scan_parquet('/data/share/data_raw/tick/2023-03-24.parquet').rename({'order_book_id': 'id'}).filter(
    pl.col.datetime.dt.time() < time(9, 27)
).drop(
    'open', 'high', # both all-0 except for 9:25, which equals `last`
).sink_parquet('/dev/shm/pre.pq')