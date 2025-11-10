import polars as pl
import datetime
import numpy as np
import torch


rank = torch.load('rank.pt')
l = len(rank) // 10
rank = rank[-l:]

t = 50

df = pl.scan_parquet('~/Downloads/2025-01-02.parquet').select(
    'datetime', 'close',
    id=pl.col.order_book_id,
).filter(
    pl.col.datetime.dt.time().eq(datetime.time(10,t-1)) | 
    pl.col.datetime.dt.time().eq(datetime.time(10,t))
).sort('id', 'datetime').with_columns(
    'id',
    ret = pl.col.close / pl.col.close.shift(1).over('id'),
).filter(
    pl.col.datetime.dt.time() == datetime.time(10,t)
).sort('id')

ret = df.collect()['ret'].to_torch()
assert len(ret == l)
print(ret[rank].mean())