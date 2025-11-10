import polars as pl
import datetime
import numpy as np

from minute.b import tm
import torch
from torch.nn import functional as F

model = tm.load_from_checkpoint('./ml-runs/models/epoch=17-step=93780.ckpt')
model.to('cpu')
model.attn1.to('cpu')
model.attn2.to('cpu')

t = 49

df = pl.scan_parquet('~/Downloads/2025-01-02.parquet').select(
    'datetime', 'close',
    id=pl.col.order_book_id,
).filter(
    pl.col.datetime.dt.time() < datetime.time(10,t)
)

print(df.collect())

df = df.collect().select(
    'id',
    'close',
    'datetime',
    hd = pl.col.datetime.dt.date().cast(pl.String) + '_' + pl.when(pl.col.datetime.dt.time() < datetime.time(12)).then(pl.lit('a')).otherwise(pl.lit('p')),
    time = pl.col.datetime.dt.time(),
).sort('id', 'datetime').select(
    'id', 'hd', 'time',
    ret = pl.col.close / pl.col.close.shift(1).over('id', 'hd')
).drop_nulls()

full = df.group_by('id', 'hd').agg(pl.len()).filter(pl.col.len == t+28)

df = df.filter((pl.col.id + '_' + pl.col.hd).is_in(full.select(id_hd=pl.col.id + '_' + pl.col.hd)['id_hd']))

quant = np.load('q.npy')

rets = torch.ones(10)
x = df.sort('id', 'time')['ret'].to_torch().view(-1, t+28)
pred = model(torch.tensor(np.searchsorted(quant, x.numpy()), device='cpu'))
pred = F.softmax(pred, dim=-1)
pred = torch.einsum('...d,d->...', pred, torch.arange(0, 128).float())
rank = torch.argsort(pred, dim=0)

torch.save(rank[:, -1], 'rank.pt')