import polars as pl
import datetime
import numpy as np

from minute.b import tm
import torch
from torch.nn import functional as F

model = tm.load_from_checkpoint('./ml-runs/models/epoch=16-step=88570.ckpt')
model.to('cpu')
model.attn1.to('cpu')
model.attn2.to('cpu')

df = pl.scan_parquet('../data/a_1min.pq').select(
    'id', 'datetime', 'close'
).filter(
    pl.col.datetime.dt >= datetime.datetime(2024,1,17,9,30) &
    pl.col.datetime.dt <= datetime.datetime(2024,1,17,9,30) &
)

df = df.collect().select(
    'id',
    'close',
    hd = pl.col.datetime.dt.date().cast(pl.String) + '_' + pl.when(pl.col.datetime.dt.time() < datetime.time(12)).then(pl.lit('a')).otherwise(pl.lit('p')),
    time = pl.col.datetime.dt.time(),
).sort('id', 'hd', 'time').select(
    'id', 'hd', 'time',
    ret = pl.col.close / pl.col.close.shift(1).over('id', 'hd')
).drop_nulls()

full = df.group_by('id', 'hd').agg(pl.len()).filter(pl.col.len == 119)

df = df.filter((pl.col.id + '_' + pl.col.hd).is_in(full.select(id_hd=pl.col.id + '_' + pl.col.hd)['id_hd']))

quant = np.load('q.npy')

rets = torch.ones(10)
for x in df.group_by('hd'):
    x = x[1].sort('id', 'time')['ret'].to_torch().view(-1, 119)
    p = 118
    pred = model(torch.tensor(np.searchsorted(quant, x[:, :-p].numpy()), device='cpu'))
    pred = F.softmax(pred, dim=-1)
    pred = torch.einsum('...d,d->...', pred, torch.arange(0, 128).float())
    rank = torch.argsort(pred, dim=0)
    ns = x.shape[0]
    i = 118 - p
    assert rank.shape == (x.shape[0], i+1)
    for b in range(10):
        rets[b] *= x[rank[int(x.shape[0] / 10 * b):int(x.shape[0] / 10 * (b+1)), i], -p].mean()
    print(rets)
    breakpoint()
print(rets)