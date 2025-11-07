import polars as pl
import datetime
import numpy as np

from minute.b import tm
import torch

model = tm.load_from_checkpoint('./ml-runs/models/epoch=13-step=72940.ckpt')
model.to('cpu')
model.attn1.to('cpu')
model.attn2.to('cpu')

df = pl.scan_parquet('../data/a_1min.pq').select(
    'id', 'datetime', 'close'
).filter(
    pl.col.datetime.dt.date() >= datetime.datetime(2024,1,1)
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
    pred = model(torch.tensor(np.searchsorted(quant, x[:, :-1].numpy()), device='cpu'))
    rank = torch.argsort(pred, dim=0)
    ns = x.shape[0]
    for i in range(118):
        for b in range(10):
            rets[b] *= x[rank[int(x.shape[0] / 10 * b):int(x.shape[0] / 10 * (b+1)), i], i+1].mean()
    print(rets)
print(rets)