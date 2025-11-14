import polars as pl
import datetime
import numpy as np

from checkpoint.r1_30_128_p_30.t1p30 import t30m, ds
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

model = t30m.load_from_checkpoint('./checkpoint/r1_30_128_p_30/epoch=36-step=192770.ckpt')
model.to('cuda')


df = pl.scan_parquet('../data/a_1min.pq').select(
    'id', 'datetime', 'close'
).filter(
    pl.col.datetime.dt.date() >= datetime.datetime(2024,1,1)
).collect()

df = df.select(
    'id',
    'close',
    hd = pl.col.datetime.dt.date().cast(pl.String) + '_' + pl.when(pl.col.datetime.dt.time() < datetime.time(12)).then(pl.lit('a')).otherwise(pl.lit('p')),
    time = pl.col.datetime.dt.time(),
).with_columns(
    ret = pl.col.close / pl.col.close.shift(1).over('id', 'hd')
).drop_nulls()

full = df.group_by('id', 'hd').agg(pl.len()).filter(pl.col.len == 119)
df = df.filter((pl.col.id + '_' + pl.col.hd).is_in(full.select(id_hd=pl.col.id + '_' + pl.col.hd)['id_hd']))


rets = torch.ones(10)

for hd in df.select('hd').unique().sort('hd')['hd']:
    x = df.filter(pl.col.hd == hd)
    x = x[1].sort('id', 'time')['ret'].to_torch().view(-1, 119)

    data = ds.from_array(x)
    loader = DataLoader(data, batch_size=4096, shuffle=False, num_workers=32)
    for i, batch in enumerate(loader):
        ret = model.validation_step(batch, i)
        logits = ret['logits']
        logits = F.softmax(logits, dim=-1)
        exp_rank = (logits * torch.arange(0, 128).float().to(logits.device).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        # for each timestamp (dim 1), split batch (dim 0) into 10 parts based on sorted indices of exp_rank
        ranks = torch.argsort(exp_rank, dim=0)
        for b in range(10):
            rets[b] *= batch[1][ranks <= int(x.shape[0] / 10 * b)].mean()

        
    # pred = model(torch.tensor(np.searchsorted(quantiles, x[:, :-p].numpy()), device='cpu'))
    # pred = F.softmax(pred, dim=-1)
    # pred = torch.einsum('...d,d->...', pred, torch.arange(0, 128).float())
    # rank = torch.argsort(pred, dim=0)
    # ns = x.shape[0]
    # i = 118 - p
    # assert rank.shape == (x.shape[0], i+1)
    # for b in range(10):
    #     rets[b] *= x[rank[int(x.shape[0] / 10 * b):int(x.shape[0] / 10 * (b+1)), i], -p].mean()
    print(rets)
    # breakpoint()
print(rets)