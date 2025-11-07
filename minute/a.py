import polars as pl
import datetime
import numpy as np

df = pl.scan_parquet('../data/a_1min.pq').select(
    'id', 'datetime', 'close'
)

def a(df, filename):
    df = df.collect().select(
        'close',
        id_hd = pl.col.id + '_' + pl.col.datetime.dt.date().cast(pl.String) + '_' + pl.when(pl.col.datetime.dt.time() < datetime.time(12)).then(pl.lit('a')).otherwise(pl.lit('p')),
        time = pl.col.datetime.dt.time(),
    ).sort('id_hd', 'time').select(
        'id_hd', 'time',
        ret = pl.col.close / pl.col.close.shift(1).over('id_hd')
    ).drop_nulls()

    full = df.group_by('id_hd').agg(pl.len()).filter(pl.col.len == 119)

    df = df.filter(pl.col.id_hd.is_in(full['id_hd']))

    df = df.group_by(pl.col.datetime.dt.date(), (pl.col.datetime.dt.time() < datetime.time(12)))
    df = df.sort(pl.col.datetime.dt.date(), (pl.col.datetime.dt.time() < datetime.time(12)), 'id', 'datetime')
    data = df['ret'].to_numpy()
    time = df['datetime'].to_numpy()

    np.save(filename, data)

a(df.filter(pl.col.datetime.dt.date() < datetime.datetime(2024,1,1)), '../data/train.npy')
a(df.filter(pl.col.datetime.dt.date() >= datetime.datetime(2024,1,1)), '../data/eval.npy')