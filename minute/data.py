import polars as pl
import datetime
import numpy as np
from copy import deepcopy

df = pl.scan_parquet('../data/a_1min.pq').select(
    'id', 'datetime', 'close'
)

def a(df, filename):
    df = df.collect().select(
        'close',
        'datetime',
        id_d = pl.col.id + '_' + pl.col.datetime.dt.date().cast(pl.String),
    ).sort('id_d', 'datetime').select(
        'id_d', 'datetime',
        ret = pl.col.close / pl.col.close.shift(1).over('id_d')
    ).drop_nulls()

    full = df.group_by('id_d').agg(pl.len()).filter(pl.col.len == 238)
    df = df.filter(pl.col.id_d.is_in(full['id_d']))

    df = df.sort('id', 'datetime')
    data = df['ret'].to_numpy()

    np.save(filename, data)

if __name__ == '__main__':
    a(df.filter(pl.col.datetime.dt.date() < datetime.datetime(2024,1,1)), '../data/fullday_t.npy')
    a(df.filter(pl.col.datetime.dt.date() >= datetime.datetime(2024,1,1)), '../data/fullday_v.npy')

def arr_to_batches(arr):
    y = 
