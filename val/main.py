import polars as pl
from datetime import date, time
import torch
from typing import Dict


class val:
    """
    a robust, no-leak, validation process
    
    
    """
    def __init__(self, freq='30min', rolling=False):
        self.freq = freq
        self.data = pl.scan_parquet('../data/a_1min.pq').filter(
            pl.col.datetime.dt.date().ge(date(2023, 1, 1)) &
            pl.col.datetime.dt.date().lt(date(2024, 1, 1))
        ).collect()
        if not rolling:
            if freq == '30min':
                self.groups = self.data.with_columns(
                    group = pl.col.datetime.dt.date().cast(pl.String) + '_' + pl.col.datetime.dt.hour().cast(pl.String) + '_' + (pl.col.datetime.dt.minute().cast(pl.Int32) // 30).cast(pl.String),
                ).group_by('group')
        self.len = len(self.groups.agg(pl.count()))
        self.set_id(0)
    
    def __len__(self):
        return self.len - 1

    def __getitem__(self, idx):
        return self.groups[idx][2]
    
    def set_id(self, idx):
        self.id = idx
        self.public = self[self.id]
        self.private = self[self.id + 1]

    def reset(self):
        self.set_id(0)
        self.l1 = []
        self.l2 = []
        self.cross_corr = []
        self.p20_ret = []
    
    def step(self):
        self.set_id(self.id + 1)

    def interact(self, predictions: Dict[str, float]):
        rets = self.private.group_by(
            'order_book_id'
        ).agg(
            ret = pl.col.close.last() / pl.col.open.first() - 1
        )

        l1 = 0
        l2 = 0
        Ex = 0
        Ey = 0
        Exy = 0
        rets = torch.zeros(20)

        # sort prediction keys based on their values
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        ranks = {key: i for i, (key, _) in enumerate(sorted_preds)}

        xs, ys = [], []

        for row in rets.iter_rows(named=True):
            x = ranks[row['order_book_id']]
            y = row['ret']
            xs.append(x)
            ys.append(y)


        
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
