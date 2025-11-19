import polars as pl
from dataclasses import dataclass
import datetime
import torch
from dataclasses import dataclass
from ..prelude.model import dummyLightning

class cross(dummyLightning):
    
    def __init__(self, config):
        super().__init__(config)
        self.data_prepare(config)

    def data_prepare(self, config):
        df = pl.scan_parquet('../data/a_1min.pq')
        if config.debug_data:
            df = df.head(100000)
        df = df.drop_nulls().with_columns(
            date = pl.col('datetime').dt.date(),
            time = pl.col('datetime').dt.time(),
        ).collect()

        # every minute from 9:30 to 11:30, from 13:00 to 15:00, to use as join
        times = pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(1970, 1, 1, 9, 30), end=pl.datetime(1970, 1, 1, 9, 30), interval='1m', eager=True
            )
        }).vstack(pl.DataFrame({
            'datetime': pl.datetime_range(
                start=pl.datetime(1970, 1, 1, 1, 0), end=pl.datetime(1970, 1, 1, 3, 0), interval='1m', eager=True
            )
        })).select(
            time = pl.col('datetime').dt.time(),
        )

        ids = df.select(oid = pl.col.id).unique().with_row_index()
        id_map = {row['oid']: row['index'] for row in ids.rows(named=True)}

        @dataclass
        class per_day:
            date: datetime.date
            ids: torch.Tensor
            data: torch.Tensor
        
        date_groups = [x for x in df.group_by('date', maintain_order=True)]
        self.data = []
        for i in range(len(date_groups) - config.window_days + 1):
            arr = date_groups[i: i + config.window_days]
            df = arr[0][1]
            for x in arr[1:]:
                df = df.vstack(x[1])
            unique_ids = df.select('id').unique()
            dates = arr[0][1].head(1)
            for x in arr[1:]:
                dates = dates.vstack(x[1].head(1))
            dt = dates.join(times, how='cross')
            full_times_per_id = unique_ids.join(dt, how='cross')

            df = full_times_per_id.join(
                df,
                on=['id', 'time'],
                how='left'
            )
            date = df['date'][0]
            ids = torch.tensor([id_map[id] for id in df['id'].unique()])
            data = df.select(pl.col('open', 'high', 'low', 'close', 'volume')).to_numpy()
            assert data.shape[-1] == 5
            data = data.reshape(len(unique_ids), -1, 5)
            self.data.append(per_day(date, ids, data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def param_prepare(self):
        pass
    
    def step(self, item):
        pass


if __name__ == '__main__':
    @dataclass
    class debug_config:
        debug_data = True
        window_days = 5
    print(cross(debug_config)[42])
