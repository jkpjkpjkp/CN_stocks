import polars as pl
from dataclasses import dataclass
import datetime
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange
from ..prelude.model import dummyLightning, apply_rotary_emb

class mha(dummyLightning):
    def __init__(self, config, pos_emb=True):
        super().__init__()
        self.pos_emb = pos_emb
        self.qkv = nn.Linear(config.hidden_size, 3 * config.num_heads * config.head_dim)
        self.readout = nn.Linear(config.num_heads * config.head_dim, config.hidden_size)
        
        device = config.device
        channel_range = torch.arange(0, config.hidden_size, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (10000 ** (channel_range / config.hidden_size))
        t = torch.arange(config.seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        self.cos, self.sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()

    def forward(self, x, embeddings=None):
        q, k, v = torch.chunk(self.qkv(x), 3, -1)
        if self.pos_emb:
            assert embeddings is None
            q = apply_rotary_emb(q, self.cos, self.sin)
            k = apply_rotary_emb(k, self.cos, self.sin)
        else:
            q = q + embeddings
            k = k + embeddings
        b = x.shape[0]
        
        shape = (b, -1, self.config.head_dim, self.config.num_heads)
        q = rearrange(q.view(*shape), 'b l d h -> b h l d')
        k = rearrange(k.view(*shape), 'b l d h -> b h l d')
        v = rearrange(v.view(*shape), 'b l d h -> b h l d')

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.pos_emb)
        y = y.transpose(1, 2).contiguous()
        y = y.view(b, -1, self.config.num_heads * self.config.head_dim)

        return self.readout(y)

class mhaa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_attn = mha(config)
        self.cross_attn = mha(config, pos_emb=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.SiLU(),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
        )
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x, embeddings):
        x = x + self.time_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x).transpose(0, 1), embeddings).transpose(0, 1)
        x = x + self.ffn(x)
        return x

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
        ).with_columns(
            open = pl.col.open / pl.col.close,
            high = pl.col.high / pl.col.close,
            low = pl.col.low / pl.col.close,
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
        self.num_ids = len(ids)
        id_map = {row['oid']: row['index'] for row in ids.rows(named=True)}

        @dataclass
        class per_day:
            date: datetime.date
            ids: torch.Tensor
            data: torch.Tensor
        
        date_groups = [x for x in df.group_by('date', maintain_order=True)]
        data_list = []
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
            data_list.append(per_day(date, ids, data))

        self.lens = torch.tensor([len(x.ids) for x in data_list])
        self.cum_lens = torch.cumsum(self.lens, dim=0)
        self.cum_lens = torch.cat([torch.tensor([0]), self.cum_lens], dim=0)
        
        contents = torch.concat((x.data for x in data_list), dim=0)
        self.m = contents.mean(dim=(0, 1)).view(None, None, -1)
        self.s = contents.std(dim=(0, 1)).view(None, None, -1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ret = (self.data[self.cum_lens[idx]: self.cum_lens[idx+1]] - self.m) / self.s
        ret = torch.asinh(ret)
    
    def param_prepare(self, config):
        self.emb = nn.ModuleDict({
            'direct': nn.Linear(config.window_minutes * 5, config.hidden_dim),
            'l1': nn.Linear(config.window_minutes * 5, config.intermediate_dim),
            'l2': nn.Linear(config.intermediate_dim, config.hidden_dim),
        })
        self.class_embed = torch.nn.Embedding(self.num_ids, config.id_dim)

        self.layers = nn.ModuleList([mhaa(config) for _ in range(config.num_layers)])

        self.readout = nn.Linear(config.hidden_dim * 240 // config.window_minutes, config.num_quantiles)

    
    def embed(self, x):
        x = x.view(-1, 240 * self.config.window_days // self.config.window_minutes, self.config.window_minutes * 5)
        direct = self.emb['direct'](x)
        l1 = F.silu(self.emb['l1'](x))
        l2 = self.emb['l2'](l1)
        return direct + l2
    
    def forward(self, x, ids):
        x = self.embed(x)
        class_embeddings = self.class_embed(ids)
        for layer in self.layers:
            x = layer(x, class_embeddings)
        return self.readout(x.view(-1, self.config.window_days, self.config.hidden_dim * 240 // self.config.window_minutes))
        
    def step(self, x):
        y_hat = self(x.data, x.ids)
        return {
            'y_hat': y_hat,
        }


if __name__ == '__main__':
    @dataclass
    class debug_config:
        debug_data = True
        window_days = 5
    x = cross(debug_config)
    print(x.step(x[42]))
    breakpoint()