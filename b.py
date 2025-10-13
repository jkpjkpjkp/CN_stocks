import torch, torch.nn as nn, torch.nn.functional as F

class forecastConfig:
    model_type = "forecast"
    is_decoder = True

    hidden_size = 64
    num_attention_heads = 4
    num_layers = 4
    fc_scale_up = 2
    dropout = 0

    window_size = 1024
    forecast_horizon = 12

    forecast_dim = 5
    forecast_weights = [1, 1, 1, 1, 0.2]


class forecastModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.LazyLinear(config.hidden_size)

        assert config.is_decoder
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * config.fc_scale_up,
                dropout=config.dropout,
            ),
            num_layers=config.num_layers,
        )

        self.forecast_head = nn.LazyLinear(config.forecast_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.forecast_head(x)
        return x

class rollingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size, forecast_horizon, group_by=None):
        import polars as pl
        df = df.sort(by=[group_by, 'datetime'])
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.group_by = group_by

        self.sizes = df.group_by(self.group_by).len()
        # drop group_id whose size is smaller than window_size + forecast_horizon
        self.sizes = self.sizes.filter(pl.col('len') >= self.window_size + self.forecast_horizon)
        self.group_ids = self.sizes[self.group_by].to_list()

        df = df.filter(pl.col(self.group_by).is_in(self.group_ids))

        breakpoint()
        self.df = df


    def __len__(self):
        return len(self.df) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        if self.group_by is not None:
            window = self.df.filter(pl.col(self.group_by) == self.df[self.group_by][idx])[idx:idx+self.window_size]
            forecast = self.df.filter(pl.col(self.group_by) == self.df[self.group_by][idx])[idx+self.window_size:idx+self.window_size+self.forecast_horizon]
        else:
            window = self.df[idx:idx+self.window_size]
            forecast = self.df[idx+self.window_size:idx+self.window_size+self.forecast_horizon]
        return window, forecast

if __name__ == '__main__':
    import polars as pl
    df = pl.read_parquet("../data/a_processed.pq")
    dataset = rollingWindowDataset(df, 10, 5, 'order_book_id')