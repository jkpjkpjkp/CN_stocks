SET_ME_TO_FALSE = True
import polars as pl

if SET_ME_TO_FALSE:
    df = pl.scan_parquet("../data/a_30min.pq").head().collect()
else:
    df = pl.read_parquet("../data/a_30min.pq")

df = df.drop(['open', 'close', 'high', 'low', 'volume'])
df = df.rename({
    'open_post': 'open',
    'close_post': 'close',
    'high_post': 'high',
    'low_post': 'low',
    'volume_post': 'volume',
})

assert df.columns == ['total_turnover', 'num_trades', 'open', 'close', 'high', 'low', 'volume', 'order_book_id', 'datetime']


from pytorch_forecasting.data.timeseries import TimeSeries

df = df.with_columns(
    time_idx=pl.col('datetime').dt.timestamp("ms") // (1000 * 60 * 30)
)

print(df['time_idx'])

dataset = TimeSeries(
    df.to_pandas(),
    time="time_idx",
    target=["open", "close", "high", "low"],
    group=["order_book_id"],
)

breakpoint()