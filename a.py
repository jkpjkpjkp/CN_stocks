SET_ME_TO_FALSE = False
TRAIN_CUTOFF = '2019-01-01'
VALIDATION_CUTOFF = '2022-01-01'


import polars as pl
from datetime import date

if SET_ME_TO_FALSE:
    df = pl.scan_parquet("../data/a_30min.pq").head().collect()
else:
    df = pl.read_parquet("../data/a_30min.pq")

main_cols = ['open', 'close', 'high', 'low', 'volume']

for col in main_cols:
    df = df.drop(col)
    df = df.rename({f'{col}_post': col})

assert df.columns == ['total_turnover', 'num_trades', 'open', 'close', 'high', 'low', 'volume', 'order_book_id', 'datetime']

train_df = df.filter(pl.col('datetime') < date.fromisoformat(TRAIN_CUTOFF))

original_df = df
df = train_df

data_stats = {}

# normalize to 0 mean 1 var
# TODO: should try by-order_book_id normalization
for col in df.columns[:-2]:
    mean, std = pl.col(col).mean(), pl.col(col).std()

    df = df.with_columns(
        (pl.col(col) - mean) / std
    )
    data_stats[col] = {
        'mean': mean,
        'std': std,
    }
assert list(data_stats.keys()) == ['total_turnover', 'num_trades', 'open', 'close', 'high', 'low', 'volume']

if SET_ME_TO_FALSE == False:
    df.write_parquet("../data/a_processed.pq")

assert SET_ME_TO_FALSE == False