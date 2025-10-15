import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="1d_p_1d")
parser.add_argument("-b", type=int, default=32768, help="Batch size")
parser.add_argument("-n", type=int, default=2, help="N_DAYS_LOOKBACK")
parser.add_argument("-e", type=int, default=1000, help="Number of epochs")
args = parser.parse_args()



N_DAYS_LOOKBACK = args.n
batch_size = args.b
def get_df(N_DAYS_LOOKBACK, head=None):
    print("--- 1. Data Loading ---")
    df = pl.scan_parquet("../data/a_30min.pq")

    df = df.drop(['open', 'high', 'low', 'close', 'volume']).rename({
        'open_post': 'open',
        'high_post': 'high',
        'low_post': 'low',
        'close_post': 'close',
        'volume_post': 'volume',
    })

    if head is not None:
        df = df.head(head)

    df = df.collect()

    print("--- 2. Polars Processing ---")
    df = df.with_columns(
        avg_price=(pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4
    ).drop(
        ["open", "high", "low", "close"]
    )#.with_columns(
    #     traded_capita=pl.col("volume") * pl.col("avg_price")
    # )

    # market_avg = df.group_by("datetime").agg(pl.col("traded_capita").sum().alias("total_capita"), (pl.col("avg_price") * pl.col("traded_capita")).sum().alias("total_price"))
    # market_avg = market_avg.select(
    #     "datetime",
    #     pl.col("total_price") / pl.col("total_capita").alias("market_price")
    # )

    # df = df.join(market_avg, on="datetime", how="left")
    # df.with_columns(
    #     price=pl.col("avg_price") / pl.col("market_price")
    # ).drop(["avg_price", "traded_capita"])

    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date")
    )

    daily_prices = (
        df
        .sort(["order_book_id", "datetime"])
        .group_by(["order_book_id", "date"])
        .agg(pl.col("avg_price").alias("prices"))
    )

    assert daily_prices["prices"].list.len().max() == 8, daily_prices["prices"].list.len().max()
    daily_prices = daily_prices.filter(pl.col("prices").list.len() == 8)



    # Create expressions for the past N days of prices.
    # We want the sequence to be chronological [day_T-13, day_T-12, ..., day_T].
    # shift(0) is the current day, shift(13) is 13 days ago.
    price_sequence_expr = [
        pl.col("prices").shift(i).over("order_book_id")
        for i in range(N_DAYS_LOOKBACK - 1, -1, -1)
    ]

    # Combine the expressions to create the final DataFrame
    # 'price_sequence' will be our feature (X): a list of 14 daily price lists
    # 'next_day_prices' will be our target (y): the list of prices for the next day
    sequences_df = daily_prices.with_columns(
        pl.concat_list(price_sequence_expr).alias("price_sequence"),
        pl.col("prices").shift(-1).over("order_book_id").alias("next_day_prices"),
    ).drop_nulls()

    # As a sanity check, ensure all our sequences have the correct length
    sequences_df = sequences_df.filter(pl.col("price_sequence").list.len() == N_DAYS_LOOKBACK * 8)

    print(f"Created {len(sequences_df)} samples of {N_DAYS_LOOKBACK}-day sequences.")
    return sequences_df

df = get_df(N_DAYS_LOOKBACK, head=int(1e6))
print("--- 3. Custom Dataset and DataLoader ---")
class StockDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.sequences = df["price_sequence"].to_numpy()
        self.targets = df["next_day_prices"].to_numpy()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Retrieve the raw sequence and target lists for the given index
        price_sequence = self.sequences[idx]
        next_day_prices = self.targets[idx]

        x_np = np.array(price_sequence, dtype=np.float32)
        y_np = np.array(next_day_prices, dtype=np.float32)

        # Normalize based on the last price of the last day in the input sequence
        val = x_np[-1]
        
        # Avoid division by zero, though unlikely with price data
        if val == 0:
            val = 1 

        x_normalized = (x_np / val) - 1
        y_normalized = (y_np / val) - 1

        scale_up = 1 / 0.030319097

        # Convert normalized numpy arrays to PyTorch tensors
        return torch.from_numpy(x_normalized * scale_up), torch.from_numpy(y_normalized * scale_up)

dataset = StockDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

print("--- 4. Model ---")
from model import Transformer, ModelArgs

targs = ModelArgs(
    block_size=N_DAYS_LOOKBACK,
    patch_size=8,
    n_head=4,
    n_layer=2,
    dim=256,
    rope_base=1024,
    max_batch_size=batch_size,
)

def vis(t):
    import matplotlib.pyplot as plt
    plt.plot(t.cpu().numpy())
    plt.show()



print("--- 5. Training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seq_len = 8
pred_len = 8
model = Transformer(targs).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10)
num_epochs = args.e

tot = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        tot += 1
        if tot % 10 == 0:
            vis(batch_X[:100])
            vis(batch_y[:100])

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            output = model(batch_X)
            loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(total_loss / len(dataloader))

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader)}"
    )

print("Training finished.")
