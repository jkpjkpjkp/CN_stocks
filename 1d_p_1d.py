import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

print("--- 1. Data Loading ---")
df = pl.scan_parquet("../data/a_30min.pq")

df = df.drop(['open', 'high', 'low', 'close', 'volume']).rename({
    'open_post': 'open',
    'high_post': 'high',
    'low_post': 'low',
    'close_post': 'close',
    'volume_post': 'volume',
})

df = df.head(100000).collect()

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


N_DAYS_LOOKBACK = 2

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

# --- MODIFICATION END ---


print("--- 3. Custom Dataset and DataLoader ---")

class StockDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        # Eagerly convert the needed columns to numpy arrays for faster access in __getitem__
        # Using .to_numpy() on a list column creates an object array of lists
        self.sequences = df["price_sequence"].to_numpy()
        self.targets = df["next_day_prices"].to_numpy()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Retrieve the raw sequence and target lists for the given index
        price_sequence = self.sequences[idx]
        next_day_prices = self.targets[idx]

        # Convert lists to numpy arrays for numerical operations
        # The input X will have shape (14, 8)
        # The target y will have shape (8,)
        x_np = np.array(price_sequence, dtype=np.float32)
        y_np = np.array(next_day_prices, dtype=np.float32)

        # Normalize based on the last price of the last day in the input sequence
        # This helps the model predict relative price changes
        val = x_np[-1]
        
        # Avoid division by zero, though unlikely with price data
        if val == 0:
            val = 1 

        x_normalized = (x_np / val) - 1
        y_normalized = (y_np / val) - 1

        # Convert normalized numpy arrays to PyTorch tensors
        return torch.from_numpy(x_normalized), torch.from_numpy(y_normalized)

# --- MODIFICATION END ---

# Use a larger batch size for more stable gradients
batch_size = 32
dataset = StockDataset(sequences_df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 4. Example Usage ---
print("\n--- 4. Checking Dataloader Output ---")
# Fetch one batch to verify the shapes
try:
    x_batch, y_batch = next(iter(dataloader))
    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    # Expected output for X: torch.Size([32, 14, 8])
    # Expected output for y: torch.Size([32, 8])
except StopIteration:
    print("Dataset is empty. Check the data processing steps.")

batch_size = 3
dataset = StockDataset(daily_prices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


print("--- 4. Model Definition (Simplified Transformer) ---")


class TransformerModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        input_dim=1,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len - 1, d_model))
        self.pos_decoder = nn.Parameter(torch.zeros(1, pred_len, d_model))

    def forward(self, src, tgt):
        b, s = src.shape
        src = self.input_proj(src.unsqueeze(-1)) + self.pos_encoder
        assert src.shape[:2] == (b, s)
        d_model = src.shape[-1]
        tgt = self.input_proj(tgt.unsqueeze(-1)) + self.pos_decoder
        output = self.transformer(src, tgt)
        assert output.shape == (b, s, d_model)
        return self.output_proj(output).squeeze(-1)


# --- 5. Training ---
print("--- 5. Training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
seq_len = 8
pred_len = 8
model = TransformerModel(seq_len, pred_len).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()

        tgt_input = torch.zeros_like(batch_y).to(device)

        output = model(batch_X, tgt_input)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.6f}"
    )

print("Training finished.")
