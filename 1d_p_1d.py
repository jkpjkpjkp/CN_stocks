import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

print("--- 1. Data Loading ---")
df = pl.scan_parquet("../data/a_30min.pq").head(100000).collect()

print("--- 2. Polars Processing ---")
df = df.with_columns(
    ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4).alias(
        "avg_price"
    ),
    pl.col("datetime").dt.date().alias("date"),
)

daily_prices = (
    df.group_by(["order_book_id", "date"])
    .agg(pl.col("avg_price").alias("prices"))
    .sort(["order_book_id", "date"])
)

daily_prices = daily_prices.filter(pl.col("prices").list.len() == 8)

daily_prices = daily_prices.with_columns(
    pl.col("prices").shift(-1).over("order_book_id").alias("next_day_prices")
)

daily_prices = daily_prices.drop_nulls("next_day_prices")

print("--- 3. Custom Dataset and DataLoader ---")


class StockDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.df = df.select(["prices", "next_day_prices"])

    def __len__(self):
        return len(self.df)

    @classmethod
    def _normalize(cls, df: pl.DataFrame):
        val = df["prices"][0][-1]
        ret = df.select(
            prices=(pl.col("prices") / val) - 1,
            next_day_prices=pl.col("next_day_prices") / val - 1,
        )
        return torch.tensor(ret["prices"][0][:-1]), torch.tensor(
            ret["next_day_prices"][0]
        )

    def __getitem__(self, idx):
        return type(self)._normalize(self.df.slice(idx, 1))


batch_size = 3
dataset = StockDataset(daily_prices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


print("--- 4. Model Definition (Simplified Transformer) ---")
seq_len = 7
pred_len = 8


class TransformerModel(nn.Module):
    def __init__(
        self,
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
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_decoder = nn.Parameter(torch.zeros(1, pred_len, d_model))

    def forward(self, src):
        breakpoint()
        src = self.input_proj(src.unsqueeze(-1)) + self.pos_encoder
        output = self.transformer(src)
        return self.output_proj(output)


# --- 5. Training ---
print("--- 5. Training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = TransformerModel().to(device)
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
