import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
import os

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="1d_p_1d")
parser.add_argument("--head", type=int, default=None, help="Number of rows to load")
parser.add_argument("-b", type=int, default=32768, help="Batch size")
parser.add_argument("-n", type=int, default=2, help="n days lookback")
parser.add_argument("-e", "--epoch", type=int, default=1000, help="Number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument(
    "--cross-section-normalize",
    type=bool,
    default=False,
    help="Normalize whole market VWAP to 1",
)
parser.add_argument("-d", "--dim", type=int, default=256, help="Dimension of model")
parser.add_argument("-l", "--n-layer", type=int, default=2, help="Number of layers")
args = parser.parse_args()


def get_df(head=None, cache=True):
    cache_file = f"../data/cache/a_{args.n}_{args.head}.pq"
    if cache and not head and os.path.exists(cache_file):
        df = pl.read_parquet(cache_file)
    else:
        df = pl.scan_parquet("../data/a_30min.pq")
        df = df.drop(
            ["open", "high", "low", "close", "volume", "total_turnover"]
        ).rename(
            {
                "open_post": "open",
                "high_post": "high",
                "low_post": "low",
                "close_post": "close",
                "volume_post": "volume",
            }
        ).drop_nans().drop_nulls()

        if head is not None:
            df = df.head(head)

        df = df.collect()

        df = df.with_columns(
            avg_price=(
                pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")
            )
            / 4
        ).drop(["open", "high", "low", "close"])

        df = df.with_columns(pl.col("datetime").dt.date().alias("date"))

        if cache and not head:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.write_parquet(cache_file)

    # Split data into train and validation based on year
    train_df = df.filter(pl.col("datetime").dt.year() < 2024)
    val_df = df.filter(pl.col("datetime").dt.year() == 2024)

    print(f"Train data: {len(train_df)} rows")
    print(f"Validation data: {len(val_df)} rows")

    def process_subset(subset_df):
        daily_prices = subset_df.group_by(["order_book_id", "date"]).agg(
            pl.col("avg_price").alias("prices"), pl.col("datetime").alias("datetimes")
        )

        assert daily_prices["prices"].list.len().max() == 8, daily_prices["prices"].list.len().max()
        daily_prices = daily_prices.filter(pl.col("prices").list.len() == 8)

        daily_prices = daily_prices.sort(["order_book_id", "date"])

        price_sequence_expr = [
            pl.col("prices").shift(i).over("order_book_id")
            for i in range(args.n - 1, -1, -1)
        ]

        sequences_df = daily_prices.with_columns(
            pl.concat_list(price_sequence_expr).alias("price_sequence"),
            pl.col("prices").shift(-1).over("order_book_id").alias("next_day_prices"),
        ).drop_nulls()

        assert len(sequences_df) == len(
            sequences_df.filter(pl.col("price_sequence").list.len() == args.n * 8)
        )

        return sequences_df

    train_sequences = process_subset(train_df)
    val_sequences = process_subset(val_df)

    print(f"Created {len(train_sequences)} training samples of {args.n}-day sequences.")
    print(f"Created {len(val_sequences)} validation samples of {args.n}-day sequences.")

    return train_sequences, val_sequences


# Get both train and validation data
train_df, val_df = get_df(head=args.head)

print("--- 3. Custom Dataset and DataLoader ---")


class StockDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.sequences = df["price_sequence"].to_numpy()
        self.targets = df["next_day_prices"].to_numpy()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        price_sequence = self.sequences[idx]
        next_day_prices = self.targets[idx]

        x_np = np.array(price_sequence, dtype=np.float32)
        y_np = np.array(next_day_prices, dtype=np.float32)

        # Normalize based on the last price of the last day in the input sequence
        val = x_np[-1]

        assert val != 0, f"price is zero at index {idx}"

        x_normalized = (x_np / val) - 1
        y_normalized = (y_np / val) - 1

        scale_up = 1 / 0.015  # so variance is close to 1

        return torch.from_numpy(x_normalized * scale_up), torch.from_numpy(
            y_normalized * scale_up
        )


# Create separate datasets for training and validation
train_dataset = StockDataset(train_df)
val_dataset = StockDataset(val_df)


def compute_scaleup(dataset):
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    nxs = np.concatenate(xs, axis=0)
    nys = np.concatenate(ys, axis=0)
    breakpoint()


# Create separate dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=args.b, shuffle=True, num_workers=8
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.b, shuffle=False, num_workers=8
)

from model import Transformer, ModelArgs

targs = ModelArgs(
    block_size=args.n,
    patch_size=8,
    n_head=4,
    n_layer=args.n_layer,
    dim=args.dim,
    rope_base=1024,
    max_batch_size=args.b,
)


def vis(t):
    import matplotlib.pyplot as plt

    assert t.shape[:-1] == (args.b,), t.shape
    x = np.arange(t.shape[1])
    for i in t:
        plt.plot(x, i.cpu().numpy())
    plt.show()


print("--- 5. Training ---")
device = "cuda"
model = Transformer(targs).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.2, patience=10
)
num_epochs = args.e

# Track best validation loss
best_val_loss = float("inf")

try:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1} Training"
        ):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                output = model(batch_X)
                loss = criterion(output, batch_y)
            assert loss < 1e9
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1} Validation"
            ):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                with torch.autocast(device_type=device):
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Step scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

except Exception as e:
    print(f"Error at epoch {epoch + 1}: {e}")
    breakpoint()

# Load best model for final evaluation
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load("best_model.pth"))

breakpoint()
print("Training finished.")
