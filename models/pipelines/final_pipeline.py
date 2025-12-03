"""
1. Multiple encoding strategies:
   - Quantize: percentile
   - Cent: delta in cents
   - Sinusoidal
   - TODO: CNN: Image encoding over K-line graphs (from draw.py)

2. Multiple preprocessing approaches:
   - Raw (ohlc + volume)
   - Per-stock normalized
   - Returns (1min, 30min, 6hr, 1day, 2day)
   - Intra-minute (close divided by open, etc.)
   - Cross-normalized (normalized per timestep)

4. Multiple prediction types and encodings:
   - Quantized predictions (cross-entropy)
   - Cent-based predictions (cross-entropy)
   - Mean + Variance predictions (NLL loss)
   - Quantile predictions (sided Huber loss)

5. Multiple prediction horizons:
   - 1min, 30min, 1day, 2day ahead
   - TODO: Next day's OHLC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import os
import gc
import psutil
import time

from ..prelude.model import dummyLightning, Rope, tm
from ..pola import SortedDS

torch.autograd.set_detect_anomaly(True)


def check_memory_usage(max_gb: float = 320.0, backoff_delay: float = 1.0)\
         -> float:
    """Check memory usage and wait with exponential backoff.

    Args:
        max_gb: Maximum memory in GB before triggering garbage collection
        backoff_delay: Current backoff delay in seconds

    Returns:
        Updated backoff delay (doubled if GC was triggered, reset to 1.0)
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    current_gb = mem_info.rss / (1024 ** 3)  # Convert bytes to GB

    if current_gb > max_gb:
        print(f"Memory usage {current_gb:.2f} GB exceeds limit {max_gb} GB. "
              f"Running GC and backing off for {backoff_delay:.2f}s...")
        gc.collect()
        time.sleep(backoff_delay)
        # Check again after GC
        mem_info = process.memory_info()
        current_gb = mem_info.rss / (1024 ** 3)
        print(f"After GC: {current_gb:.2f} GB")
        return backoff_delay * 2  # Exponential backoff

    return 1.0  # Reset backoff delay


class PriceHistoryDataset(Dataset):
    def __init__(self, config, stock_data: Dict[str, np.ndarray],
                 stock_targets: Dict[str, np.ndarray]):
        """
        Args:
            stock_data: Dict mapping stock_id to full feature array (T, F)
            stock_targets: Dict mapping stock_id to full target array (T,)
              - normalized prices
        """
        self.stock_data = stock_data
        self.targets = stock_targets
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        self.horizons = config.horizons
        self.num_horizons = len(self.horizons)

        # Build index: list of (stock_id, start_idx) tuples
        # Ensure we have enough future data for the longest horizon (480 min)
        self.index = []
        for stock_id, data in stock_data.items():
            max_horizon = max(self.horizons)
            num_windows = len(data) - self.seq_len - max_horizon  # for all horizons
            if num_windows > 0:
                for start_idx in range(num_windows):
                    self.index.append((stock_id, start_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stock_id, start_idx = self.index[idx]
        end_idx = start_idx + self.seq_len

        # Copy data from memory-mapped array to ensure thread-safety
        # and avoid holding references to the mmap
        features = self.stock_data[stock_id][start_idx:end_idx].copy()

        # Get prediction positions' targets (latter half)
        # For each position in the latter half, get targets at all horizons
        target_start = start_idx + self.seq_len // 2

        # targets shape: (pred_len, num_horizons)
        targets = np.zeros((self.pred_len, self.num_horizons),
                           dtype=np.float32)

        # return_targets shape: (pred_len, num_horizons)
        # Return = future_close / current_close
        return_targets = np.zeros((self.pred_len, self.num_horizons),
                                  dtype=np.float32)

        # Vectorize outer loop using slicing
        current_positions = target_start + np.arange(self.pred_len)
        # Copy from memory-mapped array
        current_prices = self.targets[stock_id][current_positions].copy()

        for h_idx, horizon in enumerate(self.horizons):
            future_positions = current_positions + horizon
            # Copy from memory-mapped array
            targets[:, h_idx] = self.targets[stock_id][future_positions]
            future_prices = self.targets[stock_id][future_positions].copy()

            return_targets[:, h_idx] = future_prices / current_prices

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(targets).float(),
            torch.from_numpy(return_targets).float(),
        )


class TransformerBlock(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        attn_dim = self.num_heads * self.head_dim
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, self.hidden_dim, bias=False)
        self.rope = Rope(config)
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(attn_dim)
            self.k_norm = nn.LayerNorm(attn_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, config.interim_dim),
            nn.SiLU(),
            nn.Linear(config.interim_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) or (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        residual = x

        x = self.norm1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if not self.standard_rope:
            q = self.rope(q)
            k = self.rope(k)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.standard_rope:
            q = self.rope(q)
            k = self.rope(k)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)\
                       .transpose(1, 2).contiguous()\
                       .view(batch_size, seq_len, self.attn_dim)

        x = residual + self.o_proj(attn_output)

        x = x + self.ffn(self.norm2(x))

        return x


class QuantizeEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.n_quantize,
                                      config.q_dim)
        self.proj = nn.Linear(config.q_dim
                              * config.num_quantize_features,
                              config.embed_dim)

    def forward(self, x: torch.Tensor, quantiles: torch.Tensor):
        # x: (batch, num_features)
        # quantiles: (num_quantiles, num_features)

        # Use interior quantile boundaries for bucketization
        quantiles_tensor = quantiles.squeeze(-1)[1:-1, :].to(x.device,
                                                             dtype=x.dtype)

        # Bucketize each feature with its corresponding quantiles
        tokens = torch.stack([
            torch.bucketize(x[:, i].contiguous(),
                            quantiles_tensor[:, i].contiguous(), right=True)
            for i in range(x.shape[1])
        ], dim=1)  # (batch, num_features)

        # Clamp to valid embedding indices [0, n_quantize-1]
        tokens = tokens.clamp(0, self.n_quantize - 1)

        # Embed each feature: (batch, num_features, embed_dim)
        embedded = self.embedding(tokens)

        # Flatten and project: (batch, num_features * embed_dim) -> (batch, embed_dim)
        batch_size = embedded.shape[0]
        flattened = embedded.reshape(batch_size, -1)
        return self.proj(flattened)


class CentsEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.max_cent_abs = config.max_cent_abs
        self.embedding = nn.Embedding(2 * config.max_cent_abs + 3,
                                      config.embed_dim)
        self.proj = nn.Linear(config.embed_dim * config.num_cent_feats,
                              config.embed_dim)

    def forward(self, x):
        """Convert cent differences to tokens

        Args:
            x: (batch, cent_feats) - delta price features (delta_1min, delta_30min)

        Returns:
            (batch, embed_dim) - projected embeddings
        """
        # Convert to cents and clamp
        # x: (batch, cent_feats)
        x = (x * 100).long().clamp(-self.max_cent_abs-1, self.max_cent_abs+1)
        x = x + self.max_cent_abs + 1

        # Embed each feature: (batch, cent_feats, embed_dim)
        embedded = self.embedding(x)

        # Flatten and project: (batch, cent_feats * embed_dim) -> (batch, embed_dim)
        batch_size = embedded.shape[0]
        flattened = embedded.reshape(batch_size, -1)
        return self.proj(flattened)


class SinEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)

        freqs = torch.exp(
            torch.linspace(0, np.log(config.max_freq),
                           config.sin_dim // 2)
        )
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        self.register_buffer('freqs', freqs)

        # Project flattened sin/cos features to embed_dim
        # Input will be (batch, features * embed_dim) after flattening
        # For 12 features: 12 * embed_dim
        self.proj = nn.Linear(config.num_features * config.sin_dim,
                              config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, features)"""
        # x: (batch, features), freqs: (1, 1, embed_dim // 2)
        # Add dimension for broadcasting: (batch, features, 1) * (1, 1, embed_dim // 2) -> (batch, features, embed_dim // 2)
        sin_emb = torch.sin(x.unsqueeze(-1) * self.freqs)
        cos_emb = torch.cos(x.unsqueeze(-1) * self.freqs)
        # Stack: (batch, features, embed_dim // 2, 2), then flatten to (batch, features * embed_dim)
        stacked = torch.stack((sin_emb, cos_emb), dim=-1)
        flattened = stacked.flatten(1)
        # Project to embed_dim: (batch, features * embed_dim) -> (batch, embed_dim)
        return self.proj(flattened)


class MultiEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.quantize_encoder = QuantizeEncoder(config)
        self.cent_encoder = CentsEncoder(config)
        self.sin_encoder = SinEncoder(config)

        self.combiner = nn.Linear(3 * config.embed_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features)"""
        batch_size, seq_len, feat_dim = x.shape

        x_flat = x.view(-1, feat_dim)
        # Quantize encoder uses all features with per-feature quantiles
        # Cent encoder uses delta features (delta_1min, delta_30min)
        cent_start = self.cent_feats_start
        cent_end = cent_start + self.num_cent_feats
        embeddings = [
            self.quantize_encoder(x_flat, self.quantiles),
            self.cent_encoder(x_flat[:, cent_start:cent_end]),
            self.sin_encoder(x_flat),
        ]
        output = self.combiner(torch.cat(embeddings, dim=-1))

        return output.unflatten(0, (batch_size, seq_len))


class MultiReadout(dummyLightning):
    def __init__(self, config):
        super().__init__(config)

        self.quantized_head = nn.Linear(config.hidden_dim,
                                        config.n_quantize * self.num_horizons)
        self.cent_head = nn.Linear(config.hidden_dim,
                                   config.num_cents * self.num_horizons)
        self.mean_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.variance_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.quantile_head = nn.Linear(config.hidden_dim,
                                       config.num_quantiles*self.num_horizons)

        # Return prediction heads (future_close / current_close)
        self.return_mean_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.return_variance_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.return_quantile_head = nn.Linear(config.hidden_dim,
                                              config.num_quantiles*self.num_horizons)

    def forward(self, x: torch.Tensor, target_type='mean'):
        """
        x: (batch, seq_len, hidden_dim)
        returns: (batch, seq_len, num_horizons, ...)
                 predictions for each horizon
        """
        batch_size, seq_len, _ = x.shape
        x = x.float()

        if target_type == 'quantized':
            out = self.quantized_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.n_quantize)
        elif target_type == 'cent':
            out = self.cent_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_cents)
        elif target_type == 'mean':
            return self.mean_head(x)
        elif target_type == 'var':
            return F.softplus(self.variance_head(x))
        elif target_type == 'quantile':
            out = self.quantile_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_quantiles)
        # Return predictions
        elif target_type == 'return_mean':
            return self.return_mean_head(x)
        elif target_type == 'return_var':
            return F.softplus(self.return_variance_head(x))
        elif target_type == 'return_quantile':
            out = self.return_quantile_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_quantiles)


class FinalPipeline(dummyLightning):
    """Complete pipeline combining all components"""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = MultiEncoder(config)
        self.backbone = tm(config)
        self.readout = MultiReadout(config)

        # Initialize DDP attributes
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self._ddp_model = None

    def setup_ddp(self, rank: Optional[int] = None,
                  world_size: Optional[int] = None):
        """Initialize DDP from environment or provided rank/world_size"""
        if not self.config.use_ddp:
            return

        # Get rank and world_size from environment if not provided
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))

        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get('LOCAL_RANK', rank))

        torch.cuda.set_device(self.local_rank)
        self.device = f'cuda:{self.local_rank}'

        # Initialize process group
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            dist.init_process_group(
                backend=self.config.ddp_backend,
                rank=self.rank,
                world_size=self.world_size
            )

        # Move model to device before wrapping with DDP
        self.to(self.device)

        # Wrap model with DDP
        self._ddp_model = DDP(
            self,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
        )

    def get_model(self):
        """Return the underlying model (unwrapped from DDP if necessary)"""
        return self._ddp_model.module if self._ddp_model is not None else self

    def is_root(self):
        return self.rank == 0

    def all_reduce(self, tensor: torch.Tensor, op=None) -> torch.Tensor:
        """All-reduce operation across all processes"""
        if not self.use_ddp:
            return tensor

        if op is None:
            op = dist.ReduceOp.SUM

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes"""
        if not self.use_ddp:
            return tensor

        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def reduce_dict(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reduce a dictionary of metrics across all processes"""
        if not self.use_ddp:
            return metrics

        reduced = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                reduced[key] = self.all_reduce(value) / self.world_size
            else:
                reduced[key] = value

        return reduced

    def _get_cache_path(self) -> Path:
        """Get the path to the cache file"""
        return Path(self.config.cache_dir) / 'dataset_cache.npz'

    def _get_mmap_dir(self) -> Path:
        """Get the directory for memory-mapped arrays"""
        mmap_dir = Path(self.config.cache_dir) / 'mmap_arrays'
        mmap_dir.mkdir(parents=True, exist_ok=True)
        return mmap_dir

    def _save_cache(self, train_x: Dict[str, np.ndarray],
                    train_y: Dict[str, np.ndarray],
                    val_x: Dict[str, np.ndarray],
                    val_y: Dict[str, np.ndarray],
                    quantiles: torch.Tensor):
        """Save dataset metadata to disk cache.

        Note: Individual stock arrays are already saved as memory-mapped files by _bs.
        This method only saves quantiles and metadata.
        """
        cache_path = self._get_cache_path()

        if self.is_root():
            print(f"Saving cache metadata to {cache_path}...")

            save_dict = {}

            # Save quantiles as single tensor (n_quantize+1, num_features)
            save_dict['quantiles'] = quantiles.cpu().numpy()

            # Save stock IDs as metadata (arrays are already on disk)
            train_ids = list(train_x.keys())
            val_ids = list(val_x.keys())
            save_dict['_train_ids'] = np.array(train_ids, dtype=object)
            save_dict['_val_ids'] = np.array(val_ids, dtype=object)

            np.savez_compressed(cache_path, **save_dict)
            print(f"Cache metadata saved successfully!")
            print(f"Memory-mapped arrays are in {self._get_mmap_dir()}")

    def _load_cache(self) -> Optional[Tuple[Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             torch.Tensor]]:
        """Load dataset dictionaries from disk cache using memory-mapped arrays"""
        cache_path = self._get_cache_path()
        mmap_dir = self._get_mmap_dir()

        if not cache_path.exists():
            return None

        # Check if mmap directory exists and has files
        if not mmap_dir.exists() or not any(mmap_dir.iterdir()):
            if self.is_root():
                print(f"Cache metadata found but mmap directory is missing or empty. Recomputing...")
            return None

        if self.is_root():
            print(f"Loading cache from {cache_path} (memory-mapped mode)...")

        # Load metadata
        data = np.load(cache_path, allow_pickle=True)

        # Reconstruct dictionaries
        train_x = {}
        train_y = {}
        val_x = {}
        val_y = {}

        # Load metadata (these are small, so load into memory)
        train_ids = data['_train_ids'].copy()
        val_ids = data['_val_ids'].copy()

        # Load quantiles as single tensor (n_quantize+1, num_features)
        quantiles = torch.from_numpy(data['quantiles'].copy()).float()

        # Load memory-mapped arrays for train data
        for stock_id in train_ids:
            train_features_path = mmap_dir / f"train_data_{stock_id}.npy"
            train_targets_path = mmap_dir / f"train_target_{stock_id}.npy"

            if train_features_path.exists() and train_targets_path.exists():
                train_x[stock_id] = np.load(train_features_path, mmap_mode='r')
                train_y[stock_id] = np.load(train_targets_path, mmap_mode='r')
            else:
                if self.is_root():
                    print(f"Warning: Missing memory-mapped files for train stock {stock_id}")

        # Load memory-mapped arrays for val data
        for stock_id in val_ids:
            val_features_path = mmap_dir / f"val_data_{stock_id}.npy"
            val_targets_path = mmap_dir / f"val_target_{stock_id}.npy"

            if val_features_path.exists() and val_targets_path.exists():
                val_x[stock_id] = np.load(val_features_path, mmap_mode='r')
                val_y[stock_id] = np.load(val_targets_path, mmap_mode='r')
            else:
                if self.is_root():
                    print(f"Warning: Missing memory-mapped files for val stock {stock_id}")

        if self.is_root():
            print(f"Cache loaded successfully (memory-mapped)! "
                  f"{len(train_ids)} train stocks, {len(val_ids)} val stocks")

        return train_x, train_y, val_x, val_y, quantiles

    def prepare_data(self, path: Optional[str] = None, seq_len: int = 64,
                     n_quantize: int = 256, train_frac: float = 0.9):
        if not self.config.debug_data:
            cached = self._load_cache()
            if cached is not None:
                train_x, train_y, val_x, val_y, quantiles = cached
                self.quantiles = quantiles
                self.encoder.quantiles = quantiles

                self.train_dataset = PriceHistoryDataset(
                    self.config, train_x, train_y,
                )
                self.val_dataset = PriceHistoryDataset(
                    self.config, val_x, val_y,
                )
                return

        # Cache miss or force recalculation - compute from scratch
        if self.is_root():
            print("Computing data from scratch...")

        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        path = Path(path)

        train_x, train_y, val_x, val_y = \
            self._prepare_data_chunked(path, seq_len, n_quantize, train_frac)

        self._save_cache(
            train_x, train_y,
            val_x, val_y,
            self.quantiles
        )

        self.train_dataset = PriceHistoryDataset(
            self.config, train_x, train_y,
        )
        self.val_dataset = PriceHistoryDataset(
            self.config, val_x, val_y,
        )
        return

    def _prepare_data_chunked(self, path: Path, seq_len: int, n_quantize: int,
                               train_frac: float, chunk_size: int = 300):
        """Prepare data in chunks using SortedDS for efficient iteration.

        Args:
            path: Path to parquet file
            seq_len: Sequence length
            n_quantize: Number of quantile bins
            train_frac: Fraction of data for training
            chunk_size: Number of stocks to process per chunk (default 300)

        Returns:
            train_x, train_y, val_x,
            val_y
        """
        if self.is_root():
            print(f"Processing data in chunks of {chunk_size} stocks...")

        # Step 1: Get time-based cutoff for train/val split
        if self.is_root():
            print("Computing train/val cutoff...")
        
        if self.is_root():
            print(f"Train/val cutoff timestamp: {cutoff}")

        # Step 2: Build full pipeline as lazy frame, sorted by id for SortedDS
        df = pl.scan_parquet(str(path))
        df = self._compute_features(df)

        cutoff = (
            df
            .select(pl.col('datetime')
                    .sample(1000000)
                    .unique()
                    .cast(pl.Int64)
                    .quantile(0.9))
            .collect()['datetime'][0])

        df = df.with_columns(
            is_train=(pl.col('datetime').cast(pl.Int64) <= cutoff)
        )

        # Compute per-stock normalization stats from training data
        stats = df.filter(pl.col('is_train')).group_by('id').agg([
            pl.col('close').mean().alias('mean_close'),
            pl.col('close').std().alias('std_close'),
            pl.col('volume').mean().alias('mean_volume'),
            pl.col('volume').std().alias('std_volume')
        ])
        df = df.join(stats, on='id', how='left')
        df = df.with_columns(
            ((pl.col('close') - pl.col('mean_close'))
             / (pl.col('std_close') + 1e-8)).alias('close_norm'),
            ((pl.col('volume') - pl.col('mean_volume'))
             / (pl.col('std_volume') + 1e-8)).alias('volume_norm'),
        )

        # Step 3: Compute quantiles from sample
        if self.is_root():
            print("Computing quantiles...")
        self.quantiles = self._compute_quantiles(df, n_quantize)
        self.encoder.quantiles = self.quantiles
        gc.collect()

        # Step 4: Create SortedDS for efficient chunking
        if self.is_root():
            print("Building SortedDS breakpoints...")
        sorted_ds = SortedDS(df, id='id')
        total_stocks = sorted_ds.bp.shape[0]

        if self.is_root():
            print(f"Total stocks: {total_stocks}")

        # Step 5: Process stocks in chunks using breakpoints
        train_x = {}
        train_y = {}
        val_x = {}
        val_y = {}
        mmap_dir = self._get_mmap_dir()

        for chunk_idx, chunk_df in enumerate(sorted_ds.chunks(chunk_size)):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_stocks)

            if self.is_root():
                print(f"\nChunk {chunk_idx + 1} "
                      f"(stocks {chunk_start}-{chunk_end})")

            chunk_df = chunk_df.collect()
            chunk_stock_ids = chunk_df['id'].unique().to_list()

            self._process_stock_chunk(
                chunk_df, chunk_stock_ids, seq_len, mmap_dir,
                train_x, train_y,
                val_x, val_y
            )

            del chunk_df
            gc.collect()

            # Check memory after each chunk
            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024 ** 3)
            if self.is_root():
                print(f"  Memory: {mem_gb:.2f} GB, "
                      f"train: {len(train_x)}, "
                      f"val: {len(val_x)}")

        if self.is_root():
            print(f"\nCompleted {total_stocks} stocks")
            print(f"Final: {len(train_x)} train stocks, "
                  f"{len(val_x)} val stocks")

        return (train_x, train_y,
                val_x, val_y)

    def _process_stock_chunk(self, chunk_df: pl.DataFrame,
                              stock_ids: list, seq_len: int, mmap_dir: Path,
                              train_x: dict, train_y: dict,
                              val_x: dict, val_y: dict):
        """Process a chunk of stocks and save to mmap files."""
        for stock_id in stock_ids:
            stock_df = chunk_df.filter(pl.col('id') == stock_id)

            if len(stock_df) <= seq_len + 1:
                continue

            features = (stock_df.select(self.features)
                        .to_numpy()
                        .astype(np.float32))
            is_train = stock_df['is_train'].to_numpy()
            close = stock_df['close_norm'].to_numpy().astype(np.float32)

            train_mask = is_train.astype(bool)
            val_mask = ~train_mask

            # Save training data
            if train_mask.sum() > seq_len + 1:
                train_features_path = mmap_dir / f"train_data_{stock_id}.npy"
                train_targets_path = mmap_dir / f"train_target_{stock_id}.npy"

                np.save(train_features_path, features[train_mask])
                np.save(train_targets_path, close[train_mask])

                train_x[stock_id] = np.load(
                    train_features_path, mmap_mode='r')
                train_y[stock_id] = np.load(
                    train_targets_path, mmap_mode='r')

            # Save validation data
            if val_mask.sum() > seq_len + 1:
                val_features_path = mmap_dir / f"val_data_{stock_id}.npy"
                val_targets_path = mmap_dir / f"val_target_{stock_id}.npy"

                np.save(val_features_path, features[val_mask])
                np.save(val_targets_path, close[val_mask])

                val_x[stock_id] = np.load(
                    val_features_path, mmap_mode='r')
                val_y[stock_id] = np.load(
                    val_targets_path, mmap_mode='r')

    def get_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = True):
        """Create dataloader with DDP support via DistributedSampler"""
        from torch.utils.data import DataLoader

        if self.use_ddp:
            # Use DistributedSampler for DDP
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last
            )
            # When using DistributedSampler, shuffle must be False in DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=drop_last
            )
        else:
            # Standard DataLoader for non-distributed training
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=drop_last
            )

        return dataloader

    @property
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def _compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all feature types"""

        # Returns at different time scales (1min, 30min, 1day, 2days)
        df = df.with_columns(
            delta_1min=pl.col('close').shift(-1).over('id') - pl.col('close'),
            delta_30min=pl.col('close').shift(-30).over('id') - pl.col('close'),
            ret_1min=(pl.col('close').shift(-1).over('id') / pl.col('close') - 1),
            ret_30min=(pl.col('close').shift(-30).over('id') / pl.col('close') - 1),
            ret_1day=(pl.col('close').shift(-240).over('id') / pl.col('close') - 1),
            ret_2day=(pl.col('close').shift(-480).over('id') / pl.col('close') - 1),  # 2 days
        )

        # Intra-minute relative features
        df = df.with_columns([
            (pl.col('close') / pl.col('open')).alias('close_open'),
            (pl.col('high') / pl.col('open')).alias('high_open'),
            (pl.col('low') / pl.col('open')).alias('low_open'),
            (pl.col('high') / pl.col('low')).alias('high_low')
        ])

        # Fill NaN values
        df = df.with_columns([
            pl.col('delta_1min').fill_null(0.),
            pl.col('delta_30min').fill_null(0.),
            pl.col('ret_1min').fill_null(0.),
            pl.col('ret_30min').fill_null(0.),
            pl.col('ret_1day').fill_null(0.),
            pl.col('ret_2day').fill_null(0.),
            pl.col('close_open').fill_null(1.),
            pl.col('high_open').fill_null(1.),
            pl.col('low_open').fill_null(1.),
            pl.col('high_low').fill_null(1.)
        ])

        return df

    def _split_data(self, df: pl.DataFrame, train_frac: float) -> pl.DataFrame:
        cutoff = df.select(
            pl.col('datetime').sample(1000000).unique().cast(pl.Int64).quantile(0.9)
        ).collect()['datetime'][0]

        df = df.with_columns(
            is_train=(pl.col('datetime').cast(pl.Int64) <= cutoff)
        )
        if self.debug_data:
            assert len(df.collect().filter('is_train'))
            assert len(df.collect().filter(~pl.col('is_train')))

        stats = df.filter(pl.col('is_train')).group_by('id').agg([
            pl.col('close').mean().alias('mean_close'),
            pl.col('close').std().alias('std_close'),
            pl.col('volume').mean().alias('mean_volume'),
            pl.col('volume').std().alias('std_volume')
        ])

        df = df.join(stats, on='id', how='left')

        # normalize
        df = df.with_columns(
            ((pl.col('close') - pl.col('mean_close')) / (pl.col('std_close') + 1e-8)).alias('close_norm'),
            ((pl.col('volume') - pl.col('mean_volume')) / (pl.col('std_volume') + 1e-8)).alias('volume_norm'),
        )

        return df

    def _compute_quantiles(self, df: pl.DataFrame, n_quantize: int)\
            -> torch.Tensor:
        """Compute quantiles for all features using sorting and indexing

        Returns:
            torch.Tensor: (n_quantize+1, num_features) quantile boundaries
        """
        train_df = df.filter(pl.col('is_train'))
        q_positions = [i / n_quantize for i in range(n_quantize + 1)]

        # Sample rows once, then select all columns from those rows
        sampled = (train_df
                   .select(self.features)
                   .collect()
                   .sample(n=1000000))

        # Build quantiles by sorting and indexing
        quantiles_list = []
        for col in self.features:
            # Get the column data and sort it
            col_data = sampled[col].sort()
            n = len(col_data)

            # Compute index positions for each quantile
            col_quantiles = []
            for q in q_positions:
                # Linear interpolation index: q * (n - 1)
                idx = q * (n - 1)
                idx_lower = int(np.floor(idx))
                idx_upper = int(np.ceil(idx))

                if idx_lower == idx_upper:
                    # Exact index
                    quantile_val = col_data[idx_lower]
                else:
                    # Linear interpolation between adjacent values
                    weight = idx - idx_lower
                    quantile_val = (1 - weight) * col_data[idx_lower] + weight * col_data[idx_upper]

                col_quantiles.append(quantile_val)

            quantiles_list.append(torch.tensor(col_quantiles, dtype=torch.float32))

        del sampled
        gc.collect()

        # Stack to (n_quantize+1, num_features)
        quantiles = torch.stack(quantiles_list, dim=1)
        return quantiles

    def _detect_trading_gaps(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect trading gaps and mark with special tokens:
        - lunch: short intraday gaps (~1 hour)
        - dinner: overnight gaps between trading days (~15-18 hours)
        - skipped_day: weekend/holiday gaps (24+ hours, can have multiple)
        """
        # Calculate time gaps in seconds between consecutive rows per stock
        df = df.with_columns([
            pl.col('datetime').diff().over('id').dt.total_seconds().alias('time_gap_seconds')
        ])

        return df

    def _bs(self, df, seq_len) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray],
                                         Dict[str, np.ndarray]]:
        """
        Build sequences and return the data dictionaries with memory-mapped arrays.

        Returns:
            train_x, train_y,
            val_x, val_y
        """

        time_stats = df.group_by('datetime').agg(
            time_mean_close=pl.col('close').mean(),
            time_std_close=pl.col('close').std(),
        )

        df = df.join(time_stats, on='datetime')
        df = df.with_columns(
            close_time_norm=((pl.col('close') - pl.col('time_mean_close'))
                             / (pl.col('time_std_close') + 1e-8))
        )

        train_x = {}
        train_y = {}

        val_x = {}
        val_y = {}

        # Get memory-mapped directory
        mmap_dir = self._get_mmap_dir()

        # Initialize backoff delay
        backoff_delay = 1.0

        # TODO: align for cross attention
        stock_ids = df.select('id').unique().collect()['id']
        total_stocks = len(stock_ids)

        for idx, (stock_df, stockid) in enumerate(df.chunks(1, with_name=True)):
            if self.is_root() and idx % 100 == 0:
                print(f"Processing stock {idx}/{total_stocks}: {stock_id}")

            stock_df = stock_df.collect()
            if len(stock_df) <= seq_len + 1:
                continue

            # Check memory before conversion and apply backoff if needed
            backoff_delay = check_memory_usage(max_gb=300.0, backoff_delay=backoff_delay)

            # Convert to numpy
            features = stock_df.select(self.features).to_numpy().astype(np.float32)
            is_train = stock_df['is_train'].to_numpy()
            close = stock_df['close_norm'].to_numpy().astype(np.float32)

            # Split each stock's data by time - train data and val data
            train_mask = is_train.astype(bool)
            val_mask = ~train_mask

            # Only add to train set if we have enough train data
            if train_mask.sum() > seq_len + 1:
                # Create memory-mapped files for this stock's training data
                train_features_path = mmap_dir / f"train_data_{stock_id}.npy"
                train_targets_path = mmap_dir / f"train_target_{stock_id}.npy"

                # Save to disk as memory-mapped array
                train_features = features[train_mask]
                train_targets = close[train_mask]

                # Write to disk
                np.save(train_features_path, train_features)
                np.save(train_targets_path, train_targets)

                # Load as memory-mapped (read-only for efficiency)
                train_x[stock_id] = np.load(train_features_path, mmap_mode='r')
                train_y[stock_id] = np.load(train_targets_path, mmap_mode='r')

                # Immediately flush to disk (though np.save already does this)
                del train_features, train_targets

            # Only add to val set if we have enough val data
            if val_mask.sum() > seq_len + 1:
                # Create memory-mapped files for this stock's validation data
                val_features_path = mmap_dir / f"val_data_{stock_id}.npy"
                val_targets_path = mmap_dir / f"val_target_{stock_id}.npy"

                # Save to disk as memory-mapped array
                val_features = features[val_mask]
                val_targets = close[val_mask]

                # Write to disk
                np.save(val_features_path, val_features)
                np.save(val_targets_path, val_targets)

                # Load as memory-mapped (read-only for efficiency)
                val_x[stock_id] = np.load(val_features_path, mmap_mode='r')
                val_y[stock_id] = np.load(val_targets_path, mmap_mode='r')

                # Immediately flush to disk
                del val_features, val_targets

            # Clean up intermediate arrays
            del stock_df, features, is_train, close

            # Periodic garbage collection to prevent memory buildup
            if idx % 100 == 0:
                gc.collect()

        if self.is_root():
            print(f"Completed processing {total_stocks} stocks with memory-mapped arrays")

        return train_x, train_y, val_x, val_y

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        features = self.backbone(encoded)
        return features

    def step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        x, y, y_returns = batch
        # x: (batch, seq_len, features)
        # y: (batch, pred_len, num_horizons) - normalized prices where pred_len = seq_len // 2
        # y_returns: (batch, pred_len, num_horizons) - return multipliers (future_close / current_close)

        seq_len = x.shape[1]
        losses = {}
        
        features = self.forward(x)[:, seq_len//2:, :]
        # Get predictions for all sequence positions
        # But we only care about predictions from positions seq_len//2 onwards
        pred_quantized = self.readout(features, 'quantized')  # (batch, pred_len, num_horizons, n_quantize)
        y = y.to(pred_quantized.dtype)
        y_returns_typed = y_returns.to(pred_quantized.dtype)

        # Quantized prediction loss - predicting returns
        # Use close feature quantiles (index 0)
        quantiles_tensor = self.quantiles.squeeze(-1)[:, 0].to(y_returns_typed.device, dtype=y_returns_typed.dtype)
        target_quantized = torch.bucketize(y_returns_typed, quantiles_tensor)  # (batch, pred_len, num_horizons)
        target_quantized = torch.clamp(target_quantized, 0, self.config.n_quantize - 1)

        losses['quantized'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = F.cross_entropy(
                pred_quantized[:, :, h_idx, :].reshape(-1, self.config.n_quantize),
                target_quantized[:, :, h_idx].reshape(-1)
            )
            losses[f'quantized_{h_name}'] = h_loss
            losses['quantized'] += h_loss / len(self.horizons)

        # Mean + Variance prediction loss (NLL)
        pred_mean = self.readout(features, 'mean')  # (batch, pred_len, num_horizons)
        pred_var = self.readout(features, 'var')  # (batch, pred_len, num_horizons)
        pred_var += 1e-8  # (batch, pred_len, num_horizons)

        losses['nll'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            # Full Gaussian NLL: 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y-μ)²/σ²
            nll = 1/2 * (torch.log(2 * torch.pi * pred_var[:, :, h_idx]) +
                            (y[:, :, h_idx] - pred_mean[:, :, h_idx]) ** 2 / pred_var[:, :, h_idx])
            h_loss = nll.mean()
            losses[f'nll_{h_name}'] = h_loss
            losses['nll'] += h_loss / len(self.horizons)

        # Quantile prediction loss
        pred_quantiles = self.readout(features, 'quantile')

        losses['quantile'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = self._quantile_loss(pred_quantiles[:, :, h_idx, :], y[:, :, h_idx])
            losses[f'quantile_{h_name}'] = h_loss
            losses['quantile'] += h_loss / len(self.horizons)

        # ===== Return prediction losses =====
        y_returns = y_returns.to(pred_mean.dtype)

        # Return variance prediction loss (NLL)
        pred_return_mean = self.readout(features, 'return_mean')  # (batch, pred_len, num_horizons)
        pred_return_var = self.readout(features, 'return_var')  # (batch, pred_len, num_horizons)
        pred_return_var += 1e-8

        losses['return_nll'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            nll = 1/2 * (torch.log(2 * torch.pi * pred_return_var[:, :, h_idx]) +
                         (y_returns[:, :, h_idx] - pred_return_mean[:, :, h_idx]) ** 2 / pred_return_var[:, :, h_idx])
            h_loss = nll.mean()
            losses[f'return_nll_{h_name}'] = h_loss
            losses['return_nll'] += h_loss / len(self.horizons)

        # Return quantile prediction loss
        pred_return_quantiles = self.readout(features, 'return_quantile')

        losses['return_quantile'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = self._quantile_loss(pred_return_quantiles[:, :, h_idx, :], y_returns[:, :, h_idx])
            losses[f'return_quantile_{h_name}'] = h_loss
            losses['return_quantile'] += h_loss / len(self.horizons)
        
        assert (
            losses['quantized'] >= 0 and
            losses['quantile'] >= 0 and
            losses['return_quantile'] >= 0
        ), losses
        # Combine losses
        total_loss = (
            0.15 * losses['quantized'] +
            0.15 * losses['nll'] +
            0.15 * losses['quantile'] +
            0.15 * losses['return_nll'] +
            0.10 * losses['return_quantile']
        )

        # Log aggregate losses with hierarchical grouping
        # Price prediction losses
        self.log('price_prediction/quantized', losses['quantized'])
        self.log('price_prediction/nll', losses['nll'])
        self.log('price_prediction/quantile', losses['quantile'])

        # Return prediction losses
        self.log('return_prediction/nll', losses['return_nll'])
        self.log('return_prediction/quantile', losses['return_quantile'])

        for h_name in self.horizons:
            # Price prediction per horizon
            self.log(f'price_prediction/horizons/{h_name}/quantized', losses[f'quantized_{h_name}'])
            self.log(f'price_prediction/horizons/{h_name}/nll', losses[f'nll_{h_name}'])
            self.log(f'price_prediction/horizons/{h_name}/quantile', losses[f'quantile_{h_name}'])

            # Return prediction per horizon
            self.log(f'return_prediction/horizons/{h_name}/nll', losses[f'return_nll_{h_name}'])
            self.log(f'return_prediction/horizons/{h_name}/quantile', losses[f'return_quantile_{h_name}'])

        result = {
            'loss': total_loss,
            'quantized_loss': losses['quantized'],
            'nll_loss': losses['nll'],
            'quantile_loss': losses['quantile'],
            'return_nll_loss': losses['return_nll'],
            'return_quantile_loss': losses['return_quantile']
        }

        for h_name in self.horizons:
            result[f'quantized_{h_name}'] = losses[f'quantized_{h_name}']
            result[f'nll_{h_name}'] = losses[f'nll_{h_name}']
            result[f'quantile_{h_name}'] = losses[f'quantile_{h_name}']
            result[f'return_nll_{h_name}'] = losses[f'return_nll_{h_name}']
            result[f'return_quantile_{h_name}'] = losses[f'return_quantile_{h_name}']

        return result

    def _quantile_loss(self, pred, target):
        """ pred: (b, l, num_horizons, num_quantiles)
          target: (b, l, num_horizons)
        """
        quantiles = torch.tensor(self.quantiles, device=pred.device)
        quantiles = quantiles.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        quantiles = quantiles.expand(pred.shape)
        target = target.unsqueeze(-1)

        weight = torch.where(
            pred > target,
            quantiles,
            1 - quantiles
        )
        delta = torch.abs(pred - target)
        huber = torch.where(
            delta <= 1,
            0.5 * delta ** 2,
            delta - 0.5
        )

        return (huber * weight).mean()


@dataclass
class FinalPipelineConfig:
    # Model
    embed_dim: int = 256
    hidden_dim: int = 256
    expand: float = 2.
    num_layers: int = 6
    num_heads: int = 8
    attn: float = 1.
    max_freq: float = 10000.
    standard_rope: bool = False  # use fine-grained rope
    qk_norm: bool = False

    # Training
    vram_gb: int = 16
    batch_size: int | None = None
    lr: float = 3e-4
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Data
    seq_len: int = 4096
    n_quantize: int = 128
    max_cent_abs: int = 64
    cache_dir: str | None = None

    features: tuple = ('close', 'close_norm', 'delta_1min', 'delta_30min',
                       'ret_1min', 'ret_30min', 'ret_1day',
                       'ret_2day', 'close_open', 'high_open',
                       'low_open', 'high_low', 'volume', 'volume_norm')
    cent_feats: tuple = ('delta_1min', 'delta_30min')
    horizons: tuple = (1, 30, 240, 480)
    quantiles: tuple = (10, 25, 50, 75, 90)

    debug_data: int | None = None
    debug_model: bool = False

    # DDP settings
    use_ddp: bool = False
    ddp_backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: str = '12355'

    debug_ddp: bool = False
    num_workers: int | None = None
    shrink_model: bool = False
    device: str = 'cuda'

    def __post_init__(self):
        # Handle shrink_model before other calculations
        if self.shrink_model:
            self.hidden_dim //= 4
            self.num_heads //= 2
            self.num_layers //= 2
            self.embed_dim //= 2

        # Model
        self.interim_dim = int(self.hidden_dim * self.expand)
        self.head_dim = int(self.hidden_dim * self.attn / self.num_heads)

        # Embedding
        self.q_dim = self.sin_dim = self.embed_dim // 2
        self.num_cents = self.max_cent_abs * 2 + 1

        # Features
        self.num_features = len(self.features)
        self.num_quantize_features = self.num_features
        self.num_cent_feats = len(self.cent_feats)  # number of cent features
        # Find the starting index of cent features in the features tuple
        self.cent_feats_start = self.features.index(self.cent_feats[0]) if self.cent_feats else 0
        self.num_horizons = len(self.horizons)
        self.num_quantiles = len(self.quantiles)
        self.pred_len = self.seq_len // 2

        # Dataloader
        self.batch_size = self.vram_gb * 2 ** 20 // self.seq_len // self.num_layers // self.interim_dim
        self.num_workers = self.num_workers or os.cpu_count() // 2
        self.cache_dir = self.cache_dir or str(Path.home() / 'h' / 'cache' / 'pipeline_data' / f'debug_{self.debug_data}')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


config = FinalPipelineConfig(
    shrink_model=True,
    debug_data=10,
    debug_model=True,
    use_ddp=False,
)


def parse_args_to_config(base_config: FinalPipelineConfig) -> FinalPipelineConfig:
    """GENERATED. Parse CLI arguments and override config fields"""
    import argparse
    import sys
    from dataclasses import fields

    parser = argparse.ArgumentParser(description='Train FinalPipeline')

    for field in fields(FinalPipelineConfig):
        field_name = field.name
        tp = field.type
        default_val = getattr(base_config, field_name)

        if tp == bool or tp == 'bool':
            parser.add_argument(f'--{field_name}', action='store_true',
                              help=f'Set {field_name}=True')
            parser.add_argument(f'--no_{field_name}', dest=field_name,
                              action='store_false', help=f'Set {field_name}=False')
            parser.set_defaults(**{field_name: default_val})
        elif 'bool | int' in str(tp) or 'int | bool' in str(tp):
            # Special handling for bool | int union type
            parser.add_argument(f'--{field_name}', type=str, default=str(default_val),
                              help=f'{field_name} (bool or int)')
        elif tp == int or tp == 'int' or 'int | None' in str(tp):
            parser.add_argument(f'--{field_name}', type=int, default=default_val,
                              help=f'{field_name} (default: {default_val})')
        elif tp == float or tp == 'float':
            parser.add_argument(f'--{field_name}', type=float, default=default_val,
                              help=f'{field_name} (default: {default_val})')
        elif tp == str or tp == 'str' or 'str | None' in str(tp) or 'Optional[str]' in str(tp):
            parser.add_argument(f'--{field_name}', type=str, default=default_val,
                              help=f'{field_name} (default: {default_val})')
        elif 'tuple' in str(tp):
            # Skip tuple types - they are not meant to be overridden from CLI
            parser.set_defaults(**{field_name: default_val})
        else:
            # Fallback for any other type
            parser.set_defaults(**{field_name: default_val})

    # Filter out module-like arguments (e.g., "models.pipelines.final_pipeline")
    # These come from wrappers like utils.oom_debug_hook
    filtered_argv = [
        arg for arg in sys.argv[1:]
        if not arg.count('.') >= 2
    ]

    args = parser.parse_args(filtered_argv)

    # Override config with parsed args
    overrides = {}
    for field in fields(FinalPipelineConfig):
        field_name = field.name
        tp = field.type
        arg_val = getattr(args, field_name)

        # Handle bool | int type
        if 'bool | int' in str(tp) or 'int | bool' in str(tp):
            if arg_val.lower() in ('true', '1', 'yes'):
                overrides[field_name] = True
            elif arg_val.lower() in ('false', '0', 'no'):
                overrides[field_name] = False
            else:
                overrides[field_name] = int(arg_val)
        else:
            overrides[field_name] = arg_val

    return FinalPipelineConfig(**overrides)


if __name__ == "__main__":
    config = parse_args_to_config(config)
    pipeline = FinalPipeline(config)
    is_root = not config.use_ddp or pipeline.is_root()

    if config.use_ddp:
        pipeline.setup_ddp()
        if is_root:
            print(f"{pipeline.world_size} GPUs")\

    if is_root:
        print("Preparing data...")
    pipeline.prepare_data()

    if config.use_ddp:
        dist.barrier()

    if is_root:
        print("Starting training...")
    pipeline.fit()

    if config.use_ddp:
        dist.destroy_process_group()
    if is_root:
        breakpoint()
