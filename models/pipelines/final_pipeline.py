"""
1. Multiple encoding strategies:
   - Quantize: 1/256 percentile-based token encoding
   - Cent quantize: cent-level delta encoding for small values
   - Sin encoding: Sinusoidal encoding for continuous values
   - CNN: Image encoding over K-line graphs (from draw.py) (TODO)

2. Multiple preprocessing approaches:
   - Raw values (close, open, high, low, volume)
   - Per-stock normalized values (using training set statistics)
   - Returns at varied time scales (1min, 30min, 6hr, 1day, 2day)
   - Intra-minute relative features (close/open, high/open, etc.)
   - Time-normalized values (cross-stock normalized per timestep)

4. Multiple prediction types and encodings:
   - Quantized predictions (cross-entropy loss)
   - Cent-based predictions
   - Mean predictions (Huber loss)
   - Mean + Variance predictions (NLL loss)
   - Quantile predictions (10th, 25th, 50th, 75th, 90th with sided loss)

5. Multiple prediction horizons:
   - 1min, 30min, 1day, 2day ahead
   - Next day OHLC predictions
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

from ..prelude.model import dummyLightning
from ..prelude.model import Rope

torch.autograd.set_detect_anomaly(True)


class PriceHistoryDataset(Dataset):
    def __init__(self, stock_data: Dict[str, np.ndarray],
                 stock_targets: Dict[str, np.ndarray],
                 stock_returns: Dict[str, np.ndarray],
                 seq_len: int,
                 pred_len: int):
        """
        Args:
            stock_data: Dict mapping stock_id to full feature array (T, F)
            stock_targets: Dict mapping stock_id to full target array (T,)
              - normalized prices
            stock_returns: Dict mapping stock_id to full raw close prices (T,)
              - for return calculation
            seq_len: Sequence length for context
            pred_len: Prediction length
        """
        self.stock_data = stock_data
        self.targets = stock_targets
        self.returns = stock_returns
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Prediction horizons in minutes: 1min, 30min, 1day, 2days
        self.horizons = [1, 30, 240, 480]
        self.num_horizons = len(self.horizons)

        # Build index: list of (stock_id, start_idx) tuples
        # Ensure we have enough future data for the longest horizon (480 min)
        self.index = []
        for stock_id, data in stock_data.items():
            max_horizon = max(self.horizons)
            num_windows = len(data) - seq_len - max_horizon  # for all horizons
            if num_windows > 0:
                for start_idx in range(num_windows):
                    self.index.append((stock_id, start_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stock_id, start_idx = self.index[idx]
        end_idx = start_idx + self.seq_len

        features = self.stock_data[stock_id][start_idx:end_idx]

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
        current_prices = self.returns[stock_id][current_positions]

        for h_idx, horizon in enumerate(self.horizons):
            future_positions = current_positions + horizon
            targets[:, h_idx] = self.targets[stock_id][future_positions]
            future_prices = self.returns[stock_id][future_positions]

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
            nn.Linear(self.hidden_dim, config.intermediate_dim),
            nn.SiLU(),
            nn.Linear(config.intermediate_dim, self.hidden_dim),
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
        self.embedding = nn.Embedding(config.num_bins, config.embed_dim)

    def forward(self, x: torch.Tensor, quantiles: np.ndarray) -> torch.Tensor:
        # x: (batch, 1) - single feature values
        quantiles_tensor = torch.from_numpy(quantiles[1:-1]).to(x.device, dtype=x.dtype)
        tokens = torch.bucketize(x.squeeze(-1), quantiles_tensor, right=True)
        result = self.embedding(tokens)
        return result


class CentsEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.max_cent_abs = config.max_cent_abs
        self.embedding = nn.Embedding(2 * config.max_cent_abs + 3,
                                      config.embed_dim)
        self.proj = nn.Linear(config.embed_dim * config.num_cent_features,
                              config.embed_dim)

    def forward(self, x):
        """Convert cent differences to tokens"""
        x = (x * 100).long().clamp(-self.max_cent_abs-1, self.max_cent_abs+1)
        x = x + self.max_cent_abs + 1
        x = x.squeeze(-1)  # TODO: ?
        return self.embedding(x)


class SinEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__()

        freqs = torch.exp(
            torch.linspace(0, np.log(config.max_freq), config.sin_embed_dim // 2)
        )
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        self.register_buffer('freqs', freqs)

        # Project flattened sin/cos features to embed_dim
        # Input will be (batch, features * embed_dim) after flattening
        # For 12 features: 12 * embed_dim
        self.proj = nn.Linear(config.num_features * config.sin_embed_dim, config.embed_dim)

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
        embeddings = [
            self.quantize_encoder(x_flat, self.quantiles),
            self.cent_encoder(x_flat[:, :self.num_cent_features]),
            self.sin_encoder(x_flat),
        ]
        output = self.combiner(torch.cat(embeddings, dim=-1))

        return output.unflatten(0, (batch_size, seq_len))


class MultiReadout(dummyLightning):
    def __init__(self, config):
        super().__init__(config)

        self.quantized_head = nn.Linear(config.hidden_dim,
                                        config.num_bins * self.num_horizons)
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
                            self.num_bins)
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
        self.config = config

        # DDP setup
        self.is_distributed = config.use_ddp
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self._ddp_model = None

        self.encoder = MultiEncoder(config)
        self.backbone = self._build_backbone()
        self.readout = MultiReadout(config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()

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

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank if not self.config.debug_ddp else 0)
            self.config.device = f'cuda:{self.local_rank}'

        # Initialize process group
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            dist.init_process_group(
                backend=self.config.ddp_backend,
                rank=self.rank,
                world_size=self.world_size
            )

        self.is_distributed = True

        # Move model to device before wrapping with DDP
        self.to(self.config.device)

        # Wrap model with DDP
        self._ddp_model = DDP(
            self,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.ddp_find_unused_parameters,
            static_graph=self.config.ddp_static_graph
        )

    def get_model(self):
        """Return the underlying model (unwrapped from DDP if necessary)"""
        return self._ddp_model.module if self._ddp_model is not None else self

    def barrier(self):
        """Synchronization barrier across all processes"""
        if self.is_distributed:
            dist.barrier()

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0

    def cleanup_ddp(self):
        """Cleanup DDP process group"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            self.is_distributed = False

    def all_reduce(self, tensor: torch.Tensor, op=None) -> torch.Tensor:
        """All-reduce operation across all processes"""
        if not self.is_distributed:
            return tensor

        if op is None:
            op = dist.ReduceOp.SUM

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes"""
        if not self.is_distributed:
            return tensor

        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def reduce_dict(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reduce a dictionary of metrics across all processes"""
        if not self.is_distributed:
            return metrics

        reduced = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                reduced[key] = self.all_reduce(value) / self.world_size
            else:
                reduced[key] = value

        return reduced

    def _build_backbone(self) -> dummyLightning:
        layers = []

        for _ in range(self.config.num_layers):
            layers.append(TransformerBlock(self.config))

        return nn.Sequential(*layers)

    def _get_cache_path(self) -> Path:
        """Get the path to the cache file"""
        return Path(self.config.cache_dir) / 'dataset_cache.npz'

    def _save_cache(self, train_stock_data: Dict[str, np.ndarray],
                    train_stock_targets: Dict[str, np.ndarray],
                    train_stock_returns: Dict[str, np.ndarray],
                    val_stock_data: Dict[str, np.ndarray],
                    val_stock_targets: Dict[str, np.ndarray],
                    val_stock_returns: Dict[str, np.ndarray],
                    quantiles: Dict[str, np.ndarray]):
        """Save dataset dictionaries to disk cache"""
        cache_path = self._get_cache_path()

        if self.is_main_process():
            print(f"Saving cache to {cache_path}...")

            # Convert dicts to a format savez can handle
            # We'll save with keys like 'train_data_<stock_id>', 'train_target_<stock_id>', etc.
            save_dict = {}

            # Save quantiles
            for key, value in quantiles.items():
                save_dict[f'quantile_{key}'] = value

            # Save train data
            for stock_id, data in train_stock_data.items():
                save_dict[f'train_data_{stock_id}'] = data
            for stock_id, targets in train_stock_targets.items():
                save_dict[f'train_target_{stock_id}'] = targets
            for stock_id, returns in train_stock_returns.items():
                save_dict[f'train_return_{stock_id}'] = returns

            # Save val data
            for stock_id, data in val_stock_data.items():
                save_dict[f'val_data_{stock_id}'] = data
            for stock_id, targets in val_stock_targets.items():
                save_dict[f'val_target_{stock_id}'] = targets
            for stock_id, returns in val_stock_returns.items():
                save_dict[f'val_return_{stock_id}'] = returns

            # Save stock IDs as metadata
            train_ids = list(train_stock_data.keys())
            val_ids = list(val_stock_data.keys())
            save_dict['_train_ids'] = np.array(train_ids, dtype=object)
            save_dict['_val_ids'] = np.array(val_ids, dtype=object)
            save_dict['_quantile_keys'] = np.array(list(quantiles.keys()), dtype=object)

            np.savez_compressed(cache_path, **save_dict)
            print(f"Cache saved successfully!")

    def _load_cache(self) -> Optional[Tuple[Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray],
                                             Dict[str, np.ndarray]]]:
        """Load dataset dictionaries from disk cache"""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            return None

        if self.is_main_process():
            print(f"Loading cache from {cache_path}...")

        data = np.load(cache_path, allow_pickle=True)

        # Reconstruct dictionaries
        train_stock_data = {}
        train_stock_targets = {}
        train_stock_returns = {}
        val_stock_data = {}
        val_stock_targets = {}
        val_stock_returns = {}
        quantiles = {}

        # Load metadata
        train_ids = data['_train_ids']
        val_ids = data['_val_ids']
        quantile_keys = data['_quantile_keys']

        # Load quantiles
        for key in quantile_keys:
            quantiles[key] = data[f'quantile_{key}']

        # Load train data
        for stock_id in train_ids:
            train_stock_data[stock_id] = data[f'train_data_{stock_id}']
            train_stock_targets[stock_id] = data[f'train_target_{stock_id}']
            train_stock_returns[stock_id] = data[f'train_return_{stock_id}']

        # Load val data
        for stock_id in val_ids:
            val_stock_data[stock_id] = data[f'val_data_{stock_id}']
            val_stock_targets[stock_id] = data[f'val_target_{stock_id}']
            val_stock_returns[stock_id] = data[f'val_return_{stock_id}']

        if self.is_main_process():
            print(f"Cache loaded successfully! "
                  f"{len(train_ids)} train stocks, {len(val_ids)} val stocks")

        return train_stock_data, train_stock_targets, train_stock_returns, val_stock_data, val_stock_targets, val_stock_returns, quantiles

    def prepare_data(self, path: Optional[str] = None, seq_len: int = 64,
                     num_bins: int = 256, train_frac: float = 0.9):
        """Prepare data with all preprocessing strategies"""

        # Try to load from cache if not forcing recalculation
        if not self.config.debug_no_cache:
            cached = self._load_cache()
            if cached is not None:
                train_stock_data, train_stock_targets, train_stock_returns, val_stock_data, val_stock_targets, val_stock_returns, quantiles = cached
                self.quantiles = quantiles
                self.encoder.quantiles = self.quantiles['close']

                # Build datasets
                pred_len = seq_len // 2
                self.train_dataset = PriceHistoryDataset(
                    train_stock_data, train_stock_targets, train_stock_returns,
                    seq_len, pred_len
                )
                self.val_dataset = PriceHistoryDataset(
                    val_stock_data, val_stock_targets, val_stock_returns,
                    seq_len, pred_len
                )
                return

        # Cache miss or force recalculation - compute from scratch
        if self.is_main_process():
            print("Computing data from scratch...")

        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        path = Path(path)

        if self.config.debug_data:
            df = pl.scan_parquet(str(path))
            n = self.config.debug_data if isinstance(self.config.debug_data,
                                                     int) else 5
            df = df.head(n * 300 * 20 * 240)
            ids = df.select('id').unique().head(n).collect()['id']
            df = df.filter(pl.col('id').is_in(ids.implode())).collect()
        else:
            df = pl.read_parquet(str(path))
        df = df.sort(['id', 'datetime'])
        assert len(df)

        df = self._compute_features(df)
        assert len(df)
        df = self._split_data(df, train_frac)
        assert len(df)
        self.quantiles = self._compute_quantiles(df, num_bins)

        self.encoder.quantiles = self.quantiles['close']

        # Build sequences
        train_stock_data, train_stock_targets, train_stock_returns, val_stock_data, val_stock_targets, val_stock_returns = self._bs(df, seq_len)
        self.train_dataset = train_stock_data  # Will be reassigned below
        self.val_dataset = val_stock_data  # Will be reassigned below

        # Save to cache
        self._save_cache(
            train_stock_data, train_stock_targets, train_stock_returns,
            val_stock_data, val_stock_targets, val_stock_returns,
            self.quantiles
        )

        # Build dataset objects
        pred_len = seq_len // 2
        self.train_dataset = PriceHistoryDataset(
            train_stock_data, train_stock_targets, train_stock_returns,
            seq_len, pred_len
        )
        self.val_dataset = PriceHistoryDataset(
            val_stock_data, val_stock_targets, val_stock_returns,
            seq_len, pred_len
        )

    def get_dataloader(self, dataset, shuffle: bool = True, drop_last: bool = True):
        """Create dataloader with DDP support via DistributedSampler"""
        from torch.utils.data import DataLoader

        if self.is_distributed:
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
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=drop_last
            )
        else:
            # Standard DataLoader for non-distributed training
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
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

        # Returns at different time scales (1min, 30min, 6hr, 24hr, 2days)
        # Note: Assuming 240 minutes per trading day (6.5 hours)
        df = df.with_columns(
            ret_1min=pl.col('close') - pl.col('close').shift(1).over('id'),
            ret_30min=pl.col('close') - pl.col('close').shift(30).over('id'),
            ret_1min_ratio=(pl.col('close') / pl.col('close').shift(1).over('id') - 1),
            ret_30min_ratio=(pl.col('close') / pl.col('close').shift(30).over('id') - 1),
            ret_1day_ratio=(pl.col('close') / pl.col('close').shift(240).over('id') - 1),
            ret_2day_ratio=(pl.col('close') / pl.col('close').shift(480).over('id') - 1),  # 2 days
        )

        # Intra-minute relative features
        df = df.with_columns([
            (pl.col('close') / pl.col('open')).alias('close_open_ratio'),
            (pl.col('high') / pl.col('open')).alias('high_open_ratio'),
            (pl.col('low') / pl.col('open')).alias('low_open_ratio'),
            (pl.col('high') / pl.col('low')).alias('high_low_ratio')
        ])

        # Fill NaN values
        df = df.with_columns([
            pl.col('ret_1min').fill_null(0.),
            pl.col('ret_30min').fill_null(0.),
            pl.col('ret_1min_ratio').fill_null(0.),
            pl.col('ret_30min_ratio').fill_null(0.),
            pl.col('ret_1day_ratio').fill_null(0.),
            pl.col('ret_2day_ratio').fill_null(0.),
            pl.col('close_open_ratio').fill_null(1.),
            pl.col('high_open_ratio').fill_null(1.),
            pl.col('low_open_ratio').fill_null(1.),
            pl.col('high_low_ratio').fill_null(1.)
        ])

        return df

    def _split_data(self, df: pl.DataFrame, train_frac: float) -> pl.DataFrame:
        cutoff = df.select(
            pl.col('datetime').unique().cast(pl.Int64).quantile(0.9)
        )

        df = df.with_columns(
            is_train=(pl.col('datetime').cast(pl.Int64) <= cutoff['datetime'])
        )
        assert len(df.filter('is_train'))
        assert len(df.filter(~pl.col('is_train')))

        stats = df.filter(pl.col('is_train')).group_by('id').agg([
            pl.col('close').mean().alias('mean_close'),
            pl.col('close').std().alias('std_close'),
            pl.col('volume').mean().alias('mean_volume'),
            pl.col('volume').std().alias('std_volume')
        ])

        df = df.join(stats, on='id', how='left')
        assert len(df)

        # normalize
        df = df.with_columns(
            ((pl.col('close') - pl.col('mean_close')) / (pl.col('std_close') + 1e-8)).alias('close_norm'),
            ((pl.col('volume') - pl.col('mean_volume')) / (pl.col('std_volume') + 1e-8)).alias('volume_norm'),
        )
        assert len(df)

        return df

    def _compute_quantiles(self, df: pl.DataFrame, num_bins: int)\
            -> Dict[str, torch.Tensor]:
        """Compute quantiles for different features"""

        train_df = df.filter(pl.col('is_train'))
        assert len(df)

        quantiles = {}
        for col in ['close', 'ret_1min', 'ret_30min', 'volume']:
            data = train_df[col].to_torch().float()
            n = len(data)
            assert n
            # Compute k-th values for each quantile position
            q_positions = torch.linspace(0, 1, num_bins + 1)
            k_indices = (q_positions * (n - 1)).long() + 1
            k_indices = torch.clamp(k_indices, 1, n)

            data.sort()
            quantile_values = data[k_indices - 1]  # -1 for 0-indexed
            quantiles[col] = quantile_values

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
                                         Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Build sequences and return the data dictionaries.

        Returns:
            train_stock_data, train_stock_targets, train_stock_returns,
            val_stock_data, val_stock_targets, val_stock_returns
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

        train_stock_data = {}
        train_stock_targets = {}

        val_stock_data = {}
        val_stock_targets = {}

        cols = [
            'close', 'close_norm', 'ret_1min',
            'ret_30min', 'ret_1min_ratio', 'ret_30min_ratio', 'ret_1day_ratio',
            'ret_2day_ratio', 'close_open_ratio', 'high_open_ratio',
            'low_open_ratio', 'high_low_ratio', 'volume', 'volume_norm'
        ]

        # TODO: align for cross attention
        for stock_id in df['id'].unique():
            stock_df = df.filter(pl.col('id') == stock_id).sort('datetime')

            if len(stock_df) <= seq_len + 1:
                continue

            features = stock_df.select(cols).to_numpy().astype(np.float32)
            is_train = stock_df['is_train'].to_numpy()
            close = stock_df['close_norm'].to_numpy().astype(np.float32)
            
            # Split each stock's data by time - train data and val data
            train_mask = is_train.astype(bool)
            val_mask = ~train_mask

            # Only add to train set if we have enough train data
            if train_mask.sum() > seq_len + 1:
                train_stock_data[stock_id] = features[train_mask]
                train_stock_targets[stock_id] = close[train_mask]

            # Only add to val set if we have enough val data
            if val_mask.sum() > seq_len + 1:
                val_stock_data[stock_id] = features[val_mask]
                val_stock_targets[stock_id] = close[val_mask]

        return train_stock_data, train_stock_targets, train_stock_returns, val_stock_data, val_stock_targets, val_stock_returns

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
        horizon_names = ['1min', '30min', '1day', '2day']

        features = self.forward(x)[:, seq_len//2:, :]
        # Get predictions for all sequence positions
        # But we only care about predictions from positions seq_len//2 onwards
        pred_quantized = self.readout(features, 'quantized')  # (batch, pred_len, num_horizons, num_bins)
        y = y.to(pred_quantized.dtype)

        # Quantized prediction loss
        quantiles_tensor = torch.from_numpy(self.quantiles['close']).to(y.device, dtype=y.dtype)
        target_quantized = torch.bucketize(y, quantiles_tensor)  # (batch, pred_len, num_horizons)
        target_quantized = torch.clamp(target_quantized, 0, self.config.num_bins - 1)

        if self.config.debug_horizons:
            # Compute per-horizon losses
            losses['quantized'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                h_loss = self.ce_loss(
                    pred_quantized[:, :, h_idx, :].reshape(-1, self.config.num_bins),
                    target_quantized[:, :, h_idx].reshape(-1)
                )
                losses[f'quantized_{h_name}'] = h_loss
                losses['quantized'] += h_loss / len(horizon_names)
        else:
            losses['quantized'] = self.ce_loss(
                pred_quantized.reshape(-1, self.config.num_bins),
                target_quantized.reshape(-1)
            )

        # Mean prediction loss (Huber)
        pred_mean = self.readout(features, 'mean')  # (batch, pred_len, num_horizons)

        if self.config.debug_horizons:
            # Compute per-horizon losses
            losses['mean'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                h_loss = self.huber_loss(pred_mean[:, :, h_idx], y[:, :, h_idx])
                losses[f'mean_{h_name}'] = h_loss
                losses['mean'] += h_loss / len(horizon_names)
        else:
            losses['mean'] = self.huber_loss(pred_mean, y)

        # Mean + Variance prediction loss (NLL)
        pred_var = self.readout(features, 'var')  # (batch, pred_len, num_horizons)
        pred_var += 1e-8  # (batch, pred_len, num_horizons)

        if self.config.debug_horizons:
            # Compute per-horizon losses
            losses['nll'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                # Full Gaussian NLL: 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y-μ)²/σ²
                nll = 1/2 * (torch.log(2 * torch.pi * pred_var[:, :, h_idx]) +
                             (y[:, :, h_idx] - pred_mean[:, :, h_idx]) ** 2 / pred_var[:, :, h_idx])
                h_loss = nll.mean()
                losses[f'nll_{h_name}'] = h_loss
                losses['nll'] += h_loss / len(horizon_names)
        else:
            # Full Gaussian NLL: 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y-μ)²/σ²
            nll = 1/2 * (torch.log(2 * torch.pi * pred_var) +
                         (y - pred_mean) ** 2 / pred_var)
            losses['nll'] = nll.mean()

        # Quantile prediction loss
        pred_quantiles = self.readout(features, 'quantile')

        if self.config.debug_horizons:
            # Compute per-horizon losses
            losses['quantile'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                h_loss = self._quantile_loss(pred_quantiles[:, :, h_idx, :], y[:, :, h_idx])
                losses[f'quantile_{h_name}'] = h_loss
                losses['quantile'] += h_loss / len(horizon_names)
        else:
            losses['quantile'] = self._quantile_loss(pred_quantiles, y)

        # ===== Return prediction losses =====
        y_returns = y_returns.to(pred_mean.dtype)

        # Return mean prediction loss (Huber)
        pred_return_mean = self.readout(features, 'return_mean')  # (batch, pred_len, num_horizons)

        if self.config.debug_horizons:
            losses['return_mean'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                h_loss = self.huber_loss(pred_return_mean[:, :, h_idx], y_returns[:, :, h_idx])
                losses[f'return_mean_{h_name}'] = h_loss
                losses['return_mean'] += h_loss / len(horizon_names)
        else:
            losses['return_mean'] = self.huber_loss(pred_return_mean, y_returns)

        # Return variance prediction loss (NLL)
        pred_return_var = self.readout(features, 'return_var')  # (batch, pred_len, num_horizons)
        pred_return_var += 1e-8

        if self.config.debug_horizons:
            losses['return_nll'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                nll = 1/2 * (torch.log(2 * torch.pi * pred_return_var[:, :, h_idx]) +
                             (y_returns[:, :, h_idx] - pred_return_mean[:, :, h_idx]) ** 2 / pred_return_var[:, :, h_idx])
                h_loss = nll.mean()
                losses[f'return_nll_{h_name}'] = h_loss
                losses['return_nll'] += h_loss / len(horizon_names)
        else:
            nll = 1/2 * (torch.log(2 * torch.pi * pred_return_var) +
                         (y_returns - pred_return_mean) ** 2 / pred_return_var)
            losses['return_nll'] = nll.mean()

        # Return quantile prediction loss
        pred_return_quantiles = self.readout(features, 'return_quantile')

        if self.config.debug_horizons:
            losses['return_quantile'] = 0.0
            for h_idx, h_name in enumerate(horizon_names):
                h_loss = self._quantile_loss(pred_return_quantiles[:, :, h_idx, :], y_returns[:, :, h_idx])
                losses[f'return_quantile_{h_name}'] = h_loss
                losses['return_quantile'] += h_loss / len(horizon_names)
        else:
            losses['return_quantile'] = self._quantile_loss(pred_return_quantiles, y_returns)

        assert (
            losses['quantized'] >= 0 and
            losses['mean'] >= 0 and
            losses['quantile'] >= 0 and
            losses['return_mean'] >= 0 and
            losses['return_quantile'] >= 0
        ), losses
        # Combine losses
        total_loss = (
            0.15 * losses['quantized'] +
            0.15 * losses['mean'] +
            0.15 * losses['nll'] +
            0.15 * losses['quantile'] +
            0.15 * losses['return_mean'] +
            0.15 * losses['return_nll'] +
            0.10 * losses['return_quantile']
        )

        # Log aggregate losses with hierarchical grouping
        # Price prediction losses
        self.log('price_prediction/quantized', losses['quantized'])
        self.log('price_prediction/mean', losses['mean'])
        self.log('price_prediction/nll', losses['nll'])
        self.log('price_prediction/quantile', losses['quantile'])

        # Return prediction losses
        self.log('return_prediction/mean', losses['return_mean'])
        self.log('return_prediction/nll', losses['return_nll'])
        self.log('return_prediction/quantile', losses['return_quantile'])

        # Log per-horizon losses
        if self.config.debug_horizons:
            for h_name in horizon_names:
                # Price prediction per horizon
                self.log(f'price_prediction/horizons/{h_name}/quantized', losses[f'quantized_{h_name}'])
                self.log(f'price_prediction/horizons/{h_name}/mean', losses[f'mean_{h_name}'])
                self.log(f'price_prediction/horizons/{h_name}/nll', losses[f'nll_{h_name}'])
                self.log(f'price_prediction/horizons/{h_name}/quantile', losses[f'quantile_{h_name}'])

                # Return prediction per horizon
                self.log(f'return_prediction/horizons/{h_name}/mean', losses[f'return_mean_{h_name}'])
                self.log(f'return_prediction/horizons/{h_name}/nll', losses[f'return_nll_{h_name}'])
                self.log(f'return_prediction/horizons/{h_name}/quantile', losses[f'return_quantile_{h_name}'])

        result = {
            'loss': total_loss,
            'quantized_loss': losses['quantized'],
            'mean_loss': losses['mean'],
            'nll_loss': losses['nll'],
            'quantile_loss': losses['quantile'],
            'return_mean_loss': losses['return_mean'],
            'return_nll_loss': losses['return_nll'],
            'return_quantile_loss': losses['return_quantile']
        }

        # Add per-horizon losses to result if enabled
        if self.config.debug_horizons:
            for h_name in horizon_names:
                result[f'quantized_{h_name}'] = losses[f'quantized_{h_name}']
                result[f'mean_{h_name}'] = losses[f'mean_{h_name}']
                result[f'nll_{h_name}'] = losses[f'nll_{h_name}']
                result[f'quantile_{h_name}'] = losses[f'quantile_{h_name}']
                result[f'return_mean_{h_name}'] = losses[f'return_mean_{h_name}']
                result[f'return_nll_{h_name}'] = losses[f'return_nll_{h_name}']
                result[f'return_quantile_{h_name}'] = losses[f'return_quantile_{h_name}']

        return result

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Quantile loss with sided weighting

        Args:
            pred: (batch, pred_len, num_horizons, num_quantiles) or
                  (batch, pred_len, num_quantiles)
            target: (batch, pred_len, num_horizons) or
                    (batch, pred_len)
        """
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9],
                                 device=pred.device)

        # Handle both 3D and 4D tensors
        if pred.ndim == 3:
            # Shape: (batch, pred_len, num_quantiles)
            quantiles = quantiles.view(1, 1, -1)
        else:
            # Shape: (batch, pred_len, num_horizons, num_quantiles)
            quantiles = quantiles.view(1, 1, 1, -1)

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

        return torch.mean(huber * weight)


@dataclass
class FinalPipelineConfig:
    """Configuration for the final pipeline"""

    # Model architecture
    hidden_dim: int = 256
    intermediate_ratio: float = 2.
    num_layers: int = 6
    num_heads: int = 8
    attn_ratio: float = 1.
    max_freq: float = 10000.
    standard_rope: bool = False  # use fine-grained rope
    qk_norm: bool = False

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Data
    seq_len: int = 256
    train_ratio: float = 0.9
    num_bins: int = 256
    max_cent_abs: int = 64
    embed_dim: int = 256

    num_horizons: int = 4
    num_quantiles: int = 5

    # Device
    device: str = 'cuda'
    num_workers: int = 4

    # Muon optimizer for 2D parameters
    use_muon: bool = True

    debug_data: bool | int = False
    debug_horizons: bool = False

    # DDP settings
    use_ddp: bool = False
    ddp_backend: str = 'nccl'
    ddp_find_unused_parameters: bool = False
    ddp_static_graph: bool = False
    master_addr: str = 'localhost'
    master_port: str = '12355'

    debug_ddp: bool = False
    debug_model: bool = False

    # Cache settings
    cache_dir: Optional[str] = None
    debug_no_cache: bool = False

    def __post_init__(self):
        if self.debug_model:
            self.hidden_dim //= 4
            self.num_heads //= 2
            self.num_layers //= 2
            self.embed_dim //= 2

        # Model
        self.intermediate_dim = int(self.hidden_dim * self.intermediate_ratio)
        self.head_dim = int(self.hidden_dim * self.attn_ratio / self.num_heads)

        # Embedding
        self.sin_embed_dim = self.embed_dim // 2
        self.num_cents = self.max_cent_abs * 2 + 1

        # Cache directory
        if self.cache_dir is None:
            base_cache = Path.home() / 'h' / 'cache' / 'pipeline_data'
            if self.debug_data:
                n = self.debug_data if isinstance(self.debug_data, int) else 5
                self.cache_dir = str(base_cache / f'debug_{n}')
            else:
                self.cache_dir = str(base_cache / 'full')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


config = FinalPipelineConfig(
    debug_data=True,
    debug_no_cache=True,
    debug_ddp=True,
    debug_model=True,
    debug_horizons=True,
)

def parse_args_to_config(base_config: FinalPipelineConfig) -> FinalPipelineConfig:
    """Parse CLI arguments and override config fields"""
    import argparse
    import sys
    from dataclasses import fields

    parser = argparse.ArgumentParser(description='Train FinalPipeline')

    # Dynamically add arguments from dataclass fields
    for field in fields(FinalPipelineConfig):
        field_name = field.name
        field_type = field.type
        default_val = getattr(base_config, field_name)

        # Handle different types
        if field_type == bool or field_type == 'bool':
            parser.add_argument(f'--{field_name}', action='store_true',
                              help=f'Set {field_name}=True')
            parser.add_argument(f'--no_{field_name}', dest=field_name,
                              action='store_false', help=f'Set {field_name}=False')
            parser.set_defaults(**{field_name: default_val})
        elif 'bool | int' in str(field_type) or 'int | bool' in str(field_type):
            # Special handling for bool | int union type
            parser.add_argument(f'--{field_name}', type=str, default=str(default_val),
                              help=f'{field_name} (bool or int)')
        elif field_type == int or field_type == 'int':
            parser.add_argument(f'--{field_name}', type=int, default=default_val,
                              help=f'{field_name} (default: {default_val})')
        elif field_type == float or field_type == 'float':
            parser.add_argument(f'--{field_name}', type=float, default=default_val,
                              help=f'{field_name} (default: {default_val})')
        elif field_type == str or field_type == 'str' or 'Optional[str]' in str(field_type):
            parser.add_argument(f'--{field_name}', type=str, default=default_val,
                              help=f'{field_name} (default: {default_val})')

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
        field_type = field.type
        arg_val = getattr(args, field_name)

        # Handle bool | int type
        if 'bool | int' in str(field_type) or 'int | bool' in str(field_type):
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
    import sys

    # Parse CLI arguments
    config = parse_args_to_config(config)

    # Check if running with torchrun
    config.use_ddp = 'RANK' in os.environ or config.use_ddp

    pipeline = FinalPipeline(config)

    if config.use_ddp:
        pipeline.setup_ddp()
        if pipeline.is_main_process():
            print(f"Running DDP training on {pipeline.world_size} GPUs")
            print(f"Rank: {pipeline.rank}, Local Rank: {pipeline.local_rank}")

    # Prepare data
    if not config.use_ddp or pipeline.is_main_process():
        print("Preparing data...")
    pipeline.prepare_data()

    # Synchronize after data preparation
    if config.use_ddp:
        pipeline.barrier()

    # Train
    if not config.use_ddp or pipeline.is_main_process():
        print("Starting training...")
    pipeline.fit()

    # Cleanup
    if config.use_ddp:
        pipeline.cleanup_ddp()

    if not config.use_ddp or pipeline.is_main_process():
        breakpoint()
