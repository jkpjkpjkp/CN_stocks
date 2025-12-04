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

All data storage uses on-disk DuckDB in ../data/pipeline.duckdb
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import os

from ..prelude.model import dummyLightning, Rope, tm

torch.autograd.set_detect_anomaly(True)


class PriceHistoryDataset(Dataset):
    """Dataset that fetches data directly from DuckDB on-disk storage."""

    def __init__(self, config, db_path: Path, split: str):
        """
        Args:
            config: FinalPipelineConfig
            db_path: Path to DuckDB database file
            split: 'train' or 'val'
        """
        self.db_path = str(db_path)
        self.split = split
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.horizons = config.horizons
        self.num_horizons = len(self.horizons)
        self.features = config.features

        # Build index from DuckDB
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            # Get all (stock_id, start_idx) pairs from the index table
            self.index = con.execute(f"""
                SELECT stock_id, start_idx FROM {split}_index ORDER BY rowid
            """).fetchall()
        finally:
            con.close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stock_id, start_idx = self.index[idx]
        end_idx = start_idx + self.seq_len

        # Each worker gets its own connection
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            # Fetch features for this window
            features_cols = ', '.join(self.features)
            features_result = con.execute(f"""
                SELECT {features_cols}
                FROM {self.split}_data
                WHERE stock_id = ? AND row_idx >= ? AND row_idx < ?
                ORDER BY row_idx
            """, [stock_id, start_idx, end_idx]).fetchnumpy()

            features = np.column_stack([features_result[f] for f in self.features]).astype(np.float32)

            # Get prediction positions' targets (latter half)
            target_start = start_idx + self.seq_len // 2
            max_horizon = max(self.horizons)
            target_end = target_start + self.pred_len + max_horizon

            # Fetch close_norm for targets
            targets_result = con.execute(f"""
                SELECT row_idx, close_norm
                FROM {self.split}_data
                WHERE stock_id = ? AND row_idx >= ? AND row_idx < ?
                ORDER BY row_idx
            """, [stock_id, target_start, target_end]).fetchnumpy()

            close_norm = targets_result['close_norm'].astype(np.float32)

        finally:
            con.close()

        # Build targets and return_targets
        targets = np.zeros((self.pred_len, self.num_horizons), dtype=np.float32)
        return_targets = np.zeros((self.pred_len, self.num_horizons), dtype=np.float32)

        current_prices = close_norm[:self.pred_len]

        for h_idx, horizon in enumerate(self.horizons):
            future_prices = close_norm[horizon:horizon + self.pred_len]
            targets[:, h_idx] = future_prices
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
        if op is None:
            op = dist.ReduceOp.SUM

        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes"""
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=0)

    def _get_db_path(self) -> Path:
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        return Path(self.config.db_path)

    def _check_db_ready(self) -> bool:
        """Check if DuckDB database exists and has required tables"""
        db_path = self._get_db_path()
        if not db_path.exists():
            return False

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            required = {'train_data', 'val_data', 'train_index', 'val_index', 'quantiles'}
            return required.issubset(table_names)
        except Exception:
            return False
        finally:
            con.close()

    def _load_quantiles_from_db(self) -> torch.Tensor:
        """Load quantiles from DuckDB"""
        db_path = self._get_db_path()
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            result = con.execute("""
                SELECT feature_idx, q_idx, value FROM quantiles ORDER BY feature_idx, q_idx
            """).fetchnumpy()

            n_features = result['feature_idx'].max() + 1
            n_quantiles = result['q_idx'].max() + 1
            quantiles = np.zeros((n_quantiles, n_features), dtype=np.float32)
            quantiles[result['q_idx'], result['feature_idx']] = result['value']
            return torch.from_numpy(quantiles).float()
        finally:
            con.close()

    def prepare_data(self, path: Optional[str] = None):
        """Prepare data using on-disk DuckDB storage."""
        db_path = self._get_db_path()

        # Check if DB already exists with required tables
        if not self.config.debug_data and self._check_db_ready():
            if self.is_root():
                print(f"Loading from existing DuckDB: {db_path}")
            self.quantiles = self._load_quantiles_from_db()
            self.encoder.quantiles = self.quantiles
            self.train_dataset = PriceHistoryDataset(self.config, db_path, 'train')
            self.val_dataset = PriceHistoryDataset(self.config, db_path, 'val')
            return

        # Need to build DB from scratch
        if self.is_root():
            print("Building DuckDB from scratch...")

        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        path = Path(path)

        if self.is_root():
            self._build_duckdb(path, db_path)

        self.quantiles = self._load_quantiles_from_db()
        self.encoder.quantiles = self.quantiles
        self.train_dataset = PriceHistoryDataset(self.config, db_path, 'train')
        self.val_dataset = PriceHistoryDataset(self.config, db_path, 'val')

    def _build_duckdb(self, parquet_path: Path, db_path: Path):
        if db_path.exists():
            db_path.unlink()

        # Use in-memory database for faster processing
        con = duckdb.connect(':memory:')
        con.execute(f"SET memory_limit='{self.ram_limit}GB'")

        con.execute(f"""
            CREATE VIEW raw_data AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

        con.execute(f"""
            CREATE VIEW df AS
            {self._compute_features_sql()}
        """)

        print("Step 2-4: Load cached cutoff and stats, or compute them...")
        stats_cache_path = db_path.parent / 'stats_cache.parquet'
        cutoff_cache_path = db_path.parent / 'cutoff_cache.txt'

        # Get stock IDs first (needed for debug filtering)
        print("Step 2a: Get stock IDs to process...")
        if self.config.debug_data:
            # Exploit sorted order: scan until we find debug_data distinct IDs
            stock_ids = con.execute(f"""
                SELECT DISTINCT id FROM raw_data LIMIT {self.config.debug_data}
            """).fetchall()
            stock_ids = [row[0] for row in stock_ids]
            stock_ids_str = ", ".join(f"'{s}'" for s in stock_ids)
            stock_filter = f"WHERE id IN ({stock_ids_str})"
        else:
            stock_ids = None  # Will be set from stats table later
            stock_filter = ""

        # Load or compute cutoff
        if cutoff_cache_path.exists():
            cutoff = int(float(cutoff_cache_path.read_text().strip()))
            print(f"  Loaded cached cutoff: {cutoff}")
        else:
            print("  Computing train/val cutoff (90th percentile of datetime)...")
            cutoff = con.execute("""
                SELECT QUANTILE_CONT(epoch_ns(datetime), 0.9) as cutoff
                FROM (SELECT datetime FROM raw_data USING SAMPLE 1000000)
            """).fetchone()[0]
            print(f"  Train/val cutoff timestamp: {cutoff}")
            cutoff_cache_path.write_text(str(cutoff))

        # Compute stats - use cache only for full runs, always recompute for debug
        if not self.config.debug_data and stats_cache_path.exists():
            con.execute(f"""
                CREATE TABLE stats AS
                SELECT * FROM read_parquet('{stats_cache_path}')
            """)
            print(f"  Loaded cached stats from {stats_cache_path}")
            stock_ids = con.execute("SELECT id FROM stats ORDER BY id").fetchall()
            stock_ids = [row[0] for row in stock_ids]
        else:
            print("  Computing per-stock normalization stats...")
            con.execute(f"""
                CREATE TABLE stats AS
                SELECT
                    id,
                    AVG(close) AS mean_close,
                    STDDEV(close) AS std_close,
                    AVG(volume) AS mean_volume,
                    STDDEV(volume) AS std_volume
                FROM raw_data
                {stock_filter}
                {"AND" if stock_filter else "WHERE"} epoch_ns(datetime) <= {cutoff}
                GROUP BY id
            """)
            # Only cache full stats
            if not self.config.debug_data:
                con.execute(f"""
                    COPY stats TO '{stats_cache_path}' (FORMAT PARQUET)
                """)
                print(f"  Cached stats to {stats_cache_path}")
                stock_ids = con.execute("SELECT id FROM stats ORDER BY id").fetchall()
                stock_ids = [row[0] for row in stock_ids]

        total_stocks = len(stock_ids)
        print(f"  Total stocks: {total_stocks}")

        print("Step 6: Create data tables...")
        # Features that need normalization with per-stock stats
        norm_features = {'close_norm', 'volume_norm'}

        # Build SELECT columns: normalized features computed inline, direct features from df_split
        def build_select_cols():
            cols = []
            for f in self.features:
                if f == 'close_norm':
                    cols.append('(d.close - COALESCE(s.mean_close, 0)) / (COALESCE(s.std_close, 1e-8) + 1e-8) AS close_norm')
                elif f == 'volume_norm':
                    cols.append('(d.volume - COALESCE(s.mean_volume, 0)) / (COALESCE(s.std_volume, 1e-8) + 1e-8) AS volume_norm')
                else:
                    cols.append(f'd.{f}')
            return ',\n                    '.join(cols)

        select_cols = build_select_cols()

        # Build stock filter for debug mode
        if self.config.debug_data:
            stock_ids_str = ", ".join(f"'{s}'" for s in stock_ids)
            stock_filter = f"WHERE id IN ({stock_ids_str})"
        else:
            stock_filter = ""

        # First, materialize df with features (the expensive window function computation)
        print("  Materializing features (window functions)...")
        con.execute(f"""
            CREATE TABLE df_materialized AS
            SELECT * FROM df {stock_filter}
        """)
        con.execute("CREATE INDEX df_mat_id_idx ON df_materialized(id)")

        # Now train/val split and normalization is fast since features are pre-computed
        print("  Building train_data...")
        con.execute(f"""
            CREATE TABLE train_data AS
            SELECT
                d.id AS stock_id,
                ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.datetime) - 1 AS row_idx,
                {select_cols}
            FROM df_materialized d
            LEFT JOIN stats s ON d.id = s.id
            WHERE epoch_ns(d.datetime) <= {cutoff}
        """)

        # Create val_data table directly
        print("  Building val_data...")
        con.execute(f"""
            CREATE TABLE val_data AS
            SELECT
                d.id AS stock_id,
                ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.datetime) - 1 AS row_idx,
                {select_cols}
            FROM df_materialized d
            LEFT JOIN stats s ON d.id = s.id
            WHERE epoch_ns(d.datetime) > {cutoff}
        """)

        # Drop materialized table to free memory
        con.execute("DROP TABLE df_materialized")

        # Create indexes for fast lookups
        print("Step 7: Creating indexes...")
        con.execute("CREATE INDEX train_data_idx ON train_data(stock_id, row_idx)")
        con.execute("CREATE INDEX val_data_idx ON val_data(stock_id, row_idx)")

        print("Step 8: Compute and store quantiles from train_data...")
        sample_size = 10000 if self.config.debug_data else 1000000
        self._compute_and_store_quantiles(con, sample_size)

        print("Step 9: Build index tables for dataset iteration...")
        seq_len = self.config.seq_len
        max_horizon = max(self.horizons)

        # Build train_index
        print("  Building train_index...")
        con.execute(f"""
            CREATE TABLE train_index AS
            WITH stock_lengths AS (
                SELECT stock_id, MAX(row_idx) + 1 AS stock_len
                FROM train_data
                GROUP BY stock_id
            ),
            valid_stocks AS (
                SELECT stock_id, stock_len
                FROM stock_lengths
                WHERE stock_len > {seq_len} + {max_horizon}
            ),
            indices AS (
                SELECT
                    stock_id,
                    generate_series(0, stock_len - {seq_len} - {max_horizon} - 1) AS start_idx
                FROM valid_stocks
            )
            SELECT stock_id, start_idx FROM indices
        """)

        # Build val_index
        print("  Building val_index...")
        con.execute(f"""
            CREATE TABLE val_index AS
            WITH stock_lengths AS (
                SELECT stock_id, MAX(row_idx) + 1 AS stock_len
                FROM val_data
                GROUP BY stock_id
            ),
            valid_stocks AS (
                SELECT stock_id, stock_len
                FROM stock_lengths
                WHERE stock_len > {seq_len} + {max_horizon}
            ),
            indices AS (
                SELECT
                    stock_id,
                    generate_series(0, stock_len - {seq_len} - {max_horizon} - 1) AS start_idx
                FROM valid_stocks
            )
            SELECT stock_id, start_idx FROM indices
        """)

        # Get counts
        train_count = con.execute("SELECT COUNT(*) FROM train_index").fetchone()[0]
        val_count = con.execute("SELECT COUNT(*) FROM val_index").fetchone()[0]
        print(f"  Train samples: {train_count}, Val samples: {val_count}")

        # Persist in-memory database to disk
        print("Step 10: Persisting database to disk...")
        con.execute(f"ATTACH '{db_path}' AS disk_db")
        for table in ['train_data', 'val_data', 'train_index', 'val_index', 'quantiles']:
            con.execute(f"CREATE TABLE disk_db.{table} AS SELECT * FROM {table}")
        con.close()

        print(f"DuckDB database built at {db_path}")

    def _compute_and_store_quantiles(self, con: duckdb.DuckDBPyConnection, sample_size: int):
        """Compute quantiles from train_data table and store in DuckDB table."""
        n_quantize = self.config.n_quantize
        q_positions = [i / n_quantize for i in range(n_quantize + 1)]
        features_cols = ', '.join(self.features)

        # Sample rows from the already-created train_data table
        sampled = con.execute(f"""
            SELECT {features_cols}
            FROM train_data
            USING SAMPLE {sample_size}
        """).fetchnumpy()

        # Create quantiles table
        con.execute("""
            CREATE TABLE quantiles (
                feature_idx INTEGER,
                q_idx INTEGER,
                value DOUBLE,
                PRIMARY KEY (feature_idx, q_idx)
            )
        """)

        # Compute and insert quantiles for each feature
        for f_idx, col in enumerate(self.features):
            col_data = np.sort(sampled[col])
            n = len(col_data)

            for q_idx, q in enumerate(q_positions):
                idx = q * (n - 1)
                idx_lower = int(np.floor(idx))
                idx_upper = min(int(np.ceil(idx)), n - 1)

                if idx_lower == idx_upper:
                    quantile_val = col_data[idx_lower]
                else:
                    weight = idx - idx_lower
                    quantile_val = (1 - weight) * col_data[idx_lower] + weight * col_data[idx_upper]

                con.execute(
                    "INSERT INTO quantiles VALUES (?, ?, ?)",
                    [f_idx, q_idx, float(quantile_val)]
                )

    def get_dataloader(self, dataset, shuffle=False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=not self.debug_data,
        )
        # When using DistributedSampler, shuffle must be False in DataLoader
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=not self.debug_data,
        )

    @property
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    @property
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def _compute_features_sql(self) -> str:
        """Return SQL to compute all feature types"""
        return """
            SELECT *,
                -- Returns at different time scales (1min, 30min, 1day, 2days)
                COALESCE(LEAD(close, 1) OVER (PARTITION BY id ORDER BY datetime) - close, 0) AS delta_1min,
                COALESCE(LEAD(close, 30) OVER (PARTITION BY id ORDER BY datetime) - close, 0) AS delta_30min,
                COALESCE(LEAD(close, 1) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_1min,
                COALESCE(LEAD(close, 30) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_30min,
                COALESCE(LEAD(close, 240) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_1day,
                COALESCE(LEAD(close, 480) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_2day,
                -- Intra-minute relative features
                COALESCE(close / NULLIF(open, 0), 1) AS close_open,
                COALESCE(high / NULLIF(open, 0), 1) AS high_open,
                COALESCE(low / NULLIF(open, 0), 1) AS low_open,
                COALESCE(high / NULLIF(low, 0), 1) AS high_low
            FROM raw_data
        """


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
    db_path: str | None = None  # Path to DuckDB database file

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
    ddp_backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: str = '12355'

    debug_ddp: bool = False
    num_workers: int | None = None
    shrink_model: bool = False
    device: str = 'cuda'
    
    ram_limit: int = 368

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

        # DuckDB path (in ../data relative to source)
        if self.db_path is None:
            db_name = f'pipeline_debug_{self.debug_data}.duckdb' if self.debug_data else 'pipeline.duckdb'
            self.db_path = str(Path(__file__).parent.parent.parent.parent / 'data' / db_name)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)


config = FinalPipelineConfig(
    shrink_model=True,
    debug_data=10,
    debug_model=True,
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
    p = FinalPipeline(config)

    p.setup_ddp()
    if p.is_root():
        print(f"{p.world_size} GPUs")\

    if p.is_root():
        print("Preparing data...")
    p.prepare_data()
    assert len(p.train_dataset)
    assert len(p.val_dataset)

    dist.barrier()

    if p.is_root():
        print("Starting training...")
    p.fit()
    dist.destroy_process_group()
    
    if p.is_root():
        breakpoint()
