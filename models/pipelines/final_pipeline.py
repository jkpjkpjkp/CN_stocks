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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import duckdb
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass
import os

from ..prelude.model import dummyLightning, dummyConfig, TM


class PriceHistoryDataset(Dataset):
    """Dataset backed by memory-mapped numpy arrays."""
    def __init__(self, config, split: str):
        self.mmap_dir = Path(config.mmap_dir)
        self.config = config
        self.horizons, self.seq_len, self.features, self.num_horizons = \
            config.horizons, config.seq_len, config.features, config.num_horizons
        
        self.stock_ids = np.load(config.mmap_dir + f'{split}_stock_ids.npy')
        self.cumsums = np.load(config.mmap_dir + f'{split}_cumsums.npy')
        self.stock_offsets = np.load(config.mmap_dir + f'{split}_stock_offsets.npy')
        self.total_samples = int(self.cumsums[-1]) if len(self.cumsums) > 0 else 0
        self.data = np.load(
            config.mmap_dir + f'{split}_data.npy',
            mmap_mode='r'
        )  # shape: (total_rows, len(db_features))

        self.db_feat_idx = {f: i for i, f in enumerate(config.db_features)}

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        stock_rank = np.searchsorted(self.cumsums, idx, side='right')
        prev_cumsum = int(self.cumsums[stock_rank - 1]) if stock_rank else 0
        local_idx = idx - prev_cumsum
        stock_offset = int(self.stock_offsets[stock_rank])

        # slice
        max_horizon = max(self.horizons)
        target_start = local_idx + self.seq_len // 2
        target_end = target_start + self.seq_len // 2 + max_horizon
        end_idx = max(local_idx + self.seq_len, target_end)
        global_start = stock_offset + local_idx
        global_end = stock_offset + end_idx
        raw_data = self.data[global_start:global_end]

        def g(feat):
            return raw_data[:, self.db_feat_idx[feat]]
        all_features = np.zeros((raw_data.shape[0], len(self.features)), dtype=np.float32)
        for feat_idx, feat in enumerate(self.features):
            if feat in self.db_feat_idx:
                all_features[:, feat_idx] = raw_data[:, self.db_feat_idx[feat]]
            elif feat == 'close_norm':
                close = g('close')
                all_features[:, feat_idx] = close / close[0] - 1
            elif feat == 'volume_norm':
                volume = g('volume')
                all_features[:, feat_idx] = ((volume - volume[:self.seq_len // 2].mean())
                                             / (volume[:self.seq_len // 2].std() + 1e-8))
            elif feat == 'delta_1min':
                close = g('close')
                all_features[:-1, feat_idx] = close[1:] - close[:-1]
            elif feat == 'ret_1min':
                close = g('close')
                all_features[:-1, feat_idx] = (close[1:] / (close[:-1] + 1e-8) - 1) * 100
            elif feat == 'close_open':
                all_features[:, feat_idx] = (g('close') / (g('open') + 1e-8) - 1) * 1000
            elif feat == 'high_open':
                all_features[:, feat_idx] = (g('high') / (g('open') + 1e-8) - 1) * 1000
            elif feat == 'low_open':
                all_features[:, feat_idx] = (g('low') / (g('open') + 1e-8) - 1) * 1000
            elif feat == 'high_low':
                all_features[:, feat_idx] = (g('high') / (g('low') + 1e-8) - 1) * 1000

        y = np.zeros((self.seq_len//2, self.num_horizons), dtype=np.float32)
        y_ret = np.zeros((self.seq_len//2, self.num_horizons), dtype=np.float32)
        for h_idx, horizon in enumerate(self.horizons):
            y[:, h_idx] = close[horizon + self.seq_len // 2: horizon + self.seq_len] / close[0] - 1
            y_ret[:, h_idx] = (close[horizon + self.seq_len // 2: horizon + self.seq_len]
                               / (close[self.seq_len // 2: self.seq_len] + 1e-8))

        return (
            torch.from_numpy(all_features[:self.seq_len]).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(y_ret).float(),
        )


class QuantizeEncoder(dummyLightning):
    # TODO: reshape before embed
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.n_buckets * config.num_features, config.embed_dim)
        self.quantiles = torch.from_numpy(np.load(self.mmap_dir + 'quantiles.npy')).float()

    def forward(self, x: torch.Tensor):
        # x: (batch, num_features)
        # quantiles: (num_quantiles, num_features)

        # Use interior quantile boundaries for bucketization
        quantiles_tensor = self.quantiles[1:, :].to(x.device,
                                                    dtype=x.dtype)

        tokens = torch.stack([
            torch.bucketize(x[:, i].contiguous(),
                            quantiles_tensor[:, i].contiguous(), right=True)
            for i in range(x.shape[1])
        ], dim=1)  # (batch, num_features)
        tokens = tokens.clamp(0, self.n_buckets - 1)

        # Offset each feature's tokens to its own embedding range
        offsets = torch.arange(x.shape[1], device=x.device) * self.n_buckets
        tokens = tokens + offsets  # (batch, num_features)

        return self.embedding(tokens)


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
            x: (batch, cent_feats) - delta price features (delta_1min)

        Returns:
            (batch, embed_dim) - projected embeddings
        """
        x = (x * 100).long().clamp(-self.max_cent_abs-1, self.max_cent_abs+1)
        x = x + self.max_cent_abs + 1

        embedded = self.embedding(x)

        batch_size = embedded.shape[0]
        flattened = embedded.reshape(batch_size, -1)
        return self.proj(flattened)


class SinEncoder(dummyLightning):
    # TODO: values maybe very small. should adapt
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
        self.proj = nn.Linear(config.sin_dim * config.num_features, config.embed_dim)

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
            self.quantize_encoder(x_flat, self.encoder_quantiles),
            self.cent_encoder(x_flat[:, self.cent_feat_idx]),
            self.sin_encoder(x_flat),
        ]
        output = self.combiner(torch.cat(embeddings, dim=-1))

        return output.unflatten(0, (batch_size, seq_len))


class MultiReadout(dummyLightning):
    def __init__(self, config):
        super().__init__(config)

        self.cent_head = nn.Linear(config.hidden_dim,
                                   config.num_cents * self.num_horizons)
        self.variance_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.quantile_head = nn.Linear(config.hidden_dim,
                                       config.num_quantiles*self.num_horizons)

        self.return_variance_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.return_quantile_head = nn.Linear(config.hidden_dim,
                                              config.num_quantiles*self.num_horizons)

        # TODO: MuP
        nn.init.trunc_normal_(self.variance_head.weight, std=0.02, a=-0.04, b=0.08)
        nn.init.trunc_normal_(self.return_variance_head.weight, std=0.02, a=-0.04, b=0.08)

    def forward(self, x: torch.Tensor, target_type='mean'):
        """
        x: (batch, seq_len, hidden_dim)
        returns: (batch, seq_len, num_horizons, ...)
                 predictions for each horizon
        """
        batch_size, seq_len, _ = x.shape
        x = x.float()

        if target_type == 'cent':
            out = self.cent_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_cents)
        elif target_type == 'var':
            return F.softplus(self.variance_head(x))
        elif target_type == 'quantile':
            out = self.quantile_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_quantiles)
        elif target_type == 'return_var':
            return F.softplus(self.return_variance_head(x))
        elif target_type == 'return_quantile':
            out = self.return_quantile_head(x)
            return out.view(batch_size, seq_len, self.num_horizons,
                            self.num_quantiles)


class FinalPipeline(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self._init_distributed()
        self.prepare_data()
        self.prepare_model()
    
    def prepare_model(self):
        self.encoder = MultiEncoder(self.config)
        self.backbone = TM(self.config)
        self.readout = MultiReadout(self.config)
        # Prediction quantiles as fractions (config has percentiles like 10, 25, 50, 75, 90)
        self.register_buffer('pred_quantiles',
                             torch.tensor(self.config.quantiles, dtype=torch.float32) / 100.0)

        for name in self.loss_names:
            self.register_buffer(f'loss_rms_{name}', torch.tensor(1.0))

    def _check_db_ready(self) -> bool:
        """Check if DuckDB database exists and has required tables"""
        if not Path(self.db_path).exists():
            return False

        con = duckdb.connect(self.db_path, read_only=True)
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

    def _check_mmap_ready(self, require_quantile=False) -> bool:
        required_files = (
            'train_data.npy', 'train_stock_ids.npy', 'train_cumsums.npy', 'train_stock_offsets.npy',
            'val_data.npy', 'val_stock_ids.npy', 'val_cumsums.npy', 'val_stock_offsets.npy',
        )
        if require_quantile:
            required_files += ('quantiles.npy',)
        return all(Path(self.mmap_dir + f).exists() for f in required_files)

    def prepare_data(self):
        if self.is_root():
            con = self._create_db()
            if not self._check_mmap_ready():
                print("  Creating memory-mapped arrays...")
                self._create_mmap_arrays(con)
            
            print("  Compute and store quantiles from train_data...")
            self._compute_and_store_quantiles(con, 1000000)
            con.close()
        else:  # wait
            import time
            max_wait = 600
            wait_interval = 5
            elapsed = 0
            while not self._check_mmap_ready(require_quantile=True) and elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
            assert self._check_mmap_ready(require_quantile=True), "MMAP arrays not ready after waiting"
        self.train_dataset = PriceHistoryDataset(self.config, 'train')
        self.val_dataset = PriceHistoryDataset(self.config, 'val')

    def _create_mmap_arrays(self, con: duckdb.DuckDBPyConnection):
        db_features_cols = ', '.join(self.db_features)
        for split in ['train', 'val']:
            index = con.execute(f"""
                SELECT stock_id, cumsum FROM {split}_index ORDER BY cumsum
            """).fetchall()
            stock_ids = np.array([r[0] for r in index], dtype=np.int64)
            cumsums = np.array([r[1] for r in index], dtype=np.int64)

            stock_lengths = con.execute(f"""
                SELECT stock_id, COUNT(*) as cnt
                FROM {split}_data
                GROUP BY stock_id
                ORDER BY stock_id
            """).fetchall()
            stock_len_map = {r[0]: r[1] for r in stock_lengths}

            stock_offsets = np.zeros(len(stock_ids), dtype=np.int64)
            offset = 0
            for i, sid in enumerate(stock_ids):
                stock_offsets[i] = offset
                offset += stock_len_map.get(int(sid), 0)
            total_rows = offset

            data = np.zeros((total_rows, len(self.db_features)), dtype=np.float32)
            for i, stock_id in enumerate(stock_ids):
                stock_id = int(stock_id)
                result = con.execute(f"""
                    SELECT {db_features_cols}
                    FROM {split}_data
                    WHERE stock_id = ?
                    ORDER BY row_idx
                """, [stock_id]).fetchnumpy()

                n_rows = len(result[self.db_features[0]])
                if n_rows == 0:
                    continue

                start = int(stock_offsets[i])
                for col_idx, col in enumerate(self.db_features):
                    data[start:start + n_rows, col_idx] = result[col]

                if i % 500 == 0:
                    print(f"      Processed {i}/{len(stock_ids)} stocks")

            np.save(self.mmap_dir + f'{split}_data.npy', data)
            np.save(self.mmap_dir + f'{split}_stock_ids.npy', stock_ids)
            np.save(self.mmap_dir + f'{split}_cumsums.npy', cumsums)
            np.save(self.mmap_dir + f'{split}_stock_offsets.npy', stock_offsets)
            print(f"    Saved {split} arrays: {total_rows} rows")

    def _create_db(self):
        # Build directly into db_path (in /home/jkp/ssd) - persists across runs
        con = duckdb.connect(self.db_path)
        con.execute(f"SET memory_limit='{self.ram}GB'")

        con.execute(f"""
            CREATE VIEW IF NOT EXISTS raw_data AS
            SELECT * FROM read_parquet('{self.pq_path}')
        """)

        cutoff_cache_path = Path('/home/jkp/ssd') / 'pipeline_cutoff.txt'
        if cutoff_cache_path.exists():
            cutoff = int(float(cutoff_cache_path.read_text().strip()))
        else:
            print("  Computing train/val cutoff (90th percentile of datetime)...")
            cutoff = con.execute("""
                SELECT QUANTILE_CONT(epoch_ns(datetime), 0.9) as cutoff
                FROM (SELECT datetime FROM raw_data USING SAMPLE 1000000)
            """).fetchone()[0]
            cutoff_cache_path.write_text(str(cutoff))
        print(f"  Train/val cutoff timestamp: {cutoff}")

        con.execute("""
            CREATE VIEW IF NOT EXISTS df AS
            SELECT *,
                -- Returns at different time scales (30min, 1day, 2days)
                LEAD(close, 30) OVER (PARTITION BY id ORDER BY datetime) / close - 1 AS ret_30min,
                LEAD(close, 240) OVER (PARTITION BY id ORDER BY datetime) / close - 1 AS ret_1day,
                LEAD(close, 480) OVER (PARTITION BY id ORDER BY datetime) / close - 1 AS ret_2day
            FROM raw_data
        """)

        print("  Create data tables...")
        exists = con.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND (name='train_data' OR name='val_data')
        """).fetchall()
        if len(exists) != 2:
            assert len(exists) < 2
            con.execute("""
                CREATE TEMP TABLE df_materialized AS
                SELECT * FROM df
            """)

            def build1(split='train'):
                print(f"  Building {split}_data...")
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {split}_data AS
                    SELECT
                        CAST(SUBSTR(d.id, 1, 6) AS INTEGER) AS stock_id,
                        ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.datetime) - 1 AS row_idx,
                        CAST(d.datetime AS DATE) AS date,
                        {',\n  '.join(f'd.{f}' for f in self.db_features)}
                    FROM df_materialized d
                    WHERE epoch_ns(d.datetime) {'<=' if split == 'train' else '>'} {cutoff}
                """)
            build1('train')
            build1('val')

        print("  Creating indexes...")
        con.execute("CREATE INDEX IF NOT EXISTS train_data_idx ON train_data(stock_id, row_idx)")
        con.execute("CREATE INDEX IF NOT EXISTS val_data_idx ON val_data(stock_id, row_idx)")

        print("  Build index tables...")
        seq_len = self.seq_len
        max_horizon = max(self.horizons)
        
        def build2(split='train'):
            print(f"  Building {split}_index...")
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {split}_index AS
                WITH stock_lengths AS (
                    SELECT stock_id, MAX(row_idx) + 1 AS stock_len
                    FROM {split}_data
                    GROUP BY stock_id
                ),
                valid_stocks AS (
                    SELECT
                        stock_id,
                        stock_len - {seq_len} - {max_horizon} AS valid_samples
                    FROM stock_lengths
                )
                SELECT
                    stock_id,
                    SUM(valid_samples) OVER (ORDER BY stock_id) AS cumsum
                FROM valid_stocks
                WHERE valid_samples > 0
            """)
        build2('train')
        build2('val')
        return con
    
    def _compute_and_store_quantiles(self, con: duckdb.DuckDBPyConnection, sample_size: int = 1000000):
        """Compute 256 quantiles by iterating through Dataset, deduplicating high-frequency modes."""
        filename = self.mmap_dir + 'quantiles.npy'
        if Path(filename).exists():
            return

        print(f"    Constructing Dataset and sampling {sample_size} items...")
        dataset = PriceHistoryDataset(self.config, 'train')

        # Sample indices
        n_total = len(dataset)
        indices = np.random.choice(n_total, size=min(sample_size, n_total), replace=False)

        all_features = []
        for i, idx in enumerate(indices):
            x, _, _ = dataset[idx]  # x: (seq_len, num_features)
            all_features.append(x.numpy())
            if i % 10000 == 0:
                print(f"      Processed {i}/{len(indices)} samples")

        all_features = np.concatenate(all_features, axis=0)  # (samples * seq_len, num_features)
        print(f"    Collected {all_features.shape[0]} feature vectors")

        # Compute quantiles with deduplication via binary search on k
        print("    Computing quantiles on GPU with binary search for distinct values...")
        n_buckets = self.n_buckets
        quantiles = np.zeros((n_buckets, len(self.features)), dtype=np.float32)

        all_features_t = torch.from_numpy(all_features)

        for feat_idx, feat in enumerate(self.features):
            values, _ = all_features_t[:, feat_idx].contiguous().sort()
            n_vals = values.shape[0]

            def compute_quantiles_k(k: int) -> torch.Tensor:
                indices = torch.linspace(0, n_vals - 1, k, device=values.device).long()
                return values[indices]

            # Binary search for k such that unique(quantiles(k)) == 256
            lo, hi = n_buckets, min(n_vals, n_buckets * 8)

            # First check if we can achieve 256 unique at hi
            q_hi = compute_quantiles_k(hi)
            assert len(torch.unique(q_hi)) >= n_buckets

            flag = False
            while lo < hi:
                mid = (lo + hi) // 2
                q_mid = compute_quantiles_k(mid)
                unique_mid = len(torch.unique(q_mid))
                if unique_mid == n_buckets:
                    quantiles[:, feat_idx] = torch.unique(q_mid).cpu().numpy()
                    flag = True
                    break
                if unique_mid < n_buckets:
                    lo = mid + 1
                else:
                    hi = mid - 1
            assert flag
        for feat_idx in range(len(self.features)):
            for i in range(1, n_buckets):
                assert quantiles[i, feat_idx] >= quantiles[i-1, feat_idx]

        print("    Storing quantiles in DuckDB...")
        con.execute(
            "CREATE TABLE quantiles (feature_idx INTEGER, q_idx INTEGER, value FLOAT)"
        )
        for feat_idx in range(len(self.features)):
            for q_idx in range(n_buckets):
                con.execute(
                    "INSERT INTO quantiles VALUES (?, ?, ?)",
                    [feat_idx, q_idx, float(quantiles[q_idx, feat_idx])]
                )
        
        print("    Creating quantiles npy...")
        np.save(filename, quantiles)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        features = self.backbone(encoded)
        return features

    def step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        x, y, y_returns = batch
        losses = {}

        features = self.forward(x)[:, self.seq_len//2:, :]

        losses['quantile'] = 0.0
        pred_quantiles = self.readout(features, 'quantile')
        y = y.to(pred_quantiles.dtype)
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = self._sided_loss(pred_quantiles[:, :, h_idx, :], y[:, :, h_idx])
            losses[f'quantile_{h_name}'] = h_loss
            losses['quantile'] += h_loss / len(self.horizons)

        losses['nll'] = 0.0
        pred_mean = pred_quantiles[:, :, :, self.num_quantiles//2]
        pred_var = self.readout(features, 'var')
        pred_var += 1e-8
        for h_idx, h_name in enumerate(self.horizons):
            nll = 1/2 * (torch.log(2 * torch.pi * pred_var[:, :, h_idx]) +
                         (y[:, :, h_idx] - pred_mean[:, :, h_idx]) ** 2 / pred_var[:, :, h_idx])
            nll = nll.clamp(max=20.0)
            h_loss = nll.mean()
            losses[f'nll_{h_name}'] = h_loss
            losses['nll'] += h_loss / len(self.horizons)

        y_returns = y_returns.to(pred_mean.dtype)

        losses['return_quantile'] = 0.0
        pred_return_quantiles = self.readout(features, 'return_quantile')
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = self._sided_loss(pred_return_quantiles[:, :, h_idx, :], y_returns[:, :, h_idx])
            losses[f'return_quantile_{h_name}'] = h_loss
            losses['return_quantile'] += h_loss / len(self.horizons)
        
        ret_mean = pred_return_quantiles[:, :, :, self.num_quantiles//2]
        ret_var = self.readout(features, 'return_var') + 1e-8

        losses['return_nll'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            nll = (
                torch.log(2*torch.pi * ret_var[:, :, h_idx]) +
                (y_returns[:, :, h_idx] - ret_mean[:, :, h_idx]) ** 2 / ret_var[:, :, h_idx]
            ) / 2
            nll = nll.clamp(max=20.0)
            h_loss = nll.mean()
            losses[f'return_nll_{h_name}'] = h_loss
            losses['return_nll'] += h_loss / len(self.horizons)

        with torch.no_grad():  # Adaptive loss scaling with RMS estimates
            loss_values = {name: losses[name].detach() for name in self.loss_names}

            # Update EMA of RMS for each loss
            for name in self.loss_names:
                rms_buffer = getattr(self, f'loss_rms_{name}')
                current_rms = torch.sqrt(loss_values[name] ** 2 + 1e-8)  # TODO: this is so not RMS. 
                new_rms = self.loss_ema * rms_buffer + (1 - self.loss_ema) * current_rms  # EMA update
                rms_buffer.copy_(new_rms)

            # Compute geometric mean of all RMS values for normalization target
            rms_values = torch.stack([getattr(self, f'loss_rms_{name}') for name in self.loss_names])
            target_rms = torch.exp(torch.log(rms_values + 1e-8).mean())

        # Compute scale factors to normalize each loss to target magnitude
        scale_factors = {}
        for name in self.loss_names:
            rms = getattr(self, f'loss_rms_{name}')
            scale_factors[name] = (target_rms / (rms + 1e-8)).clamp(0.1, 10.0)

        total_loss = (
            scale_factors['nll'] * losses['nll'] +
            scale_factors['quantile'] * losses['quantile'] +
            scale_factors['return_nll'] * losses['return_nll'] +
            scale_factors['return_quantile'] * losses['return_quantile']
        ) / 4

        self.log('price_prediction/nll', losses['nll'])
        self.log('price_prediction/quantile', losses['quantile'])
        self.log('return_prediction/nll', losses['return_nll'])
        self.log('return_prediction/quantile', losses['return_quantile'])

        # Log adaptive scale factors
        for name in self.loss_names:
            self.log(f'loss_scale/{name}', scale_factors[name])

        for h_name in self.horizons:
            self.log(f'price/h/{h_name}/nll', losses[f'nll_{h_name}'])
            self.log(f'price/h/{h_name}/sided', losses[f'quantile_{h_name}'])
            self.log(f'ret/h/{h_name}/nll', losses[f'return_nll_{h_name}'])
            self.log(f'ret/h/{h_name}/sided', losses[f'return_quantile_{h_name}'])

        result = {
            'loss': total_loss,
            'nll_loss': losses['nll'],
            'quantile_loss': losses['quantile'],
            'return_nll_loss': losses['return_nll'],
            'return_quantile_loss': losses['return_quantile']
        }

        for h_name in self.horizons:
            result[f'nll_{h_name}'] = losses[f'nll_{h_name}']
            result[f'quantile_{h_name}'] = losses[f'quantile_{h_name}']
            result[f'return_nll_{h_name}'] = losses[f'return_nll_{h_name}']
            result[f'return_quantile_{h_name}'] = losses[f'return_quantile_{h_name}']

        if Path('/tmp/breakpoint').exists():
            breakpoint()

        return result

    def _sided_loss(self, pred, target):
        """ pred: (b, l, num_quantiles)
          target: (b, l)
        """
        quantiles = self.pred_quantiles.to(dtype=pred.dtype)  # (num_quantiles,)
        quantiles = quantiles.view(1, 1, -1).expand_as(pred)  # (b, l, num_quantiles)
        target = target.unsqueeze(-1)  # (b, l, 1)

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


def parse_args_to_config(base_config):
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
        else:
            parser.set_defaults(**{field_name: default_val})

    # Filter out module-like arguments from wrappers like profilers.oom_debug_hook
    filtered_argv = [
        arg for arg in sys.argv[1:]
        if not arg.count('.') >= 2
    ]
    args = parser.parse_args(filtered_argv)

    overrides = {}
    for field in fields(FinalPipelineConfig):
        field_name = field.name
        tp = field.type
        arg_val = getattr(args, field_name)

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


@dataclass
class FinalPipelineConfig(dummyConfig):
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
    shrink_model: bool = False

    # Training
    batch_size: int | None = None
    lr: float = 3e-4
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    num_workers: int | None = None
    loss_names: tuple = ('nll', 'quantile', 'return_nll', 'return_quantile')
    loss_ema: float = 0.99

    # Data
    seq_len: int = 4096
    n_buckets: int = 256
    max_cent_abs: int = 64
    db_path: str = '/home/jkp/ssd/pipeline.duckdb'
    db_features: tuple = ('open', 'high', 'low', 'close',
                          'ret_30min', 'ret_1day', 'ret_2day',
                          'volume')
    features: tuple = ('close_norm', 'delta_1min',
                       'ret_1min', 'ret_30min', 'ret_1day', 'ret_2day',
                       'close_open', 'high_open', 'low_open', 'high_low',
                       'volume_norm')
    cent_feats: tuple = ('delta_1min',)
    horizons: tuple = (1, 30, 240, 480)
    quantiles: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)
    mmap_dir: str = '/home/jkp/ssd/pipeline_mmap/'
    db_path: str = '/home/jkp/ssd/pipeline.duckdb'
    pq_path: str = '/home/jkp/ssd/a_1min.pq'

    debug_data: int | None = None
    no_compile: bool = False
    
    # ddp
    world_size: int = 1
    ddp_backend: str = 'nccl'
    master_addr: str = 'localhost'
    port: str = '12355'

    # Hardware
    device: str = 'cuda'
    vram: int = 80
    ram: int = 368

    def __post_init__(self):
        # Trunk
        if self.shrink_model:
            self.hidden_dim //= 4
            self.num_heads //= 2
            self.num_layers //= 2
            self.embed_dim //= 2
        self.interim_dim = int(self.hidden_dim * self.expand)
        self.head_dim = int(self.hidden_dim * self.attn / self.num_heads)

        # Embedding
        self.q_dim = self.sin_dim = self.embed_dim // 2
        self.num_cents = self.max_cent_abs * 2 + 1
        self.num_features = len(self.features)
        self.num_cent_feats = len(self.cent_feats)
        self.cent_feat_idx = [i for i, f in enumerate(self.features) if f in self.cent_feats]

        # Prediction
        self.num_horizons = len(self.horizons)
        self.num_quantiles = len(self.quantiles)
        self.pred_len = self.seq_len // 2

        self.batch_size = self.vram * 2 ** (21 if self.no_compile else 23) // self.seq_len // self.num_layers // self.interim_dim if self.batch_size is None else self.batch_size
        self.num_workers = min(os.cpu_count(), self.batch_size // 16) if self.num_workers is None else self.num_workers
        print(f'batch size: {self.batch_size}, num_workers: {self.num_workers}')

        super().__post_init__()


if __name__ == "__main__":
    config = parse_args_to_config(FinalPipelineConfig(
        shrink_model=True,
    ))
    p = FinalPipeline(config)
    p.fit()
