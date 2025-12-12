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
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from ..prelude.model import dummyLightning, dummyConfig, TM


class PriceHistoryDataset(Dataset):
    """Dataset backed by in-memory DuckDB, materialized on access."""

    def __init__(self, config, split: str, con: duckdb.DuckDBPyConnection):
        self.split = split
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.horizons = config.horizons
        self.num_horizons = len(self.horizons)
        self.features = config.features
        self.db_features = config.db_features
        self.close_norm_idx = config.feat_idx['close_norm']
        self.con = con

        # Load just the index
        result = con.execute(f"""
            SELECT stock_id, cumsum FROM {split}_index ORDER BY cumsum
        """).fetchall()
        self.stock_ids = np.array([r[0] for r in result], dtype=np.int64)
        self.cumsums = np.array([r[1] for r in result], dtype=np.int64)
        self.total_samples = int(self.cumsums[-1]) if len(self.cumsums) > 0 else 0

        # Load stats for normalization (stock_id -> (mean_close, std_close, mean_vol, std_vol))
        stats_result = con.execute("""
            SELECT CAST(SUBSTR(id, 1, 6) AS INTEGER) as stock_id,
                   mean_close, std_close, mean_volume, std_volume
            FROM stats
        """).fetchall()
        self.stats = {r[0]: (r[1], r[2], r[3], r[4]) for r in stats_result}

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search to find which stock this index belongs to
        stock_pos = np.searchsorted(self.cumsums, idx, side='right')
        stock_id = int(self.stock_ids[stock_pos])
        # Compute local offset within this stock
        prev_cumsum = int(self.cumsums[stock_pos - 1]) if stock_pos > 0 else 0
        start_idx = idx - prev_cumsum

        # Query the slice we need from DuckDB
        max_horizon = max(self.horizons)
        target_start = start_idx + self.seq_len // 2
        target_end = target_start + self.pred_len + max_horizon
        end_idx = max(start_idx + self.seq_len, target_end)

        db_features_cols = ', '.join(self.db_features)
        data = self.con.execute(f"""
            SELECT {db_features_cols}
            FROM {self.split}_data
            WHERE stock_id = ? AND row_idx >= ? AND row_idx < ?
            ORDER BY row_idx
        """, [stock_id, start_idx, end_idx]).fetchnumpy()

        # Get stats for normalization
        mean_close, std_close, mean_vol, std_vol = self.stats.get(
            stock_id, (0.0, 1.0, 0.0, 1.0))

        # Build all features array
        n_rows = len(data['close'])
        all_features = np.zeros((n_rows, len(self.features)), dtype=np.float32)

        # Map db_features to their positions
        db_feat_map = {f: i for i, f in enumerate(self.db_features)}

        for feat_idx, feat in enumerate(self.features):
            if feat in db_feat_map:
                all_features[:, feat_idx] = data[feat]
            elif feat == 'close_norm':
                all_features[:, feat_idx] = (data['close'] - mean_close) / (std_close + 1e-8)
            elif feat == 'volume_norm':
                all_features[:, feat_idx] = (data['volume'] - mean_vol) / (std_vol + 1e-8)
            elif feat == 'delta_1min':
                # close[t+1] - close[t], pad last with 0
                close = data['close']
                all_features[:-1, feat_idx] = close[1:] - close[:-1]
            elif feat == 'ret_1min':
                # close[t+1] / close[t] - 1, pad last with 0
                close = data['close']
                all_features[:-1, feat_idx] = close[1:] / (close[:-1] + 1e-8) - 1
            elif feat == 'close_open':
                all_features[:, feat_idx] = data['close'] / (data['open'] + 1e-8)
            elif feat == 'high_open':
                all_features[:, feat_idx] = data['high'] / (data['open'] + 1e-8)
            elif feat == 'low_open':
                all_features[:, feat_idx] = data['low'] / (data['open'] + 1e-8)
            elif feat == 'high_low':
                all_features[:, feat_idx] = data['high'] / (data['low'] + 1e-8)

        features = all_features[:self.seq_len]
        close_norm = all_features[self.seq_len // 2:, self.close_norm_idx]

        # Build targets and return_targets
        targets = np.zeros((self.pred_len, self.num_horizons), dtype=np.float32)
        return_targets = np.zeros((self.pred_len, self.num_horizons), dtype=np.float32)
        current_prices = close_norm[:self.pred_len]

        for h_idx, horizon in enumerate(self.horizons):
            future_prices = close_norm[horizon:horizon + self.pred_len]
            targets[:, h_idx] = future_prices
            return_targets[:, h_idx] = future_prices / (current_prices + 1e-8)

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(targets).float(),
            torch.from_numpy(return_targets).float(),
        )


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

        # TODO: will this be overridden?
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
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MultiEncoder(config)
        self.backbone = TM(config)
        self.readout = MultiReadout(config)
        # Prediction quantiles as fractions (config has percentiles like 10, 25, 50, 75, 90)
        self.register_buffer('pred_quantiles',
                             torch.tensor(config.quantiles, dtype=torch.float32) / 100.0)

    def _get_db_path(self) -> Path:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return Path(self.db_path)

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

    def prepare_data(self, path: Optional[str] = None):
        """Prepare data using on-disk DuckDB storage."""
        db_path = self._get_db_path()

        if self._check_db_ready():
            if self.is_root():
                print(f"Loading from existing DuckDB: {db_path}")
            con = duckdb.connect(str(db_path), read_only=True)
            self.encoder_quantiles = self._load_quantiles(con)
            self.encoder.encoder_quantiles = self.encoder_quantiles
            self.train_dataset = PriceHistoryDataset(self.config, 'train', con)
            self.val_dataset = PriceHistoryDataset(self.config, 'val', con)
            return

        # Need to build DB from scratch
        if self.is_root():
            print("Building DuckDB from scratch...")

        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        else:
            path = Path(path)

        if self.is_root():
            con = self.create_db(path, db_path)
            self.encoder_quantiles = self._load_quantiles(con)
            self.encoder.encoder_quantiles = self.encoder_quantiles
            self.train_dataset = PriceHistoryDataset(self.config, 'train', con)
            self.val_dataset = PriceHistoryDataset(self.config, 'val', con)
        else:
            # Non-root processes will need to wait for disk persistence
            import time
            max_wait = 600  # 10 minutes max
            wait_interval = 5
            elapsed = 0
            while not self._check_db_ready() and elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
            if not self._check_db_ready():
                raise RuntimeError(f"Database not ready after {max_wait}s")
            con = duckdb.connect(str(db_path), read_only=True)
            self.encoder_quantiles = self._load_quantiles(con)
            self.encoder.encoder_quantiles = self.encoder_quantiles
            self.train_dataset = PriceHistoryDataset(self.config, 'train', con)
            self.val_dataset = PriceHistoryDataset(self.config, 'val', con)

    def _load_quantiles(self, con: duckdb.DuckDBPyConnection) -> torch.Tensor:
        """Load quantiles from DuckDB connection."""
        result = con.execute("""
            SELECT feature_idx, q_idx, value FROM quantiles ORDER BY feature_idx, q_idx
        """).fetchnumpy()
        n_features = result['feature_idx'].max() + 1
        n_quantiles = result['q_idx'].max() + 1
        quantiles = np.zeros((n_quantiles, n_features), dtype=np.float32)
        quantiles[result['q_idx'], result['feature_idx']] = result['value']
        return torch.from_numpy(quantiles).float()

    def create_db(self, parquet_path: Path, db_path: Path):
        # Build directly into db_path (in /home/jkp/ssd) - persists across runs
        con = duckdb.connect(str(db_path))
        con.execute(f"SET memory_limit='{self.ram}GB'")

        con.execute(f"""
            CREATE VIEW IF NOT EXISTS raw_data AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

        # Only cache the cutoff value (tiny file)
        cutoff_cache_path = Path('/home/jkp/ssd') / 'pipeline_cutoff.txt'

        if self.no_cache or not cutoff_cache_path.exists():
            print("  Computing train/val cutoff (90th percentile of datetime)...")
            cutoff = con.execute("""
                SELECT QUANTILE_CONT(epoch_ns(datetime), 0.9) as cutoff
                FROM (SELECT datetime FROM raw_data USING SAMPLE 1000000)
            """).fetchone()[0]
            if not self.no_cache:
                cutoff_cache_path.write_text(str(cutoff))
        else:
            cutoff = int(float(cutoff_cache_path.read_text().strip()))
        print(f"  Train/val cutoff timestamp: {cutoff}")

        con.execute("""
            CREATE VIEW IF NOT EXISTS df AS
            SELECT *,
                -- Returns at different time scales (30min, 1day, 2days) - expensive window functions
                COALESCE(LEAD(close, 30) OVER (PARTITION BY id ORDER BY datetime) - close, 0) AS delta_30min,
                COALESCE(LEAD(close, 30) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_30min,
                COALESCE(LEAD(close, 240) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_1day,
                COALESCE(LEAD(close, 480) OVER (PARTITION BY id ORDER BY datetime) / close - 1, 0) AS ret_2day
            FROM raw_data
        """)

        print("  Computing per-stock normalization stats...")
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS stats AS
            SELECT
                id,
                AVG(close) AS mean_close,
                STDDEV(close) AS std_close,
                AVG(volume) AS mean_volume,
                STDDEV(volume) AS std_volume
            FROM raw_data
            WHERE epoch_ns(datetime) <= {cutoff}
            GROUP BY id
        """)

        stock_ids = [row[0] for row in con.execute("SELECT id FROM stats").fetchall()]
        print(f"  Num stocks: {len(stock_ids)}")

        print("Step 6: Create data tables...")

        select_cols = ',\n  '.join(f'd.{f}' for f in self.db_features)

        print("  Materializing features (window functions)...")
        train_exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='train_data'").fetchone()
        val_exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='val_data'").fetchone()
        if not (train_exists and val_exists):
            con.execute("""
                CREATE TABLE IF NOT EXISTS df_materialized AS
                SELECT * FROM df
            """)

            def build1(split='train'):
                print(f"  Building {split}_data...")
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {split}_data AS
                    SELECT
                        CAST(SUBSTR(d.id, 1, 6) AS INTEGER) AS stock_id,
                        ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.datetime) - 1 AS row_idx,
                        {select_cols}
                    FROM df_materialized d
                    WHERE epoch_ns(d.datetime) {'<=' if split == 'train' else '>'} {cutoff}
                """)
            build1('train')
            build1('val')

            # Free up memory
            con.execute("DROP TABLE df_materialized")

        print("Step 7: Creating indexes...")
        con.execute("CREATE INDEX IF NOT EXISTS train_data_idx ON train_data(stock_id, row_idx)")
        con.execute("CREATE INDEX IF NOT EXISTS val_data_idx ON val_data(stock_id, row_idx)")

        print("Step 8: Compute and store quantiles from train_data...")
        sample_size = 10000 if self.debug_data else 1000000
        self._compute_and_store_quantiles(con, sample_size)

        print("Step 9: Build index tables for dataset iteration...")
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

        print("Database build complete in /home/jkp/ssd")
        return con

    def _compute_and_store_quantiles(self, con: duckdb.DuckDBPyConnection, sample_size: int):
        quantiles_exists = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quantiles'").fetchone()
        if quantiles_exists:
            return
        n_quantize = self.n_quantize
        q_positions = [i / n_quantize for i in range(n_quantize + 1)]
        db_features_cols = ', '.join(self.db_features)

        sampled = con.execute(f"""
            SELECT {db_features_cols}
            FROM train_data
            USING SAMPLE {sample_size}
        """).fetchnumpy()

        avg_mean_close, avg_std_close, avg_mean_vol, avg_std_vol = con.execute("""
            SELECT AVG(mean_close), AVG(std_close), AVG(mean_volume), AVG(std_volume)
            FROM stats
        """).fetchone()

        # Compute derived features for quantile estimation
        close = sampled['close']
        open_price = sampled['open']
        high = sampled['high']
        low = sampled['low']
        volume = sampled['volume']

        derived = {
            'close_norm': (close - avg_mean_close) / (avg_std_close + 1e-8),
            'volume_norm': (volume - avg_mean_vol) / (avg_std_vol + 1e-8),
            'delta_1min': np.diff(close, prepend=close[0]),
            'ret_1min': np.diff(close, prepend=close[0]) / (np.concatenate([[close[0]], close[:-1]]) + 1e-8),
            'close_open': close / (open_price + 1e-8),
            'high_open': high / (open_price + 1e-8),
            'low_open': low / (open_price + 1e-8),
            'high_low': high / (low + 1e-8),
        }

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
            if col in sampled:
                col_data = np.sort(sampled[col])
            else:
                col_data = np.sort(derived[col])
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
        quantiles_tensor = self.encoder_quantiles[:, 0].to(y_returns_typed.device, dtype=y_returns_typed.dtype)
        target_quantized = torch.bucketize(y_returns_typed, quantiles_tensor)  # (batch, pred_len, num_horizons)
        target_quantized = torch.clamp(target_quantized, 0, self.n_quantize - 1)

        losses['quantized'] = 0.0
        for h_idx, h_name in enumerate(self.horizons):
            h_loss = F.cross_entropy(
                pred_quantized[:, :, h_idx, :].reshape(-1, self.n_quantize),
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
    """Parse CLI arguments and override config fields"""
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

    # Training
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

    # Features stored in DB (raw OHLCV + expensive window functions)
    db_features: tuple = ('open', 'high', 'low', 'close',
                          'delta_30min', 'ret_30min', 'ret_1day', 'ret_2day',
                          'volume')
    # All features (including cheap ones computed in getitem)
    features: tuple = ('open', 'high', 'low', 'close', 'close_norm',
                       'delta_1min', 'delta_30min', 'ret_1min', 'ret_30min',
                       'ret_1day', 'ret_2day', 'close_open', 'high_open',
                       'low_open', 'high_low', 'volume', 'volume_norm')
    cent_feats: tuple = ('delta_1min', 'delta_30min')
    horizons: tuple = (1, 30, 240, 480)
    quantiles: tuple = (10, 25, 50, 75, 90)

    debug_data: int | None = None
    debug_model: bool = False
    no_cache: bool = False  # Skip cache files, force recalc in RAM

    # DDP settings
    world_size: int = 1
    ddp_backend: str = 'nccl'
    master_addr: str = 'localhost'
    master_port: str = '12355'

    debug_ddp: bool = False
    num_workers: int | None = None
    shrink_model: bool = False

    # Hardware
    device: str = 'cuda'
    vram: int = 80
    ram: int = 368

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

        # Features - named index for clean access
        self.feat_idx = {name: i for i, name in enumerate(self.features)}
        self.num_features = len(self.features)
        self.num_quantize_features = self.num_features
        self.num_cent_feats = len(self.cent_feats)
        self.cent_feat_idx = [self.feat_idx[f] for f in self.cent_feats]

        self.num_horizons = len(self.horizons)
        self.num_quantiles = len(self.quantiles)
        self.pred_len = self.seq_len // 2

        self.batch_size = self.vram * 2 ** 21 // self.seq_len // self.num_layers // self.interim_dim

        # DuckDB - use /home/jkp/ssd for faster access and to avoid OOM
        if self.db_path is None:
            db_name = f'pipeline_debug_{self.debug_data}.duckdb' if self.debug_data else 'pipeline.duckdb'
            self.db_path = str(Path('/home/jkp/ssd') / db_name)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        super().__post_init__()


config = FinalPipelineConfig(
    shrink_model=True,
    debug_model=True,
)

if __name__ == "__main__":
    config = parse_args_to_config(config)
    p = FinalPipeline(config)

    p._init_distributed()
    if p.is_root():
        print(f"{p.world_size} GPUs")

    if p.is_root():
        print("Preparing data...")
    p.prepare_data()
    assert len(p.train_dataset)
    assert len(p.val_dataset)

    if p.is_root():
        print("Starting training...")
    p.fit()
