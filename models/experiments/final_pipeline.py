"""
Final pipeline implementation combining all components from main.md files

This pipeline implements:
1. Multiple encoding strategies:
   - Quantize: 1/256 percentile-based token encoding
   - Cent quantize: cent-level delta encoding for small values
   - CNN: Image encoding over K-line graphs (from draw.py)
   - Sin encoding: Sinusoidal encoding for continuous values

2. Multiple preprocessing approaches:
   - Raw values (close, open, high, low, volume)
   - Per-stock normalized values (using training set statistics)
   - Returns at varied time scales (1min, 30min, 6hr, 1day, 2day)
   - Intra-minute relative features (close/open, high/open, etc.)
   - Time-normalized values (cross-stock normalized per timestep)

3. Transformer architecture:
   - Uses F.scaled_dot_product_attention (modern best practice)
   - SiLU activation throughout all layers
   - Multi-head self-attention with residual connections

4. Multiple prediction types and encodings:
   - Quantized predictions (cross-entropy loss)
   - Cent-based predictions
   - Mean predictions (Huber loss)
   - Mean + Variance predictions (NLL loss)
   - Quantile predictions (10th, 25th, 50th, 75th, 90th with sided-weighted loss)

5. Multiple prediction horizons:
   - 1min, 30min, 1day, 2day ahead
   - Next day OHLC predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset

from ..prelude.model import dummyLightning
from .embedding.draw import create_kline_pixel_graph
from ..prelude.model import Rope

torch.autograd.set_detect_anomaly(True)


class EfficientTimeSeriesDataset(Dataset):
    def __init__(self, stock_data: Dict[str, np.ndarray],
                 stock_targets: Dict[str, np.ndarray],
                 stock_events: Dict[str, np.ndarray],
                 seq_len: int,
                 pred_len: int):
        """
        Args:
            stock_data: Dict mapping stock_id to full feature array (T, F)
            stock_targets: Dict mapping stock_id to full target array (T,)
            stock_events: Dict mapping stock_id to event markers (T, 3)
                         [lunch, dinner, skipped_day]
            seq_len: Sequence length for context
            pred_len: Prediction length
        """
        self.stock_data = stock_data
        self.stock_targets = stock_targets
        self.stock_events = stock_events
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Prediction horizons in minutes: 1min, 30min, 1day (390min), 2day (780min)
        self.horizons = [1, 30, 240, 480]
        self.num_horizons = len(self.horizons)

        # Build index: list of (stock_id, start_idx) tuples
        # Ensure we have enough future data for the longest horizon (480 min)
        self.index = []
        for stock_id, data in stock_data.items():
            max_horizon = max(self.horizons)
            # Need seq_len + max_horizon to ensure we can get targets for all horizons
            num_windows = len(data) - seq_len - max_horizon
            if num_windows > 0:
                for start_idx in range(num_windows):
                    self.index.append((stock_id, start_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        stock_id, start_idx = self.index[idx]

        # Slice
        end_idx = start_idx + self.seq_len
        features = self.stock_data[stock_id][start_idx:end_idx]

        # Get prediction positions' targets (latter half)
        # For each position in the latter half, get targets at all horizons
        target_start = start_idx + self.seq_len // 2

        # targets shape: (pred_len, num_horizons)
        targets = np.zeros((self.pred_len, self.num_horizons), dtype=np.float32)

        for pos_idx in range(self.pred_len):
            current_pos = target_start + pos_idx
            for h_idx, horizon in enumerate(self.horizons):
                target_pos = current_pos + horizon
                # Bounds check
                if target_pos < len(self.stock_targets[stock_id]):
                    targets[pos_idx, h_idx] = self.stock_targets[stock_id][target_pos]
                else:
                    targets[pos_idx, h_idx] = np.nan

        # Get event markers
        events = self.stock_events[stock_id][start_idx:end_idx]

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(targets).float(),
            torch.from_numpy(events).float()
        )


class TransformerBlock(dummyLightning):
    """Transformer block with self-attention using F.scaled_dot_product_attention"""

    def __init__(self, config):
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        self.rope = Rope(config)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, config.intermediate_dim),
            nn.SiLU(),
            nn.Linear(config.intermediate_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) or (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        residual = x
        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if not (hasattr(self.config, 'use_normal_rope') and self.config.use_normal_rope):
            q = self.rope(q)
            k = self.rope(k)

        # Rearrange for attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(self.config, 'use_normal_rope') and self.config.use_normal_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0 if not self.training else 0.1)

        # Rearrange back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class PercentileEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.quant_bins + 3, config.embed_dim)
        # Add 3 special tokens: lunch, dinner, skipped_day
        self.lunch_token = config.quant_bins
        self.dinner_token = config.quant_bins + 1
        self.skipped_token = config.quant_bins + 2

    def forward(self, x: torch.Tensor, quantiles: np.ndarray, events: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, 1) - single feature values
        # events: (batch, 3) - binary flags for [lunch, dinner, skipped_day]
        tokens = torch.from_numpy(np.digitize(x.cpu().numpy(), quantiles[1:-1],
                                              right=True)).to(x.device)
        # tokens shape: (batch, 1), squeeze to (batch,) for embedding
        tokens = tokens.squeeze(-1)

        # Override with special tokens if events present
        if events is not None:
            lunch_mask = events[:, 0].bool()
            dinner_mask = events[:, 1].bool()
            skipped_mask = events[:, 2].bool()

            # Priority: skipped_day > lunch > dinner (no dinner when day skipped)
            tokens = torch.where(dinner_mask & ~skipped_mask,
                                torch.full_like(tokens, self.dinner_token), tokens)
            tokens = torch.where(lunch_mask & ~skipped_mask,
                                torch.full_like(tokens, self.lunch_token), tokens)
            tokens = torch.where(skipped_mask,
                                torch.full_like(tokens, self.skipped_token), tokens)

        result = self.embedding(tokens)
        return result


class CentQuantizeEncoder(dummyLightning):
    def __init__(self, config):
        super().__init__(config)
        self.max_cents = config.max_cents
        # -max_cents to +max_cents plus special tokens for inf/-inf and 3 event tokens
        self.embedding = nn.Embedding(2 * config.max_cents + 5, config.embed_dim)
        self.lunch_token = 2 * config.max_cents + 2
        self.dinner_token = 2 * config.max_cents + 3
        self.skipped_token = 2 * config.max_cents + 4

    def forward(self, x: torch.Tensor, events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert cent differences to tokens"""
        x = x.squeeze(-1)
        tokens = torch.clamp(x, -self.max_cents-1,
                             self.max_cents+1
                             ) + self.max_cents + 1
        tokens = torch.where(torch.isnan(x), torch.zeros_like(tokens), tokens)

        # Override with special tokens if events present
        if events is not None:
            lunch_mask = events[:, 0].bool()
            dinner_mask = events[:, 1].bool()
            skipped_mask = events[:, 2].bool()

            # Priority: skipped_day > lunch > dinner (no dinner when day skipped)
            tokens = torch.where(dinner_mask & ~skipped_mask,
                                torch.full_like(tokens, self.dinner_token), tokens)
            tokens = torch.where(lunch_mask & ~skipped_mask,
                                torch.full_like(tokens, self.lunch_token), tokens)
            tokens = torch.where(skipped_mask,
                                torch.full_like(tokens, self.skipped_token), tokens)

        return self.embedding(tokens)


class CNNEncoder(dummyLightning):
    """CNN encoder for K-line graph images"""

    def __init__(self, config, embed_dim: int = 128):
        super().__init__(config)
        # Input: 180x96 RGB images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 90x48
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 45x24
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 22x12
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4x128
        )
        self.fc = nn.Linear(4 * 4 * 128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 96, 180) - RGB images"""
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class SinEncoder(nn.Module):
    """Sinusoidal encoding for continuous values"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.max_freq = config.max_freq

        # Create frequency embeddings
        freqs = torch.exp(torch.linspace(0, np.log(self.max_freq), self.embed_dim // 2))
        self.register_buffer('freqs', freqs)

        # Special token embeddings
        self.special_embeddings = nn.Embedding(3, config.embed_dim)  # lunch, dinner, skipped_day

    def forward(self, x: torch.Tensor, events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, features), events: (batch, 3)"""
        # Take the mean across feature dimension for sinusoidal encoding
        x_mean = x.mean(dim=-1, keepdim=True)  # (batch, 1)

        # Apply sinusoidal encoding
        sin_emb = torch.sin(x_mean * self.freqs.unsqueeze(0))
        cos_emb = torch.cos(x_mean * self.freqs.unsqueeze(0))

        # Interleave sin and cos
        emb = torch.stack([sin_emb, cos_emb], dim=-1).flatten(start_dim=-2)
        emb = emb[..., :self.embed_dim]

        # Override with special tokens if events present
        if events is not None:
            lunch_mask = events[:, 0].bool()
            dinner_mask = events[:, 1].bool()
            skipped_mask = events[:, 2].bool()

            # Create token indices (0=lunch, 1=dinner, 2=skipped_day)
            has_event = lunch_mask | dinner_mask | skipped_mask
            token_idx = torch.zeros(events.shape[0], dtype=torch.long, device=events.device)

            # Priority: skipped_day > lunch > dinner (no dinner when day skipped)
            token_idx = torch.where(dinner_mask & ~skipped_mask, torch.ones_like(token_idx), token_idx)
            token_idx = torch.where(lunch_mask & ~skipped_mask, torch.zeros_like(token_idx), token_idx)
            token_idx = torch.where(skipped_mask, torch.full_like(token_idx, 2), token_idx)

            special_emb = self.special_embeddings(token_idx)
            emb = torch.where(has_event.unsqueeze(-1), special_emb, emb)

        return emb


class MultiEncoder(dummyLightning):
    """Combines multiple encoding strategies"""

    def __init__(self, config, quantiles: Optional[np.ndarray] = None):
        super().__init__(config)
        self.config = config
        self.quantiles = quantiles

        # Individual encoders
        self.quantize_encoder = PercentileEncoder(config)
        self.cent_encoder = CentQuantizeEncoder(config)
        self.cnn_encoder = CNNEncoder(config)
        self.sin_encoder = SinEncoder(config)

        # Combine different encodings
        total_dim = config.hidden_dim * 3  # 4 encoding types
        self.combiner = nn.Linear(total_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor, events: Optional[torch.Tensor] = None, image_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        events: (batch, seq_len, 3) - binary flags for [lunch, dinner, skipped_day]
        """
        batch_size, seq_len, feat_dim = x.shape

        # Flatten for processing
        x_flat = x.view(-1, feat_dim)
        events_flat = events.view(-1, 3) if events is not None else None

        # Apply different encodings to different features
        embeddings = []

        # Quantize encoding (use first feature)
        # TODO: use more features
        quant_emb = self.quantize_encoder(x_flat[:, 0:1], self.quantiles,
                                          events_flat)
        embeddings.append(quant_emb)

        # Cent quantization (use return features)
        cent_emb = self.cent_encoder((x_flat[:, 2:3] * 100).long(),  # to cents
                                     events_flat)
        embeddings.append(cent_emb)

        # Sin encoding (use all features)
        sin_emb = self.sin_encoder(x_flat, events_flat)
        embeddings.append(sin_emb)

        # CNN encoding (if image data provided)
        # if image_data is not None:
        #     cnn_emb = self.cnn_encoder(image_data)
        #     # Repeat CNN embedding across sequence length
        #     cnn_emb = cnn_emb.unsqueeze(1).repeat(1, seq_len, 1)
        #     embeddings.append(cnn_emb.view(-1, self.config.hidden_dim))

        # Combine embeddings - all should have shape (batch_size * seq_len,
        #                                             hidden_dim)
        combined = torch.cat(embeddings, dim=-1)
        output = self.combiner(combined)

        # Reshape back to sequence format
        return output.view(batch_size, seq_len, -1)


class MultiReadout(dummyLightning):
    """Multiple readout strategies for different prediction types with multi-horizon support"""

    def __init__(self, config):
        super().__init__(config)

        # Prediction horizons: 1min, 30min, 1day (390min), 2day (780min)
        self.num_horizons = 4

        # Different readout heads - each predicts for all horizons
        self.quantized_head = nn.Linear(config.hidden_dim, config.quant_bins * self.num_horizons)
        self.cent_head = nn.Linear(config.hidden_dim, 129 * self.num_horizons)  # -64 to +64 plus special tokens
        self.mean_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.variance_head = nn.Linear(config.hidden_dim, self.num_horizons)
        self.quantile_head = nn.Linear(config.hidden_dim, 5 * self.num_horizons)  # 10th, 25th, 50th, 75th, 90th

    def forward(self, x: torch.Tensor, target_type: str = 'mean') -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        returns: (batch, seq_len, num_horizons, ...) - predictions for each horizon
        """
        batch_size, seq_len, _ = x.shape

        if target_type == 'quantized':
            out = self.quantized_head(x)  # (batch, seq_len, quant_bins * num_horizons)
            return out.view(batch_size, seq_len, self.num_horizons, -1)  # (batch, seq_len, num_horizons, quant_bins)
        elif target_type == 'cent':
            out = self.cent_head(x)
            return out.view(batch_size, seq_len, self.num_horizons, -1)
        elif target_type == 'mean':
            return self.mean_head(x).unsqueeze(-1)  # (batch, seq_len, num_horizons, 1)
        elif target_type == 'mean_var':
            mean = self.mean_head(x)  # (batch, seq_len, num_horizons)
            var = F.softplus(self.variance_head(x))  # (batch, seq_len, num_horizons)
            return torch.stack([mean, var], dim=-1)  # (batch, seq_len, num_horizons, 2)
        elif target_type == 'quantile':
            out = self.quantile_head(x)  # (batch, seq_len, 5 * num_horizons)
            return out.view(batch_size, seq_len, self.num_horizons, 5)  # (batch, seq_len, num_horizons, 5)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class FinalPipeline(dummyLightning):
    """Complete pipeline combining all components"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = MultiEncoder(config)

        self.backbone = self._build_backbone()

        # Multi-readout for different prediction types
        self.readout = MultiReadout(config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()

    def _build_backbone(self) -> dummyLightning:
        layers = []

        for _ in range(self.config.num_layers):
            layers.append(TransformerBlock(self.config))

        return nn.Sequential(*layers)

    def prepare_data(self, path: Optional[str] = None, seq_len: int = 64,
                     quant_bins: int = 256, train_frac: float = 0.9):
        """Prepare data with all preprocessing strategies"""

        if path is None:
            path = Path.home() / 'h' / 'data' / 'a_1min.pq'
        path = Path(path)

        if self.config.debug_data:
            df = pl.scan_parquet(str(path))
            ids = df.select('id').unique().collect().sample(5)['id'].to_list()
            df = df.filter(pl.col('id').is_in(ids)).collect()
        else:
            df = pl.read_parquet(str(path))
        df = df.sort(['id', 'datetime'])

        df = self._compute_features(df)
        df = self._detect_trading_gaps(df)  # Add gap detection
        df = self._split_data(df, train_frac)
        self.quantiles = self._compute_quantiles(df, quant_bins)

        self.encoder.quantiles = self.quantiles['close']

        # Build sequences
        self.train_dataset, self.val_dataset = self._build_sequences(df, seq_len)

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
        unique_datetimes = df['datetime'].unique().sort().to_numpy()
        cutoff = np.quantile(unique_datetimes, train_frac)

        df = df.with_columns((pl.col('datetime') <= cutoff).alias('is_train'))

        stats = df.filter(pl.col('is_train')).group_by('id').agg([
            pl.col('close').mean().alias('mean_close'),
            pl.col('close').std().alias('std_close'),
            pl.col('volume').mean().alias('mean_volume'),
            pl.col('volume').std().alias('std_volume')
        ])

        df = df.join(stats, on='id')

        # normalized features
        df = df.with_columns([
            ((pl.col('close') - pl.col('mean_close')) / (pl.col('std_close') + 1e-6)).alias('close_norm'),
            ((pl.col('volume') - pl.col('mean_volume')) / (pl.col('std_volume') + 1e-6)).alias('volume_norm')
        ])

        return df

    def _compute_quantiles(self, df: pl.DataFrame, quant_bins: int) -> Dict[str, np.ndarray]:
        """Compute quantiles for different features"""

        train_df = df.filter(pl.col('is_train'))

        quantiles = {}
        for col in ['close', 'ret_1min', 'ret_30min', 'volume']:
            data = train_df[col].to_numpy()
            quantiles[col] = np.quantile(data, np.linspace(0, 1, quant_bins + 1))

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

        # Initialize event columns
        df = df.with_columns([
            pl.lit(0).cast(pl.Int8).alias('is_lunch'),
            pl.lit(0).cast(pl.Int8).alias('is_dinner'),
            pl.lit(0).cast(pl.Int8).alias('is_skipped_day')
        ])

        # Classify gaps:
        # Lunch: 30 min - 2 hours (1800 - 7200 seconds)
        # Dinner: 12 - 20 hours (43200 - 72000 seconds) - overnight between trading days
        # Skipped day: > 20 hours (72000 seconds) - weekends, holidays
        df = df.with_columns([
            ((pl.col('time_gap_seconds') >= 1800) &
             (pl.col('time_gap_seconds') < 7200)).cast(pl.Int8).alias('is_lunch'),

            ((pl.col('time_gap_seconds') >= 43200) &
             (pl.col('time_gap_seconds') < 72000)).cast(pl.Int8).alias('is_dinner'),

            (pl.col('time_gap_seconds') >= 72000).cast(pl.Int8).alias('is_skipped_day')
        ])

        # Fill nulls (first row of each stock) with 0
        df = df.with_columns([
            pl.col('is_lunch').fill_null(0),
            pl.col('is_dinner').fill_null(0),
            pl.col('is_skipped_day').fill_null(0)
        ])

        return df

    def _build_sequences(self, df: pl.DataFrame, seq_len: int) -> Tuple[Dataset, Dataset]:
        """
        Build efficient sequences that store full data per stock and slice in __getitem__.

        For a sequence of length seq_len:
        - Positions 0 to seq_len//2-1: context only, no prediction
        - Positions seq_len//2 to seq_len-1: each predicts the next timestep
        """

        feature_cols = [
            'close', 'close_norm', 'ret_1min', 'ret_30min',
            'ret_1min_ratio', 'ret_30min_ratio', 'ret_1day_ratio', 'ret_2day_ratio',
            'close_open_ratio', 'high_open_ratio', 'low_open_ratio', 'high_low_ratio',
            'volume', 'volume_norm'
        ]

        event_cols = ['is_lunch', 'is_dinner', 'is_skipped_day']

        # Compute time-normalized values (cross-stock normalized per timestep)
        time_stats = df.group_by('datetime').agg([
            pl.col('close').mean().alias('time_mean_close'),
            pl.col('close').std().alias('time_std_close')
        ])

        df = df.join(time_stats, on='datetime')
        df = df.with_columns([
            ((pl.col('close') - pl.col('time_mean_close')) / (pl.col('time_std_close') + 1e-6)).alias('close_time_norm')
        ])

        pred_len = seq_len // 2  # Latter half of sequence

        # Store data per stock to avoid duplication
        train_stock_data = {}
        train_stock_targets = {}
        train_stock_events = {}

        val_stock_data = {}
        val_stock_targets = {}
        val_stock_events = {}

        # TODO: align for cross attention
        for stock_id in df['id'].unique():
            stock_df = df.filter(pl.col('id') == stock_id).sort('datetime')

            if len(stock_df) <= seq_len + 1:
                continue

            features = stock_df.select(feature_cols).to_numpy().astype(np.float32)
            events = stock_df.select(event_cols).to_numpy().astype(np.float32)
            is_train = stock_df['is_train'].to_numpy()
            close = stock_df['close'].to_numpy().astype(np.float32)

            # Split each stock's data by time - train data and val data
            train_mask = is_train.astype(bool)
            val_mask = ~train_mask

            # Only add to train set if we have enough train data
            if train_mask.sum() > seq_len + 1:
                train_stock_data[stock_id] = features[train_mask]
                train_stock_targets[stock_id] = close[train_mask]
                train_stock_events[stock_id] = events[train_mask]

            # Only add to val set if we have enough val data
            if val_mask.sum() > seq_len + 1:
                val_stock_data[stock_id] = features[val_mask]
                val_stock_targets[stock_id] = close[val_mask]
                val_stock_events[stock_id] = events[val_mask]

        train_dataset = EfficientTimeSeriesDataset(
            train_stock_data, train_stock_targets, train_stock_events,
            seq_len, pred_len
        )
        val_dataset = EfficientTimeSeriesDataset(
            val_stock_data, val_stock_targets, val_stock_events,
            seq_len, pred_len
        )

        return train_dataset, val_dataset

    def forward(self, x: torch.Tensor, events: Optional[torch.Tensor] = None,
                target_type: str = 'mean') -> torch.Tensor:
        """Forward pass through the pipeline"""

        encoded = self.encoder(x, events)
        features = self.backbone(encoded)
        return features

    def step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        x, y, events = batch
        # x: (batch, seq_len, features)
        # y: (batch, pred_len) where pred_len = seq_len // 2
        # events: (batch, seq_len, 3)

        seq_len = x.shape[1]
        losses = {}

        features = self.forward(x, events)
        # Get predictions for all sequence positions
        # But we only care about predictions from positions seq_len//2 onwards
        pred_quantized = self.readout(features, 'quantized')  # (batch, seq_len, quant_bins)
        pred_quantized = pred_quantized[:, seq_len//2:, :]  # (batch, pred_len, quant_bins)

        y = y.to(pred_quantized.dtype)

        # Quantized prediction loss
        quantiles_tensor = torch.from_numpy(self.quantiles['close']).to(y.device, dtype=y.dtype)
        target_quantized = torch.bucketize(y, quantiles_tensor)  # (batch, pred_len)
        target_quantized = torch.clamp(target_quantized, 0, self.config.quant_bins - 1)
        losses['quantized'] = self.ce_loss(
            pred_quantized.reshape(-1, self.config.quant_bins),
            target_quantized.reshape(-1)
        )

        # Mean prediction loss (Huber)
        pred_mean = self.readout(features, 'mean')  # (batch, seq_len, 1)
        pred_mean = pred_mean[:, seq_len//2:, :]  # (batch, pred_len, 1)
        losses['mean'] = self.huber_loss(pred_mean.squeeze(-1), y)

        # Mean + Variance prediction loss (NLL)
        pred_mean_var = self.readout(features, 'mean_var')  # (batch, seq_len, 2)
        pred_mean_var = pred_mean_var[:, seq_len//2:, :]  # (batch, pred_len, 2)
        pred_mean_nll = pred_mean_var[..., 0]  # (batch, pred_len)
        pred_var = pred_mean_var[..., 1] + 1e-6  # (batch, pred_len)
        losses['nll'] = 0.5 * (torch.log(pred_var) + (y - pred_mean_nll) ** 2 / pred_var).mean()

        # Quantile prediction loss
        pred_quantiles = self.readout(features, 'quantile')  # (batch, seq_len, 5)
        pred_quantiles = pred_quantiles[:, seq_len//2:, :]  # (batch, pred_len, 5)
        quantile_targets = self._compute_quantile_targets(y)  # (batch, pred_len, 5)
        losses['quantile'] = self._quantile_loss(pred_quantiles, quantile_targets)

        assert (
            losses['quantized'] >= 0 and
            losses['mean'] >= 0 and
            losses['nll'] >= 0 and
            losses['quantile'] >= 0
        ), losses
        # Combine losses
        total_loss = (
            0.25 * losses['quantized'] +
            0.25 * losses['mean'] +
            0.25 * losses['nll'] +
            0.25 * losses['quantile']
        )

        # Check for NaN values and report which component is problematic
        if torch.isnan(total_loss):
            for name, loss_val in losses.items():
                if torch.isnan(loss_val):
                    raise ValueError(f"NaN detected in {name} loss: {loss_val.item()}")
            raise ValueError(f"NaN in total loss but not in components: {total_loss.item()}")

        return {
            'loss': total_loss,
            'quantized_loss': losses['quantized'],
            'mean_loss': losses['mean'],
            'nll_loss': losses['nll'],
            'quantile_loss': losses['quantile']
        }

    def _compute_quantile_targets(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile targets for training.

        y: (batch, pred_len) - targets for multiple positions
        returns: (batch, pred_len, 5) - quantile targets
        """
        # Compute empirical quantiles from training data (flatten all values)
        # Convert to float32 for numpy compatibility
        y_np = y.detach().cpu().float().numpy().flatten()
        quantiles = np.quantile(y_np, [0.1, 0.25, 0.5, 0.75, 0.9])

        # Create targets - expand to (batch, pred_len, 5)
        batch_size, pred_len = y.shape
        targets = torch.zeros(batch_size, pred_len, 5, device=y.device, dtype=y.dtype)

        for i, q in enumerate(quantiles):
            targets[..., i] = torch.where(y >= q, torch.ones_like(y), torch.zeros_like(y))

        return targets

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quantile loss with sided weighting"""

        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)

        losses = []
        for i, q in enumerate(quantiles):
            # Fix indexing: pred and target are (batch, pred_len, 5)
            # We want the i-th quantile dimension, not the i-th sequence position
            error = target[:, :, i] - pred[:, :, i]

            # Sided weighting
            weight = torch.where(error > 0,
                                 torch.full_like(error, q),
                                 torch.full_like(error, 1 - q))

            losses.append((weight * torch.abs(error)).mean())

        return torch.stack(losses).mean()


@dataclass
class FinalPipelineConfig:
    """Configuration for the final pipeline"""

    # Model architecture
    hidden_dim: int = 256
    intermediate_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    quant_bins: int = 256
    max_freq: float = 10000.

    # Training
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Data
    seq_len: int = 256
    train_ratio: float = 0.9
    max_cents: int = 64
    embed_dim: int = 256

    # Device
    device: str = 'cuda'
    num_workers: int = 4

    # Muon optimizer for 2D parameters
    use_muon: bool = True

    debug_data: bool = False


if __name__ == "__main__":
    config = FinalPipelineConfig(
        debug_data=True,
    )
    pipeline = FinalPipeline(config)
    pipeline.prepare_data()
    pipeline.fit()
    breakpoint()