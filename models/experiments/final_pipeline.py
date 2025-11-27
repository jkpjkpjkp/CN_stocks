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

6. Portfolio optimization:
   - 2-sided softmax for stock weights (can be positive or negative)
   - Trading costs: 0.5% on long, 30% on short
   - GRPO (Group Relative Policy Optimization) rollouts for portfolio training

Usage:
    config = FinalPipelineConfig()
    pipeline = create_final_pipeline(config)
    pipeline.prepare_data()
    pipeline.fit()  # Standard training
    # or use pipeline.grpo_step() for GRPO training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset

from ..prelude.model import dummyLightning, transformerConfig
from .embedding.draw import create_kline_pixel_graph
from .embedding.main import TimeSeriesDataset
from ..prelude.model import Rope


class TransformerBlock(nn.Module):
    """Transformer block with self-attention using F.scaled_dot_product_attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config
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

    @torch.compile
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


class PercentileEncoder(nn.Module):
    def __init__(self, quant_bins: int = 256, embed_dim: int = 128):
        super().__init__()
        self.quant_bins = quant_bins
        self.embedding = nn.Embedding(quant_bins, embed_dim)

    def forward(self, x: torch.Tensor, quantiles: np.ndarray) -> torch.Tensor:
        """Convert continuous values to quantized tokens"""
        # x: (batch, 1) - single feature values
        print(f"PercentileEncoder input shape: {x.shape}")
        tokens = torch.from_numpy(np.digitize(x.cpu().numpy(), quantiles[1:-1],
                                              right=True)).to(x.device)
        print(f"PercentileEncoder tokens shape before squeeze: {tokens.shape}")
        # tokens shape: (batch, 1), squeeze to (batch,) for embedding
        tokens = tokens.squeeze(-1)
        print(f"PercentileEncoder tokens shape after squeeze: {tokens.shape}")
        result = self.embedding(tokens)
        print(f"PercentileEncoder output shape: {result.shape}")
        return result


class CentQuantizeEncoder(nn.Module):
    """Cent-based quantization for small price differences"""

    def __init__(self, embed_dim: int = 128, max_cents: int = 64):
        super().__init__()
        self.max_cents = max_cents
        # -max_cents to +max_cents plus special tokens for inf/-inf
        self.embedding = nn.Embedding(2 * max_cents + 3, embed_dim)  # +3 for -inf, +inf, and 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert cent differences to tokens"""
        x = x.squeeze(-1)
        tokens = torch.clamp(x.round().long(), -self.max_cents, self.max_cents) + self.max_cents + 1
        # Handle infinities
        tokens = torch.where(torch.isinf(x), torch.sign(x).long() * (self.max_cents + 1) + self.max_cents + 1, tokens)
        tokens = torch.where(torch.isnan(x), torch.zeros_like(tokens), tokens)
        return self.embedding(tokens)


class CNNEncoder(nn.Module):
    """CNN encoder for K-line graph images"""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
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

    def __init__(self, embed_dim: int = 128, max_freq: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq

        # Create frequency embeddings
        freqs = torch.exp(torch.linspace(0, np.log(max_freq), embed_dim // 2))
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, features)"""
        # Take the mean across feature dimension for sinusoidal encoding
        x_mean = x.mean(dim=-1, keepdim=True)  # (batch, 1)

        # Apply sinusoidal encoding
        sin_emb = torch.sin(x_mean * self.freqs.unsqueeze(0))
        cos_emb = torch.cos(x_mean * self.freqs.unsqueeze(0))

        # Interleave sin and cos
        emb = torch.stack([sin_emb, cos_emb], dim=-1).flatten(start_dim=-2)
        return emb[..., :self.embed_dim]


class MultiEncoder(nn.Module):
    """Combines multiple encoding strategies"""

    def __init__(self, config, quantiles: Optional[np.ndarray] = None):
        super().__init__()
        self.config = config
        self.quantiles = quantiles

        # Individual encoders
        self.quantize_encoder = PercentileEncoder(config.quant_bins, config.hidden_dim)
        self.cent_encoder = CentQuantizeEncoder(config.hidden_dim, max_cents=64)
        self.cnn_encoder = CNNEncoder(config.hidden_dim)
        self.sin_encoder = SinEncoder(config.hidden_dim)

        # Combine different encodings
        total_dim = config.hidden_dim * 3  # 4 encoding types
        self.combiner = nn.Linear(total_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor, image_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, features)"""
        batch_size, seq_len, feat_dim = x.shape

        # Flatten for processing
        x_flat = x.view(-1, feat_dim)

        # Apply different encodings to different features
        embeddings = []

        # Quantize encoding (use first feature)
        if self.quantiles is not None:
            quant_emb = self.quantize_encoder(x_flat[:, 0:1], self.quantiles)
            embeddings.append(quant_emb)

        # Cent quantization (use return features)
        if feat_dim > 2:
            cent_emb = self.cent_encoder(x_flat[:, 2:3] * 100)  # Convert to cents
            embeddings.append(cent_emb)

        # Sin encoding (use all features)
        sin_emb = self.sin_encoder(x_flat)
        embeddings.append(sin_emb)

        # CNN encoding (if image data provided)
        if image_data is not None:
            cnn_emb = self.cnn_encoder(image_data)
            # Repeat CNN embedding across sequence length
            cnn_emb = cnn_emb.unsqueeze(1).repeat(1, seq_len, 1)
            embeddings.append(cnn_emb.view(-1, self.config.hidden_dim))

        # Combine embeddings - all should have shape (batch_size * seq_len, hidden_dim)
        combined = torch.cat(embeddings, dim=-1)
        output = self.combiner(combined)

        # Reshape back to sequence format
        return output.view(batch_size, seq_len, -1)


class MultiReadout(nn.Module):
    """Multiple readout strategies for different prediction types"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Different readout heads
        self.quantized_head = nn.Linear(config.hidden_dim, config.quant_bins)
        self.cent_head = nn.Linear(config.hidden_dim, 129)  # -64 to +64 plus special tokens
        self.mean_head = nn.Linear(config.hidden_dim, 1)
        self.variance_head = nn.Linear(config.hidden_dim, 1)
        self.quantile_head = nn.Linear(config.hidden_dim, 5)  # 10th, 25th, 50th, 75th, 90th

    def forward(self, x: torch.Tensor, target_type: str = 'mean') -> torch.Tensor:
        """x: (batch, seq_len, hidden_dim)"""
        if target_type == 'quantized':
            return self.quantized_head(x)
        elif target_type == 'cent':
            return self.cent_head(x)
        elif target_type == 'mean':
            return self.mean_head(x)
        elif target_type == 'mean_var':
            mean = self.mean_head(x)
            var = F.softplus(self.variance_head(x))
            return torch.cat([mean, var], dim=-1)
        elif target_type == 'quantile':
            return self.quantile_head(x)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")


class PortfolioOptimizer(nn.Module):
    """TODO: take into account last holding and only cost delta"""
    """Portfolio optimization with GRPO rollouts"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trading_cost_long = 0.005  # 0.5%
        self.trading_cost_short = 0.30  # 30%

        # Predict stock weights using 2-sided softmax
        self.weight_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2)  # Positive and negative weights
        )

    def forward(self, predictions: torch.Tensor, current_weights: torch.Tensor) -> torch.Tensor:
        """
        predictions: (batch, num_stocks, hidden_dim)
        current_weights: (batch, num_stocks)
        """
        # Predict raw weights
        raw_weights = self.weight_predictor(predictions)  # (batch, num_stocks, 2)

        # Apply 2-sided softmax
        pos_weights = F.softmax(raw_weights[..., 0], dim=-1)
        neg_weights = -F.softmax(raw_weights[..., 1], dim=-1)
        new_weights = pos_weights + neg_weights

        # Calculate trading costs
        weight_changes = torch.abs(new_weights - current_weights)
        long_trades = torch.clamp(weight_changes, min=0)
        short_trades = torch.clamp(-weight_changes, max=0)

        trading_costs = self.trading_cost_long * long_trades.sum() + self.trading_cost_short * torch.abs(short_trades).sum()

        return new_weights, trading_costs


class FinalPipeline(dummyLightning):
    """Complete pipeline combining all components"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = MultiEncoder(config)

        self.backbone = self._build_backbone()

        # Multi-readout for different prediction types
        self.readout = MultiReadout(config)

        # Portfolio optimizer
        self.portfolio_optimizer = PortfolioOptimizer(config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()

    def _build_backbone(self) -> nn.Module:
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
            df = pl.scan_parquet(str(path)).head(1000000).collect()
        else:
            df = pl.read_parquet(str(path))
        df = df.sort(['id', 'datetime'])

        df = self._compute_features(df)
        df = self._split_data(df, train_frac)
        self.quantiles = self._compute_quantiles(df, quant_bins)

        self.encoder.quantiles = self.quantiles['close']

        # Build sequences
        self.train_dataset, self.val_dataset = self._build_sequences(df, seq_len)

    def _compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute all feature types"""

        # Returns at different time scales (1min, 30min, 6hr, 24hr, 2days)
        # Note: Assuming 390 minutes per trading day (6.5 hours)
        df = df.with_columns([
            (pl.col('close') - pl.col('close').shift(1).over('id')).alias('ret_1min'),
            (pl.col('close') - pl.col('close').shift(30).over('id')).alias('ret_30min'),
            (pl.col('close') - pl.col('close').shift(360).over('id')).alias('ret_6hr'),  # 6 hours
            (pl.col('close') - pl.col('close').shift(390).over('id')).alias('ret_1day'),  # 1 day
            (pl.col('close') - pl.col('close').shift(780).over('id')).alias('ret_2day'),  # 2 days
            (pl.col('close') / pl.col('close').shift(1).over('id') - 1).alias('ret_1min_ratio'),
            (pl.col('close') / pl.col('close').shift(30).over('id') - 1).alias('ret_30min_ratio'),
            (pl.col('close') / pl.col('close').shift(390).over('id') - 1).alias('ret_1day_ratio'),
        ])

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
            pl.col('ret_6hr').fill_null(0.),
            pl.col('ret_1day').fill_null(0.),
            pl.col('ret_2day').fill_null(0.),
            pl.col('ret_1min_ratio').fill_null(0.),
            pl.col('ret_30min_ratio').fill_null(0.),
            pl.col('ret_1day_ratio').fill_null(0.),
            pl.col('close_open_ratio').fill_null(1.),
            pl.col('high_open_ratio').fill_null(1.),
            pl.col('low_open_ratio').fill_null(1.),
            pl.col('high_low_ratio').fill_null(1.)
        ])

        return df

    def _split_data(self, df: pl.DataFrame, train_frac: float) -> pl.DataFrame:
        """Split data by time to avoid leakage"""

        # Compute a single cutoff date from all unique datetimes
        unique_datetimes = df['datetime'].unique().sort().to_numpy()
        cutoff = np.quantile(unique_datetimes, train_frac)

        df = df.with_columns((pl.col('datetime') <= cutoff).alias('is_train'))

        # Compute per-stock statistics on training data
        stats = df.filter(pl.col('is_train')).group_by('id').agg([
            pl.col('close').mean().alias('mean_close'),
            pl.col('close').std().alias('std_close'),
            pl.col('volume').mean().alias('mean_volume'),
            pl.col('volume').std().alias('std_volume')
        ])

        df = df.join(stats, on='id')

        # Add normalized features
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

    def _build_sequences(self, df: pl.DataFrame, seq_len: int) -> Tuple[Dataset, Dataset]:
        """Build sequences for training and validation with multiple prediction horizons"""

        train_sequences = []
        val_sequences = []
        train_targets = []
        val_targets = []

        feature_cols = [
            'close', 'close_norm', 'ret_1min', 'ret_30min', 'ret_6hr', 'ret_1day', 'ret_2day',
            'ret_1min_ratio', 'ret_30min_ratio', 'ret_1day_ratio', 'close_open_ratio',
            'high_open_ratio', 'low_open_ratio', 'high_low_ratio',
            'volume', 'volume_norm'
        ]

        # Compute time-normalized values (cross-stock normalized per timestep)
        # Group by minute and compute mean/std across all stocks
        time_stats = df.group_by('datetime').agg([
            pl.col('close').mean().alias('time_mean_close'),
            pl.col('close').std().alias('time_std_close')
        ])

        df = df.join(time_stats, on='datetime')
        df = df.with_columns([
            ((pl.col('close') - pl.col('time_mean_close')) / (pl.col('time_std_close') + 1e-6)).alias('close_time_norm')
        ])

        for stock_id in df['id'].unique():
            stock_df = df.filter(pl.col('id') == stock_id).sort('datetime')

            # Need enough data for longest prediction horizon (2 days = 780 minutes)
            if len(stock_df) <= seq_len + 780:
                continue

            features = stock_df.select(feature_cols).to_numpy()
            is_train = stock_df['is_train'].to_numpy()
            close = stock_df['close'].to_numpy()
            close_time_norm = stock_df['close_time_norm'].to_numpy()
            open = stock_df['open'].to_numpy()
            high = stock_df['high'].to_numpy()
            low = stock_df['low'].to_numpy()

            # Create sliding windows with multiple target horizons
            for i in range(len(features) - seq_len - 780):
                seq = features[i:i+seq_len]

                # Multiple target types:
                # 1. Next close (1min ahead)
                # 2. 30min ahead close
                # 3. 1day ahead close
                # 4. Next day OHLC (390 minutes ahead)
                # 5. Time-normalized next close
                target_dict = {
                    'close_1min': close[i+seq_len],
                    'close_30min': close[i+seq_len+30] if i+seq_len+30 < len(close) else close[-1],
                    'close_1day': close[i+seq_len+390] if i+seq_len+390 < len(close) else close[-1],
                    'close_2day': close[i+seq_len+780] if i+seq_len+780 < len(close) else close[-1],
                    'open_next': open[i+seq_len+390] if i+seq_len+390 < len(open) else open[-1],
                    'high_next': high[i+seq_len+390] if i+seq_len+390 < len(high) else high[-1],
                    'low_next': low[i+seq_len+390] if i+seq_len+390 < len(low) else low[-1],
                    'close_time_norm': close_time_norm[i+seq_len]
                }

                # For now, use close_1min as primary target (we'll use others in ensemble)
                target = target_dict['close_1min']

                if is_train[i+seq_len]:
                    train_sequences.append(seq)
                    train_targets.append(target)
                else:
                    val_sequences.append(seq)
                    val_targets.append(target)

        # Convert to numpy arrays
        X_train = np.stack(train_sequences) if train_sequences else np.empty((0, seq_len, len(feature_cols)))
        y_train = np.array(train_targets) if train_targets else np.empty(0)
        X_val = np.stack(val_sequences) if val_sequences else np.empty((0, seq_len, len(feature_cols)))
        y_val = np.array(val_targets) if val_targets else np.empty(0)

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train) if len(X_train) > 0 else None
        val_dataset = TimeSeriesDataset(X_val, y_val) if len(X_val) > 0 else None

        return train_dataset, val_dataset

    def forward(self, x: torch.Tensor, target_type: str = 'mean') -> torch.Tensor:
        """Forward pass through the pipeline"""

        # Encode input
        encoded = self.encoder(x)

        # Apply transformer backbone
        features = self.backbone(encoded)

        # Apply readout
        predictions = self.readout(features, target_type)

        return predictions

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step with multiple loss components"""
        x, y = batch

        # Generate predictions with different readout types
        losses = {}

        # Quantized prediction loss
        pred_quantized = self.forward(x, 'quantized')
        quantiles_tensor = torch.from_numpy(self.quantiles['close']).to(y.device)
        target_quantized = torch.bucketize(y, quantiles_tensor)
        losses['quantized'] = self.ce_loss(pred_quantized.view(-1, self.config.quant_bins), target_quantized.view(-1))

        # Mean prediction loss (Huber)
        pred_mean = self.forward(x, 'mean')
        losses['mean'] = self.huber_loss(pred_mean.squeeze(), y)

        # Mean + Variance prediction loss (NLL)
        pred_mean_var = self.forward(x, 'mean_var')
        pred_mean, pred_var = pred_mean_var[..., 0], pred_mean_var[..., 1]
        losses['nll'] = 0.5 * (torch.log(pred_var) + (y - pred_mean) ** 2 / pred_var).mean()

        # Quantile prediction loss
        pred_quantiles = self.forward(x, 'quantile')
        quantile_targets = self._compute_quantile_targets(y)
        losses['quantile'] = self._quantile_loss(pred_quantiles, quantile_targets)

        # Combine losses
        total_loss = (
            0.25 * losses['quantized'] +
            0.25 * losses['mean'] +
            0.25 * losses['nll'] +
            0.25 * losses['quantile']
        )

        return {
            'loss': total_loss,
            'quantized_loss': losses['quantized'],
            'mean_loss': losses['mean'],
            'nll_loss': losses['nll'],
            'quantile_loss': losses['quantile']
        }

    def _compute_quantile_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Compute quantile targets for training"""

        # Compute empirical quantiles from training data
        y_np = y.detach().cpu().numpy()
        quantiles = np.quantile(y_np, [0.1, 0.25, 0.5, 0.75, 0.9])

        # Create targets
        targets = torch.zeros_like(y).unsqueeze(-1).repeat(1, 5)
        for i, q in enumerate(quantiles):
            targets[:, i] = torch.where(y >= q, torch.ones_like(y), torch.zeros_like(y))

        return targets.to(y.device)

    def _quantile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Quantile loss with sided weighting"""

        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)

        losses = []
        for i, q in enumerate(quantiles):
            error = target[:, i] - pred[:, i]

            # Sided weighting
            weight = torch.where(error > 0,
                               torch.full_like(error, q),
                               torch.full_like(error, 1 - q))

            losses.append((weight * torch.abs(error)).mean())

        return torch.stack(losses).mean()

    def optimize_portfolio(self, predictions: torch.Tensor,
                          current_weights: torch.Tensor) -> torch.Tensor:
        """Optimize portfolio weights using learned policy"""

        new_weights, trading_costs = self.portfolio_optimizer(predictions, current_weights)

        # Portfolio return (simplified - would need actual returns)
        portfolio_return = (new_weights * predictions.squeeze()).sum() - trading_costs

        return portfolio_return, new_weights

    def grpo_step(self, batch: Tuple[torch.Tensor, torch.Tensor], num_rollouts: int = 4) -> Dict[str, torch.Tensor]:
        """GRPO training step with multiple rollouts"""

        x, y = batch
        batch_size = x.size(0)

        # Generate predictions for stock returns
        features = self.encoder(x)
        features = self.backbone(features)

        # Generate multiple rollouts
        rollout_returns = []
        rollout_weights = []

        # Initialize portfolio with equal weights
        current_weights = torch.zeros(batch_size, device=x.device)

        for _ in range(num_rollouts):
            # Add noise to create diverse rollouts
            noisy_features = features + torch.randn_like(features) * 0.1

            # Predict returns
            predicted_returns = self.readout(noisy_features, 'mean').squeeze(-1)

            # Optimize portfolio
            portfolio_return, new_weights = self.optimize_portfolio(
                predicted_returns[:, -1:],  # Use last timestep
                current_weights
            )

            rollout_returns.append(portfolio_return)
            rollout_weights.append(new_weights)

        # Stack rollouts
        rollout_returns = torch.stack(rollout_returns)  # (num_rollouts, batch)
        rollout_weights = torch.stack(rollout_weights)  # (num_rollouts, batch)

        # Compute advantages (relative to mean)
        mean_return = rollout_returns.mean(dim=0, keepdim=True)
        advantages = rollout_returns - mean_return

        # GRPO loss: weight rollouts by their advantage
        grpo_loss = -(advantages * rollout_returns).mean()

        # Also include standard prediction loss
        standard_loss = self.step(batch)['loss']

        # Combine losses
        total_loss = 0.7 * standard_loss + 0.3 * grpo_loss

        return {
            'loss': total_loss,
            'grpo_loss': grpo_loss,
            'standard_loss': standard_loss,
            'mean_portfolio_return': rollout_returns.mean()
        }


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

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 100
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Data
    seq_len: int = 64
    train_ratio: float = 0.9

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4

    # Muon optimizer for 2D parameters
    use_muon: bool = True

    debug_data: bool = False


if __name__ == "__main__":
    # Example usage
    config = FinalPipelineConfig(
        debug_data = True,
    )
    pipeline = FinalPipeline(config)

    # Prepare data
    pipeline.prepare_data()

    # Train the model
    pipeline.fit()

    print("Final pipeline training completed!")