"""
Simple Quantile-Based Portfolio Strategy

Strategy:
- Use last 960 minutes (2 x 480min periods) of the 1024 prediction window
- At start of each 480min period, select top 10% stocks by predicted 480min return
- Hold selected stocks with equal weight
- No transaction fees considered
"""

import torch
import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

from ..pipelines.final_pipeline import FinalPipeline, FinalPipelineConfig


@dataclass
class QuantilePortfolioConfig:
    top_quantile: float = 0.10  # top 10% of stocks
    seq_len: int = 2048  # model sequence length
    pred_len: int = 1024  # latter half for predictions
    min_stock_coverage: float = 0.9  # drop stocks with <90% data in window
    batch_size: int = 64  # batch size for inference
    checkpoint_path: str = '/home/jkp/ssd/me'
    val_data_path: str = '/home/jkp/ssd/a_1min.pq'
    db_path: str = '/home/jkp/ssd/pipeline.duckdb'
    cutoff_path: str = '/home/jkp/ssd/pipeline_cutoff.txt'
    device: str = 'cuda'


class QuantilePortfolioSimulator:
    def __init__(self, config: Optional[QuantilePortfolioConfig] = None):
        self.config = config or QuantilePortfolioConfig()
        self.model = None
        self.stats = None

    def load_model(self):
        """Load the prediction model from checkpoint."""
        print("Loading model checkpoint...")
        import __main__
        __main__.FinalPipelineConfig = FinalPipelineConfig

        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
        model_config = checkpoint['config']
        self.model = FinalPipeline(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        self.model.eval()

        mmap_dir = Path('/home/jkp/ssd/pipeline_mmap')
        self.model.encoder_quantiles = torch.from_numpy(
            np.load(mmap_dir / 'quantiles.npy')
        ).float().to(self.config.device)
        self.model.encoder.encoder_quantiles = self.model.encoder_quantiles
        print("Model loaded.")

    def load_stats(self):
        """Load stock stats from training database."""
        print("Loading stock stats from training set...")
        con = duckdb.connect(self.config.db_path, read_only=True)
        result = con.execute("""
            SELECT id, mean_close, std_close, mean_volume, std_volume
            FROM stats
        """).fetchall()
        con.close()

        self.stats = {
            row[0][:6]: (row[1], row[2], row[3], row[4])
            for row in result
        }
        print(f"Loaded stats for {len(self.stats)} stocks.")

    def load_validation_data(self) -> pl.DataFrame:
        """Load validation data (after cutoff) as polars DataFrame."""
        print("Loading validation data...")
        cutoff_ns = int(float(Path(self.config.cutoff_path).read_text().strip()))

        df = pl.scan_parquet(self.config.val_data_path).filter(
            pl.col('datetime').cast(pl.Int64) > cutoff_ns
        ).sort(['datetime', 'id']).collect()

        print(f"Validation data: {len(df)} rows")
        return df

    def get_unique_timestamps(self, df: pl.DataFrame) -> np.ndarray:
        """Get all unique timestamps in sorted order."""
        timestamps = (
            df.select(pl.col('datetime').unique())
            .sort('datetime')['datetime']
            .to_numpy()
        )
        return timestamps

    def prepare_window_data(
        self, df: pl.DataFrame, timestamps: np.ndarray, window_start: int
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
        """Prepare aligned data for a 2048-minute window.

        Returns:
            features: {stock_id: (seq_len, num_features) array}
            prices: {stock_id: (seq_len,) array of close prices}
            valid_stocks: list of stock_ids with sufficient coverage
        """
        cfg = self.config
        seq_len = cfg.seq_len
        window_end = window_start + seq_len

        if window_end > len(timestamps):
            return {}, {}, []

        window_timestamps = timestamps[window_start:window_end]
        ts_start, ts_end = window_timestamps[0], window_timestamps[-1]

        window_df = df.filter(
            (pl.col('datetime') >= ts_start) &
            (pl.col('datetime') <= ts_end)
        )

        ts_to_idx = {ts: i for i, ts in enumerate(window_timestamps)}

        features_dict = {}
        prices_dict = {}
        valid_stocks = []

        model_config = self.model.config
        num_features = len(model_config.features)
        feat_idx = {f: i for i, f in enumerate(model_config.features)}

        for stock_id, group in window_df.group_by('id'):
            stock_id = stock_id[0][:6]

            if stock_id not in self.stats:
                continue

            group_df = group.sort('datetime')
            stock_timestamps = group_df['datetime'].to_numpy()

            coverage = len(stock_timestamps) / seq_len
            if coverage < cfg.min_stock_coverage:
                continue

            features = np.full((seq_len, num_features), np.nan, dtype=np.float32)
            close_prices = np.full(seq_len, np.nan, dtype=np.float32)

            close = group_df['close'].to_numpy()
            open_p = group_df['open'].to_numpy()
            high = group_df['high'].to_numpy()
            low = group_df['low'].to_numpy()
            volume = group_df['volume'].to_numpy()

            mean_close, std_close, mean_vol, std_vol = self.stats[stock_id]
            std_close = std_close + 1e-8
            std_vol = std_vol + 1e-8

            aligned_idx = np.array([ts_to_idx.get(ts, -1) for ts in stock_timestamps])
            valid_mask = aligned_idx >= 0
            aligned_idx = aligned_idx[valid_mask]

            close = close[valid_mask]
            open_p = open_p[valid_mask]
            high = high[valid_mask]
            low = low[valid_mask]
            volume = volume[valid_mask]

            close_prices[aligned_idx] = close

            features[aligned_idx, feat_idx['close_norm']] = (close - mean_close) / std_close * 10
            features[aligned_idx, feat_idx['volume_norm']] = (volume - mean_vol) / std_vol

            if len(close) > 1:
                delta_1min = np.zeros(len(close))
                delta_1min[:-1] = close[1:] - close[:-1]
                features[aligned_idx, feat_idx['delta_1min']] = delta_1min

                ret_1min = np.zeros(len(close))
                ret_1min[:-1] = (close[1:] / (close[:-1] + 1e-8) - 1) * 100
                features[aligned_idx, feat_idx['ret_1min']] = ret_1min

            for f in ['ret_30min', 'ret_240min', 'ret_480min', 'ret_1200min', 'ret_4800min', 'ret_19200min', 'ret_76800min']:
                if f in feat_idx:
                    features[aligned_idx, feat_idx[f]] = 0

            features[aligned_idx, feat_idx['close_open']] = (close / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['high_open']] = (high / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['low_open']] = (low / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['high_low']] = (high / (low + 1e-8) - 1) * 1000

            for col in range(num_features):
                mask = np.isnan(features[:, col])
                if mask.any() and not mask.all():
                    idx = np.where(~mask, np.arange(seq_len), 0)
                    np.maximum.accumulate(idx, out=idx)
                    features[:, col] = features[idx, col]

            price_mask = np.isnan(close_prices)
            if price_mask.any() and not price_mask.all():
                idx = np.where(~price_mask, np.arange(seq_len), 0)
                np.maximum.accumulate(idx, out=idx)
                close_prices = close_prices[idx]

            if np.isnan(features).any() or np.isnan(close_prices).any():
                continue

            features_dict[stock_id] = features
            prices_dict[stock_id] = close_prices
            valid_stocks.append(stock_id)

        return features_dict, prices_dict, valid_stocks

    @torch.no_grad()
    def batch_predict_returns(
        self, features_dict: Dict[str, np.ndarray], valid_stocks: list[str]
    ) -> Dict[str, np.ndarray]:
        """Batch inference for all stocks.

        Returns: {stock_id: (pred_len, num_horizons) array of predicted returns}
        """
        if not valid_stocks:
            return {}

        cfg = self.config
        seq_len = cfg.seq_len

        stock_features = [features_dict[sid] for sid in valid_stocks]
        batch = np.stack(stock_features, axis=0)

        predictions = {}
        num_stocks = len(valid_stocks)

        for batch_start in range(0, num_stocks, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, num_stocks)
            batch_stocks = valid_stocks[batch_start:batch_end]

            x = torch.from_numpy(batch[batch_start:batch_end]).to(self.config.device)

            encoded = self.model.encoder(x)
            hidden = self.model.backbone(encoded)

            return_mean = self.model.readout(hidden, 'return_mean')
            preds = return_mean[:, seq_len // 2:, :].cpu().numpy()

            for i, stock_id in enumerate(batch_stocks):
                predictions[stock_id] = preds[i]

        return predictions

    def simulate_period(
        self,
        prices_dict: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray],
        valid_stocks: list[str]
    ) -> tuple[float, float]:
        """Simulate trading for one 1024-minute prediction period.

        Strategy:
        - Use last 960 minutes (2 x 480min periods) of the 1024 prediction window
        - At start of each 480min period, select top 10% stocks by predicted 480min return
        - Hold selected stocks with equal weight
        - No transaction fees

        Returns: (log_return, raw_return)
        """
        cfg = self.config

        # Two 480-minute holding periods in prediction window indices:
        # Period 1: t=64 to t=543 (480 minutes)
        # Period 2: t=544 to t=1023 (480 minutes)
        period_starts = [64, 544]
        period_ends = [544, 1024]

        cumulative_return = 1.0

        for t_start, t_end in zip(period_starts, period_ends):
            # Get 480min predictions at start of holding period
            stock_preds = {
                sid: predictions[sid][t_start][3]  # index 3 = 480min horizon
                for sid in valid_stocks
                if sid in predictions
            }

            # Select top 10% stocks
            num_stocks = len(stock_preds)
            top_k = int(num_stocks * cfg.top_quantile)

            sorted_stocks = sorted(
                stock_preds.keys(),
                key=lambda s: stock_preds[s],
                reverse=True
            )[:top_k]

            # Get prices at start and end of holding period
            # prices_dict has shape (seq_len,), prediction window starts at seq_len//2
            start_prices = {
                sid: prices_dict[sid][cfg.seq_len // 2 + t_start]
                for sid in sorted_stocks
            }
            end_prices = {
                sid: prices_dict[sid][cfg.seq_len // 2 + t_end - 1]
                for sid in sorted_stocks
            }

            # Equal weight allocation
            weight = 1.0 / len(sorted_stocks)

            # Calculate period return
            period_return = 0.0
            for sid in sorted_stocks:
                stock_return = (end_prices[sid] / start_prices[sid]) - 1
                period_return += weight * stock_return

            cumulative_return *= (1 + period_return)

        raw_return = cumulative_return - 1
        log_return = np.log(cumulative_return)

        return log_return, raw_return

    def run_simulation(self):
        """Run the full portfolio simulation."""
        self.load_model()
        self.load_stats()
        df = self.load_validation_data()

        timestamps = self.get_unique_timestamps(df)
        print(f"Total unique timestamps: {len(timestamps)}", flush=True)

        cfg = self.config
        seq_len = cfg.seq_len
        pred_len = cfg.pred_len

        results = []
        period_idx = 0

        for window_start in range(0, len(timestamps) - seq_len, pred_len):
            print(f"\n--- Period {period_idx} (window_start={window_start}) ---", flush=True)

            features_dict, prices_dict, valid_stocks = self.prepare_window_data(
                df, timestamps, window_start
            )

            if len(valid_stocks) < 10:
                print(f"Skipping: only {len(valid_stocks)} valid stocks", flush=True)
                continue

            print(f"Valid stocks: {len(valid_stocks)}", flush=True)

            predictions = self.batch_predict_returns(features_dict, valid_stocks)

            log_ret, raw_ret = self.simulate_period(
                prices_dict, predictions, valid_stocks
            )

            results.append({
                'period': period_idx,
                'window_start': window_start,
                'num_stocks': len(valid_stocks),
                'log_return': log_ret,
                'return': raw_ret,
            })

            print(f"Return: {raw_ret * 100:.2f}%, Log return: {log_ret:.4f}", flush=True)
            period_idx += 1

        if results:
            total_log_return = sum(r['log_return'] for r in results)
            avg_return = np.mean([r['return'] for r in results])

            print(f"\n{'=' * 50}")
            print("SUMMARY")
            print(f"{'=' * 50}")
            print(f"Total periods: {len(results)}")
            print(f"Total log return: {total_log_return:.4f}")
            print(f"Average return per period: {avg_return * 100:.2f}%")
            print(f"Cumulative return: {(np.exp(total_log_return) - 1) * 100:.2f}%")

        return results


if __name__ == "__main__":
    config = QuantilePortfolioConfig()
    simulator = QuantilePortfolioSimulator(config)
    results = simulator.run_simulation()
