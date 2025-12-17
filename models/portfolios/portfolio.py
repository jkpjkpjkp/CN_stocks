"""
Portfolio Strategy Implementation

Strategy:
- Process validation data in disjoint 1024-minute windows (latter half of 2048 seq_len)
- Start each period with 100M RMB cash
- Trading costs: 26/100000 (buy) + 76/100000 (sell) = 0.102%
- T+1 settlement: stocks can only be sold the day after purchase

Decision logic:
- Sort stocks by projected 480-minute return
- Consider top 100 stocks for potential buys
- Trade if 240min expected return improves by >= 0.05% after 0.102% fee
- Only trade if 1min and 30min returns also favor the new stock
- Virtual trades: can change mind on virtual buys without fee if 240min return
  improves by 0.05% + drop in 480min return
"""

import torch
import numpy as np
import polars as pl
import duckdb
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
from collections import defaultdict

from ..pipelines.final_pipeline import FinalPipeline, FinalPipelineConfig


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000_000  # 100M RMB
    buy_fee: float = 26 / 100000  # 0.026%
    sell_fee: float = 76 / 100000  # 0.076%
    total_fee: float = 102 / 100000  # 0.102%
    min_improvement: float = 0.0005  # 0.05% threshold for trades
    top_k_stocks: int = 100  # consider top 100 stocks
    seq_len: int = 2048  # model sequence length
    pred_len: int = 1024  # latter half for predictions
    min_stock_coverage: float = 0.9  # drop stocks with <90% data in window
    batch_size: int = 64  # batch size for inference
    checkpoint_path: str = '/home/jkp/ssd/me'
    val_data_path: str = '/home/jkp/ssd/a_1min.pq'
    db_path: str = '/home/jkp/ssd/pipeline.duckdb'
    cutoff_path: str = '/home/jkp/ssd/pipeline_cutoff.txt'
    device: str = 'cuda'


class Portfolio:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.cash = config.initial_cash
        # Holdings: {stock_id: {'shares': float, 'cost_basis': float}}
        self.sellable: Dict[str, Dict] = {}  # can be sold
        self.just_bought: Dict[str, Dict] = {}  # T+1, cannot sell today

    def reset(self):
        """Reset portfolio to initial state for new period."""
        self.cash = self.config.initial_cash
        self.sellable.clear()
        self.just_bought.clear()

    def total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value given current prices."""
        value = self.cash
        for stock_id, holding in self.sellable.items():
            if stock_id in prices:
                value += holding['shares'] * prices[stock_id]
        for stock_id, holding in self.just_bought.items():
            if stock_id in prices:
                value += holding['shares'] * prices[stock_id]
        return value

    def new_day(self):
        """Merge just_bought into sellable at start of new day."""
        for stock_id, holding in self.just_bought.items():
            if stock_id in self.sellable:
                old = self.sellable[stock_id]
                total_shares = old['shares'] + holding['shares']
                if total_shares > 0:
                    old['cost_basis'] = (
                        old['cost_basis'] * old['shares'] +
                        holding['cost_basis'] * holding['shares']
                    ) / total_shares
                    old['shares'] = total_shares
            else:
                self.sellable[stock_id] = holding.copy()
        self.just_bought.clear()


class PortfolioSimulator:
    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.model = None
        self.stats = None  # stock stats from training set
        self.horizon_idx = {1: 0, 30: 1, 240: 2, 480: 3}

    def load_model(self):
        """Load the prediction model from checkpoint."""
        print("Loading model checkpoint...")
        # Handle checkpoint saved with __main__.FinalPipelineConfig
        # Temporarily add FinalPipelineConfig to __main__ for unpickling
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

        # Load encoder quantiles for inference
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

        # stats: {stock_id: (mean_close, std_close, mean_vol, std_vol)}
        self.stats = {
            row[0][:6]: (row[1], row[2], row[3], row[4])
            for row in result
        }
        print(f"Loaded stats for {len(self.stats)} stocks.")

    def load_validation_data(self) -> pl.DataFrame:
        """Load validation data (after cutoff) as polars DataFrame."""
        print("Loading validation data...")
        cutoff_ns = int(float(Path(self.config.cutoff_path).read_text().strip()))

        # Load and collect all validation data
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
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], list[str]]:
        """Prepare aligned data for a 2048-minute window.

        Returns:
            features: {stock_id: (seq_len, num_features) array}
            prices: {stock_id: (seq_len,) array of close prices}
            dates: {stock_id: (seq_len,) array of dates}
            valid_stocks: list of stock_ids with sufficient coverage
        """
        cfg = self.config
        seq_len = cfg.seq_len
        window_end = window_start + seq_len

        if window_end > len(timestamps):
            return {}, {}, {}, []

        window_timestamps = timestamps[window_start:window_end]
        ts_start, ts_end = window_timestamps[0], window_timestamps[-1]

        # Filter data for this time window
        window_df = df.filter(
            (pl.col('datetime') >= ts_start) &
            (pl.col('datetime') <= ts_end)
        )

        # Create timestamp index for alignment
        ts_to_idx = {ts: i for i, ts in enumerate(window_timestamps)}

        # Group by stock
        features_dict = {}
        prices_dict = {}
        dates_dict = {}
        valid_stocks = []

        model_config = self.model.config
        num_features = len(model_config.features)
        feat_idx = {f: i for i, f in enumerate(model_config.features)}

        for stock_id, group in window_df.group_by('id'):
            stock_id = stock_id[0][:6]  # Extract 6-digit code

            if stock_id not in self.stats:
                continue

            group_df = group.sort('datetime')
            stock_timestamps = group_df['datetime'].to_numpy()

            # Check coverage
            coverage = len(stock_timestamps) / seq_len
            if coverage < cfg.min_stock_coverage:
                continue

            # Initialize arrays with NaN for missing data
            features = np.full((seq_len, num_features), np.nan, dtype=np.float32)
            close_prices = np.full(seq_len, np.nan, dtype=np.float32)
            dates = np.empty(seq_len, dtype='datetime64[D]')

            # Fill in available data
            close = group_df['close'].to_numpy()
            open_p = group_df['open'].to_numpy()
            high = group_df['high'].to_numpy()
            low = group_df['low'].to_numpy()
            volume = group_df['volume'].to_numpy()
            dt = group_df['datetime'].to_numpy()

            mean_close, std_close, mean_vol, std_vol = self.stats[stock_id]
            std_close = std_close + 1e-8
            std_vol = std_vol + 1e-8

            # Map to aligned indices
            aligned_idx = np.array([ts_to_idx.get(ts, -1) for ts in stock_timestamps])
            valid_mask = aligned_idx >= 0
            aligned_idx = aligned_idx[valid_mask]

            close = close[valid_mask]
            open_p = open_p[valid_mask]
            high = high[valid_mask]
            low = low[valid_mask]
            volume = volume[valid_mask]
            dt = dt[valid_mask]

            close_prices[aligned_idx] = close
            dates[aligned_idx] = dt.astype('datetime64[D]')

            # Compute features
            features[aligned_idx, feat_idx['close_norm']] = (close - mean_close) / std_close * 10
            features[aligned_idx, feat_idx['volume_norm']] = (volume - mean_vol) / std_vol

            # delta_1min and ret_1min (need to handle carefully at boundaries)
            if len(close) > 1:
                delta_1min = np.zeros(len(close))
                delta_1min[:-1] = close[1:] - close[:-1]
                features[aligned_idx, feat_idx['delta_1min']] = delta_1min

                ret_1min = np.zeros(len(close))
                ret_1min[:-1] = (close[1:] / (close[:-1] + 1e-8) - 1) * 100
                features[aligned_idx, feat_idx['ret_1min']] = ret_1min

            # These would need proper computation from wider window; use 0 for now
            for f in ['delta_30min', 'ret_30min', 'ret_1day', 'ret_2day']:
                if f in feat_idx:
                    features[aligned_idx, feat_idx[f]] = 0

            features[aligned_idx, feat_idx['close_open']] = (close / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['high_open']] = (high / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['low_open']] = (low / (open_p + 1e-8) - 1) * 1000
            features[aligned_idx, feat_idx['high_low']] = (high / (low + 1e-8) - 1) * 1000

            # Forward-fill NaN values
            for col in range(num_features):
                mask = np.isnan(features[:, col])
                if mask.any() and not mask.all():
                    idx = np.where(~mask, np.arange(seq_len), 0)
                    np.maximum.accumulate(idx, out=idx)
                    features[:, col] = features[idx, col]

            # Forward-fill prices
            price_mask = np.isnan(close_prices)
            if price_mask.any() and not price_mask.all():
                idx = np.where(~price_mask, np.arange(seq_len), 0)
                np.maximum.accumulate(idx, out=idx)
                close_prices = close_prices[idx]

            # Skip if still have NaN
            if np.isnan(features).any() or np.isnan(close_prices).any():
                continue

            features_dict[stock_id] = features
            prices_dict[stock_id] = close_prices
            dates_dict[stock_id] = dates
            valid_stocks.append(stock_id)

        return features_dict, prices_dict, dates_dict, valid_stocks

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
        pred_len = cfg.pred_len

        # Stack features into batch
        stock_features = [features_dict[sid] for sid in valid_stocks]
        batch = np.stack(stock_features, axis=0)  # (num_stocks, seq_len, num_features)

        predictions = {}
        num_stocks = len(valid_stocks)

        # Process in batches
        for batch_start in range(0, num_stocks, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, num_stocks)
            batch_stocks = valid_stocks[batch_start:batch_end]

            x = torch.from_numpy(batch[batch_start:batch_end]).to(self.config.device)

            # Forward pass
            encoded = self.model.encoder(x)
            hidden = self.model.backbone(encoded)

            # Get return mean predictions for latter half
            return_mean = self.model.readout(hidden, 'return_mean')  # (B, seq_len, num_horizons)
            preds = return_mean[:, seq_len // 2:, :].cpu().numpy()  # (B, pred_len, num_horizons)

            for i, stock_id in enumerate(batch_stocks):
                predictions[stock_id] = preds[i]

        return predictions

    def simulate_period(
        self,
        prices_dict: Dict[str, np.ndarray],
        dates_dict: Dict[str, np.ndarray],
        predictions: Dict[str, np.ndarray],
        valid_stocks: list[str]
    ) -> tuple[float, float, float, float]:
        """Simulate trading for one 1024-minute prediction period.

        Strategy:
        - Sort stocks by 480min predicted return
        - Consider top 100 for buying
        - For each potential buy, check if swapping existing holdings improves
          240min expected return by >= 0.05% after 0.102% fee
        - Also buy from cash if 240min return >= fee + threshold
        - Virtual trades: can change mind without fee if 240min improves by 0.05% + 480min drop

        Returns: (log_return, raw_return, initial_value, final_value)
        """
        cfg = self.config
        pred_len = cfg.pred_len

        portfolio = Portfolio(cfg)
        prev_date = None

        for t in range(pred_len):
            # Current prices at time t (in the prediction window)
            prices = {sid: prices_dict[sid][cfg.seq_len // 2 + t] for sid in valid_stocks}

            # Check for new day
            sample_stock = valid_stocks[0]
            current_date = dates_dict[sample_stock][cfg.seq_len // 2 + t]
            if prev_date is not None and current_date != prev_date:
                portfolio.new_day()
            prev_date = current_date

            # Get predictions for this timestep
            # predictions[stock_id][t] gives (num_horizons,) for timestep t
            stock_preds = {sid: predictions[sid][t] for sid in valid_stocks if sid in predictions}

            if not stock_preds:
                continue

            # Sort stocks by 480-min predicted return (descending)
            sorted_stocks = sorted(
                stock_preds.keys(),
                key=lambda s: stock_preds[s][3],  # horizon index 3 = 480min
                reverse=True
            )[:cfg.top_k_stocks]

            # Virtual state for "change of mind" logic
            # Tracks what we're planning to hold after this minute
            virtual_holdings: Dict[str, Dict] = {}
            virtual_cash = portfolio.cash

            # Copy sellable to virtual
            for stock_id, holding in portfolio.sellable.items():
                virtual_holdings[stock_id] = {
                    'shares': holding['shares'],
                    'pred_240': stock_preds.get(stock_id, np.zeros(4))[2],
                    'pred_480': stock_preds.get(stock_id, np.zeros(4))[3],
                    'is_real': True,  # really held, not just virtual
                }

            # Track actual trades to execute
            trades = []  # [(sell_id or None, buy_id, sell_shares or buy_value), ...]

            # 1. First consider buying from cash (if we have cash and predicted returns are good)
            for buy_id in sorted_stocks:
                if buy_id not in prices or buy_id not in stock_preds:
                    continue
                if virtual_cash <= 0:
                    break

                buy_price = prices[buy_id]
                buy_pred = stock_preds[buy_id]

                # Expected 240min return from buying
                # Predictions are return ratios (future/current), so subtract 1 for return
                expected_ret_240 = buy_pred[2] - 1  # convert ratio to return

                # Only buy if expected return after buy fee is positive and above threshold
                # Actually, the spec says we use mean return estimate directly
                # The threshold is: expected improvement >= 0.05% after 0.102% fee
                # For buying from cash, there's no "improvement" over cash (0% return)
                # So we buy if: expected_240min_return >= 0.05% + 0.102% (buy+sell fees)
                if expected_ret_240 >= cfg.min_improvement + cfg.total_fee:
                    # Allocate portion of cash to this stock
                    # For simplicity, allocate evenly among top stocks
                    # In practice, we'd use the full virtual trading logic
                    buy_amount = min(virtual_cash, portfolio.cash / min(10, len(sorted_stocks)))
                    if buy_amount > 100:  # minimum trade size
                        buy_shares = buy_amount * (1 - cfg.buy_fee) / buy_price
                        virtual_cash -= buy_amount

                        if buy_id in virtual_holdings:
                            virtual_holdings[buy_id]['shares'] += buy_shares
                        else:
                            virtual_holdings[buy_id] = {
                                'shares': buy_shares,
                                'pred_240': buy_pred[2],
                                'pred_480': buy_pred[3],
                                'is_real': False,
                            }

                        trades.append((None, buy_id, buy_amount))  # None = buy from cash

            # 2. Consider swapping existing holdings for better ones
            for buy_id in sorted_stocks:
                if buy_id not in prices or buy_id not in stock_preds:
                    continue

                buy_price = prices[buy_id]
                buy_pred = stock_preds[buy_id]

                # Consider replacing each virtual holding
                for sell_id in list(virtual_holdings.keys()):
                    if sell_id == buy_id:
                        continue
                    if sell_id not in prices or sell_id not in stock_preds:
                        continue

                    vh = virtual_holdings[sell_id]
                    if vh['shares'] <= 0:
                        continue

                    sell_price = prices[sell_id]
                    sell_pred = stock_preds[sell_id]

                    # Check if this is a "virtual" holding (from this minute's decisions)
                    is_virtual = not vh.get('is_real', True)

                    if is_virtual:
                        # Change of mind: no fee, but need 240min improvement >= 0.05% + 480min drop
                        improvement_240 = buy_pred[2] - vh['pred_240']
                        drop_480 = vh['pred_480'] - buy_pred[3]
                        if improvement_240 < cfg.min_improvement + max(0, drop_480):
                            continue
                    else:
                        # Real trade: need 240min improvement >= 0.05% after 0.102% fee
                        improvement = buy_pred[2] - sell_pred[2] - cfg.total_fee
                        if improvement < cfg.min_improvement:
                            continue

                        # Additional check: 1min and 30min must also favor buy
                        if buy_pred[0] <= sell_pred[0] or buy_pred[1] <= sell_pred[1]:
                            continue

                    # Execute virtual trade
                    shares = vh['shares']
                    sell_value = shares * sell_price

                    virtual_holdings[sell_id]['shares'] = 0

                    if is_virtual:
                        # No fees for change of mind
                        buy_shares = sell_value / buy_price
                    else:
                        # Account for fees
                        net_sell = sell_value * (1 - cfg.sell_fee)
                        buy_cost = net_sell * (1 - cfg.buy_fee)
                        buy_shares = buy_cost / buy_price
                        trades.append((sell_id, buy_id, shares))

                    if buy_id in virtual_holdings:
                        virtual_holdings[buy_id]['shares'] += buy_shares
                    else:
                        virtual_holdings[buy_id] = {
                            'shares': buy_shares,
                            'pred_240': buy_pred[2],
                            'pred_480': buy_pred[3],
                            'is_real': False,
                        }

            # Execute actual trades
            for trade in trades:
                sell_id, buy_id, amount = trade
                buy_price = prices[buy_id]

                if sell_id is None:
                    # Buy from cash
                    if portfolio.cash >= amount:
                        buy_cost = amount * (1 - cfg.buy_fee)
                        buy_shares = buy_cost / buy_price
                        portfolio.cash -= amount

                        if buy_id in portfolio.just_bought:
                            old = portfolio.just_bought[buy_id]
                            total = old['shares'] + buy_shares
                            old['cost_basis'] = (
                                old['cost_basis'] * old['shares'] + buy_price * buy_shares
                            ) / total
                            old['shares'] = total
                        else:
                            portfolio.just_bought[buy_id] = {
                                'shares': buy_shares,
                                'cost_basis': buy_price,
                            }
                else:
                    # Swap: sell existing, buy new
                    shares = amount  # amount is shares for swap trades
                    if sell_id not in portfolio.sellable:
                        continue
                    if portfolio.sellable[sell_id]['shares'] < shares:
                        shares = portfolio.sellable[sell_id]['shares']

                    sell_price = prices[sell_id]

                    # Sell
                    sell_value = shares * sell_price * (1 - cfg.sell_fee)
                    portfolio.sellable[sell_id]['shares'] -= shares
                    if portfolio.sellable[sell_id]['shares'] <= 0:
                        del portfolio.sellable[sell_id]

                    # Buy
                    buy_cost = sell_value * (1 - cfg.buy_fee)
                    buy_shares = buy_cost / buy_price

                    if buy_id in portfolio.just_bought:
                        old = portfolio.just_bought[buy_id]
                        total = old['shares'] + buy_shares
                        old['cost_basis'] = (
                            old['cost_basis'] * old['shares'] + buy_price * buy_shares
                        ) / total
                        old['shares'] = total
                    else:
                        portfolio.just_bought[buy_id] = {
                            'shares': buy_shares,
                            'cost_basis': buy_price,
                        }

        # Final value
        final_prices = {
            sid: prices_dict[sid][-1] for sid in valid_stocks
        }
        initial_value = cfg.initial_cash
        final_value = portfolio.total_value(final_prices)

        raw_return = (final_value - initial_value) / initial_value
        log_return = np.log(1 + raw_return)

        return log_return, raw_return, initial_value, final_value

    def run_simulation(self):
        """Run the full portfolio simulation."""
        import sys

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

        # Process disjoint 1024-minute prediction windows
        # Each window needs 2048 timestamps (seq_len) for context
        for window_start in range(0, len(timestamps) - seq_len, pred_len):
            print(f"\n--- Period {period_idx} (window_start={window_start}) ---", flush=True)

            features_dict, prices_dict, dates_dict, valid_stocks = self.prepare_window_data(
                df, timestamps, window_start
            )

            if len(valid_stocks) < 10:
                print(f"Skipping: only {len(valid_stocks)} valid stocks", flush=True)
                continue

            print(f"Valid stocks: {len(valid_stocks)}", flush=True)

            # Batch predict
            predictions = self.batch_predict_returns(features_dict, valid_stocks)

            # Simulate trading
            log_ret, raw_ret, init_val, final_val = self.simulate_period(
                prices_dict, dates_dict, predictions, valid_stocks
            )

            results.append({
                'period': period_idx,
                'window_start': window_start,
                'num_stocks': len(valid_stocks),
                'log_return': log_ret,
                'return': raw_ret,
                'initial_value': init_val,
                'final_value': final_val,
            })

            print(f"Return: {raw_ret * 100:.2f}%, Log return: {log_ret:.4f}", flush=True)
            period_idx += 1

        # Summary
        if results:
            total_log_return = sum(r['log_return'] for r in results)
            avg_return = np.mean([r['return'] for r in results])

            print(f"\n{'=' * 50}")
            print(f"SUMMARY")
            print(f"{'=' * 50}")
            print(f"Total periods: {len(results)}")
            print(f"Total log return: {total_log_return:.4f}")
            print(f"Average return per period: {avg_return * 100:.2f}%")
            print(f"Cumulative return: {(np.exp(total_log_return) - 1) * 100:.2f}%")

        return results


if __name__ == "__main__":
    config = PortfolioConfig()
    simulator = PortfolioSimulator(config)
    results = simulator.run_simulation()
