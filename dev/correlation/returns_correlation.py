"""
Compute correlation matrices between returns of different periods and different lags.
For the last trading week of 2022.
"""

import polars as pl
import numpy as np
from pathlib import Path

# Load data for last trading week of 2022 (Dec 26-30)
print("Loading data...")
df = pl.scan_parquet("/home/jkp/ssd/a_1min.pq").filter(
    (pl.col("datetime") >= pl.datetime(2022, 12, 26)) &
    (pl.col("datetime") <= pl.datetime(2022, 12, 30, 23, 59, 59))
).collect()

print(f"Loaded {len(df):,} rows, {df['id'].n_unique()} unique stocks")

# Sort by id and datetime
df = df.sort(["id", "datetime"])

# Compute returns for different periods
return_periods = [1, 5, 15, 30, 60]  # in minutes

print("\nComputing returns for different periods...")
for period in return_periods:
    col_name = f"ret_{period}m"
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(period).over("id") - 1).alias(col_name)
    )

# Pivot each return series to wide format
print("Pivoting to wide format...")
returns_wide = {}
for period in return_periods:
    col_name = f"ret_{period}m"
    wide = df.select(["datetime", "id", col_name]).pivot(
        values=col_name,
        index="datetime",
        on="id"
    ).sort("datetime")

    price_cols = [c for c in wide.columns if c != "datetime"]
    returns_wide[period] = wide.select(price_cols).to_numpy()
    print(f"  {period}m returns: shape {returns_wide[period].shape}")

timestamps = wide["datetime"].to_list()
n_time, n_stocks = returns_wide[1].shape

# Compute cross-period lagged correlations
# For each (period1, period2) pair and each lag:
# Compute correlation between ret_period1[t] and ret_period2[t+lag]
lags = [0, 1, 5, 10, 30, 60]  # in minutes

print(f"\nComputing cross-period lagged correlations...")
print(f"Periods: {return_periods}, Lags: {lags}")

# Store results: dict of (period1, period2, lag) -> correlation matrix
correlations = {}

for p1 in return_periods:
    for p2 in return_periods:
        for lag in lags:
            key = (p1, p2, lag)
            r1 = returns_wide[p1]
            r2 = returns_wide[p2]

            if lag == 0:
                # Valid rows where both returns exist
                valid = ~(np.isnan(r1) | np.isnan(r2))
                r1_clean = np.where(valid, r1, 0)
                r2_clean = np.where(valid, r2, 0)
                n_valid = valid.sum(axis=0)

                # Standardize
                r1_std = (r1_clean - np.nanmean(r1, axis=0)) / (np.nanstd(r1, axis=0) + 1e-10)
                r2_std = (r2_clean - np.nanmean(r2, axis=0)) / (np.nanstd(r2, axis=0) + 1e-10)
                r1_std = np.where(valid, r1_std, 0)
                r2_std = np.where(valid, r2_std, 0)

                corr = (r1_std.T @ r2_std) / n_time
            else:
                r1_lag = r1[:-lag]
                r2_lag = r2[lag:]
                n_t = r1_lag.shape[0]

                # Standardize
                r1_std = (r1_lag - np.nanmean(r1_lag, axis=0)) / (np.nanstd(r1_lag, axis=0) + 1e-10)
                r2_std = (r2_lag - np.nanmean(r2_lag, axis=0)) / (np.nanstd(r2_lag, axis=0) + 1e-10)

                # Handle NaNs
                r1_std = np.nan_to_num(r1_std, 0)
                r2_std = np.nan_to_num(r2_std, 0)

                corr = (r1_std.T @ r2_std) / n_t

            correlations[key] = corr

# Compute summary statistics for each combination
print("\n=== Mean Cross-Stock Correlation (off-diagonal) ===")
print(f"{'Period1':>8} {'Period2':>8} {'Lag':>6} {'Mean':>10} {'Std':>10}")
print("-" * 50)

mask = ~np.eye(n_stocks, dtype=bool)
summary = []

for p1 in return_periods:
    for p2 in return_periods:
        for lag in lags:
            corr = correlations[(p1, p2, lag)]
            off_diag = corr[mask]
            mean_corr = np.nanmean(off_diag)
            std_corr = np.nanstd(off_diag)
            summary.append({
                'period1': p1, 'period2': p2, 'lag': lag,
                'mean': mean_corr, 'std': std_corr
            })
            if p1 <= p2:  # Only print upper triangle to reduce output
                print(f"{p1:>8}m {p2:>8}m {lag:>6} {mean_corr:>10.6f} {std_corr:>10.6f}")

# Save results
output_dir = Path("/home/jkp/h/src/experiments/correlation")
np.savez(
    output_dir / "returns_lagged_correlations.npz",
    **{f"p{p1}_p{p2}_lag{lag}": correlations[(p1, p2, lag)]
       for p1 in return_periods for p2 in return_periods for lag in lags},
    return_periods=np.array(return_periods),
    lags=np.array(lags),
    n_stocks=n_stocks
)
print(f"\nSaved to {output_dir / 'returns_lagged_correlations.npz'}")

# Create summary dataframe
summary_df = pl.DataFrame(summary)
summary_df.write_csv(output_dir / "returns_correlation_summary.csv")
print(f"Saved summary to {output_dir / 'returns_correlation_summary.csv'}")
