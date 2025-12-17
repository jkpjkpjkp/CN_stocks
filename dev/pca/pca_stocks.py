import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict

print("Loading data in chunks...")
pf = pq.ParquetFile('/home/jkp/ssd/a_1min.pq')

# First pass: get unique dates and stocks
all_stocks = set()
date_range = []

# Sample a few row groups to understand the data
for i in range(0, min(100, pf.num_row_groups), 10):
    chunk = pf.read_row_group(i, columns=['id', 'datetime']).to_pandas()
    all_stocks.update(chunk['id'].unique())
    date_range.extend([chunk['datetime'].min(), chunk['datetime'].max()])

print(f"Approximate stocks: {len(all_stocks)}")
print(f"Date range: {min(date_range)} to {max(date_range)}")

# Process data in daily chunks to reduce memory
# Read all data but process daily
print("\nLoading full dataset...")
df = pd.read_parquet('/home/jkp/ssd/a_1min.pq', columns=['id', 'datetime', 'close'])
print(f"Total rows: {len(df):,}")

# Extract date for grouping
df['date'] = df['datetime'].dt.date

# Get unique dates
unique_dates = sorted(df['date'].unique())
print(f"Number of trading days: {len(unique_dates)}")

# Use daily close prices (last price of each day) instead of minute data
# This dramatically reduces the data size
print("\nAggregating to daily data...")
daily_df = df.groupby(['date', 'id'])['close'].last().reset_index()
del df  # Free memory
print(f"Daily data rows: {len(daily_df):,}")

# Pivot daily data
print("\nPivoting daily data...")
df_pivot = daily_df.pivot(index='date', columns='id', values='close')
print(f"Daily pivot shape: {df_pivot.shape}")

# Drop stocks with too many missing values (keep stocks with >80% data)
threshold = 0.8
valid_stocks = df_pivot.columns[df_pivot.notna().mean() > threshold]
df_pivot = df_pivot[valid_stocks]
print(f"After filtering for >{threshold*100}% data: {df_pivot.shape}")

# Forward fill, then backfill any remaining
df_pivot = df_pivot.ffill().bfill()
print(f"After filling: {df_pivot.shape}")

# Convert to returns
print("\nComputing returns...")
returns = df_pivot.pct_change().dropna()
print(f"Returns shape: {returns.shape}")

# Remove infinite values
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
print(f"After removing inf: {returns.shape}")

# Standardize returns
returns_standardized = (returns - returns.mean()) / returns.std()
returns_standardized = returns_standardized.dropna(axis=1)  # Drop any columns with NaN
print(f"Standardized shape: {returns_standardized.shape}")

# Run PCA on the full dataset
print("\nRunning PCA on full dataset...")
n_components = min(100, returns_standardized.shape[1])
pca = PCA(n_components=n_components)
pca.fit(returns_standardized)

# Plot log eigenvalues (variance explained)
eigenvalues = pca.explained_variance_
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(eigenvalues) + 1), np.log(eigenvalues), 'bo-', markersize=4)
plt.xlabel('Principal Component')
plt.ylabel('Log Eigenvalue (Log Variance Explained)')
plt.title('PCA Log Eigenvalue Spectrum')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(eigenvalues) + 1), np.cumsum(pca.explained_variance_ratio_) * 100, 'ro-', markersize=4)
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('Cumulative Variance Explained')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jkp/h/src/pca_eigenvalues.png', dpi=150)
plt.close()
print("Saved pca_eigenvalues.png")

# Analyze eigenvalue evolution over time using rolling windows
print("\nAnalyzing eigenvalue evolution over time (rolling 60-day windows)...")
window_size = 60
step_size = 20  # Move window by 20 days at a time

eigenvalue_history = []
dates_history = []

n_days = len(returns_standardized)
for start in range(0, n_days - window_size, step_size):
    end = start + window_size
    window_data = returns_standardized.iloc[start:end]

    # Drop columns with any NaN in this window
    window_data = window_data.dropna(axis=1)

    if window_data.shape[1] < 50:  # Need enough stocks
        continue

    n_comp = min(30, window_data.shape[1])
    pca_window = PCA(n_components=n_comp)
    pca_window.fit(window_data)

    eigenvalue_history.append(pca_window.explained_variance_[:n_comp])
    dates_history.append(returns_standardized.index[end])

eigenvalue_history = np.array(eigenvalue_history)
dates_history = pd.to_datetime(dates_history)

print(f"Number of windows: {len(dates_history)}")

# Plot eigenvalue evolution over time
plt.figure(figsize=(14, 8))

# Plot log eigenvalues for first 10 components over time
for i in range(min(10, eigenvalue_history.shape[1])):
    plt.plot(dates_history, np.log(eigenvalue_history[:, i]), label=f'PC{i+1}', alpha=0.8, linewidth=1.5)

plt.xlabel('Date')
plt.ylabel('Log Eigenvalue')
plt.title('Evolution of PCA Log-Eigenvalues Over Time\n(60-day Rolling Windows)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jkp/h/src/pca_eigenvalues_over_time.png', dpi=150)
plt.close()
print("Saved pca_eigenvalues_over_time.png")

# Heatmap view
plt.figure(figsize=(14, 6))
n_pcs_to_show = min(20, eigenvalue_history.shape[1])
plt.imshow(np.log(eigenvalue_history[:, :n_pcs_to_show].T), aspect='auto', cmap='viridis')
plt.colorbar(label='Log Eigenvalue')
plt.ylabel('Principal Component')
plt.xlabel('Time')
plt.title('Log-Eigenvalue Heatmap Over Time')

# Set x-axis labels
n_ticks = 10
tick_positions = np.linspace(0, len(dates_history)-1, n_ticks, dtype=int)
plt.xticks(tick_positions, [dates_history[i].strftime('%Y-%m') for i in tick_positions], rotation=45)

# Set y-axis labels
plt.yticks(range(n_pcs_to_show), [f'PC{i+1}' for i in range(n_pcs_to_show)])
plt.tight_layout()
plt.savefig('/home/jkp/h/src/pca_eigenvalues_heatmap.png', dpi=150)
plt.close()
print("Saved pca_eigenvalues_heatmap.png")

# Plot PC1 eigenvalue specifically to show market regime changes
plt.figure(figsize=(14, 5))
plt.plot(dates_history, np.log(eigenvalue_history[:, 0]), 'b-', linewidth=1.5)
plt.fill_between(dates_history, np.log(eigenvalue_history[:, 0]), alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Log Eigenvalue of PC1')
plt.title('First Principal Component Eigenvalue Over Time\n(Higher = More Correlated Market)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/jkp/h/src/pca_pc1_over_time.png', dpi=150)
plt.close()
print("Saved pca_pc1_over_time.png")

# Print eigenvector info (loadings for first few components)
print("\n" + "="*60)
print("Top 10 stock loadings for first 3 principal components:")
print("="*60)
for i in range(min(3, n_components)):
    loadings = pd.Series(pca.components_[i], index=returns_standardized.columns)
    print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.2f}% variance):")
    print("  Top positive loadings:")
    for stock, val in loadings.nlargest(5).items():
        print(f"    {stock}: {val:.4f}")
    print("  Top negative loadings:")
    for stock, val in loadings.nsmallest(5).items():
        print(f"    {stock}: {val:.4f}")

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
print(f"Total stocks analyzed: {returns_standardized.shape[1]}")
print(f"Total trading days: {returns_standardized.shape[0]}")
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by top 5 PCs: {pca.explained_variance_ratio_[:5].sum()*100:.2f}%")
print(f"Variance explained by top 10 PCs: {pca.explained_variance_ratio_[:10].sum()*100:.2f}%")

print("\nDone!")
