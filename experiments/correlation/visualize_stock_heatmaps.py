"""Visualize full stock-by-stock correlation matrices as heatmaps."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("/home/jkp/h/src/experiments/correlation")
data = np.load(output_dir / "returns_lagged_correlations.npz", allow_pickle=True)

return_periods = data["return_periods"]
lags = data["lags"]
n_stocks = int(data["n_stocks"])

# Select subset of combinations to plot
# (period, lag) pairs
combos = [
    (1, 0), (1, 5),    # 1m returns at lag 0 and 5
    (5, 0), (5, 5),    # 5m returns at lag 0 and 5
    (15, 0), (15, 10), # 15m returns at lag 0 and 10
    (30, 0), (30, 30), # 30m returns at lag 0 and 30
    (60, 0), (60, 60), # 60m returns at lag 0 and 60
]

# Also add some cross-period combos
cross_combos = [
    (1, 5, 0),   # 1m vs 5m at lag 0
    (1, 60, 0),  # 1m vs 60m at lag 0
    (5, 30, 0),  # 5m vs 30m at lag 0
    (15, 60, 0), # 15m vs 60m at lag 0
]

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()

# Subsample stocks for visualization
step = max(1, n_stocks // 300)
idx = np.arange(0, n_stocks, step)

plot_idx = 0

# Same-period correlations
for period, lag in combos[:8]:
    if plot_idx >= 12:
        break
    key = f"p{period}_p{period}_lag{lag}"
    corr = data[key]

    # Sort by mean correlation to reveal structure
    mean_corr = np.nanmean(corr, axis=1)
    sort_idx = np.argsort(mean_corr)[::-1]
    corr_sorted = corr[sort_idx][:, sort_idx]

    # Subsample
    corr_sub = corr_sorted[::step, ::step]

    im = axes[plot_idx].imshow(corr_sub, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[plot_idx].set_title(f"{period}m ret, lag={lag}m", fontsize=11)
    axes[plot_idx].set_xlabel("Stock")
    axes[plot_idx].set_ylabel("Stock")
    plot_idx += 1

# Cross-period correlations
for p1, p2, lag in cross_combos:
    if plot_idx >= 12:
        break
    key = f"p{p1}_p{p2}_lag{lag}"
    corr = data[key]

    mean_corr = np.nanmean(corr, axis=1)
    sort_idx = np.argsort(mean_corr)[::-1]
    corr_sorted = corr[sort_idx][:, sort_idx]
    corr_sub = corr_sorted[::step, ::step]

    im = axes[plot_idx].imshow(corr_sub, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[plot_idx].set_title(f"{p1}m vs {p2}m, lag={lag}m", fontsize=11)
    axes[plot_idx].set_xlabel("Stock")
    axes[plot_idx].set_ylabel("Stock")
    plot_idx += 1

plt.tight_layout()
fig.colorbar(im, ax=axes, shrink=0.6, label="Correlation", location="right")
plt.suptitle("Stock-by-Stock Correlation Matrices (sorted by mean corr)\nLast trading week Dec 2022",
             fontsize=14, y=1.02)
plt.savefig(output_dir / "returns_stock_heatmaps.png", dpi=150, bbox_inches="tight")
print(f"Saved to {output_dir / 'returns_stock_heatmaps.png'}")

plt.close()
