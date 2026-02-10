"""
    This script generates a 2D histogram of mean vs variance from FlyWire data.
-------------------------------------------------------------------------------
created on:
    Tue 27 Jan 2026
-------------------------------------------------------------------------------
last change:
    Tue 27 Jan 2026
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import logging

plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 1.7
height = 1.7

alpha_min = 0.2

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

#------------------------------------------------------------------------------
# LOAD DATA
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'

# Import neuron data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
loss_df = pd.read_parquet(data_dir+'loss_data.parquet')

# Append data
neuron_df = neuron_df.merge(loss_df, on='root_id', how='outer')

# Keep only high in-degree neurons
k_min = 10
neuron_df = neuron_df[neuron_df['in_deg'] >= k_min]

# Compute relevant moments
neuron_df['mean'] = neuron_df['in_strength']/neuron_df['in_deg']
neuron_df['std'] = (neuron_df['sum_w2']/neuron_df['in_deg']) - neuron_df['mean']**2

# Filter to positive variance only
nonneg = neuron_df['std'] > 1e-5
df_filtered = neuron_df[nonneg].copy()

#------------------------------------------------------------------------------
# 2D HISTOGRAM: MEAN VS VARIANCE
#------------------------------------------------------------------------------
# Get range of statistics
mu_min, mu_max = np.min(df_filtered['mean']), np.max(df_filtered['mean'])
var_min, var_max = np.min(df_filtered['std']), np.max(df_filtered['std'])

# Number of bins for histogram
n_bins = 100

# Number of quantile bins for grid overlay
n_mean = 5   # quintiles for mean
n_var = 5    # quintiles for variance within each mean bin

# Compute mean quantile boundaries
mean_quantiles = np.percentile(df_filtered['mean'], np.linspace(0, 100, n_mean + 1))

# Compute variance quantiles within each mean bin
var_quantiles_per_bin = []
for i in range(n_mean):
    if i < n_mean - 1:
        mask = (df_filtered['mean'] >= mean_quantiles[i]) & (df_filtered['mean'] < mean_quantiles[i+1])
    else:
        # Include right edge for the last bin
        mask = (df_filtered['mean'] >= mean_quantiles[i]) & (df_filtered['mean'] <= mean_quantiles[i+1])
    subset = df_filtered.loc[mask, 'std']
    var_q = np.percentile(subset, np.linspace(0, 100, n_var + 1))
    var_quantiles_per_bin.append(var_q)

# Set up figure
fig = plt.figure(figsize=(1.3*width, height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8])
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])

# Create histogram bins (log for both mean and variance)
xbins = np.logspace(np.log10(mu_min), np.log10(mu_max), n_bins)
ybins = np.logspace(np.log10(var_min), np.log10(var_max), n_bins)

# Compute histogram counts and normalize to probability
counts, xedges, yedges = np.histogram2d(
    df_filtered['mean'], df_filtered['std'],
    bins=[xbins, ybins]
)
prob = counts / counts.sum()

# Plot with pcolormesh (mask zeros for LogNorm)
prob_masked = np.ma.masked_where(prob == 0, prob)
im = ax_scatter.pcolormesh(
    xedges, yedges, prob_masked.T,
    norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color")
)

# Set axis scales (log for both axes)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')

# Set axis limits explicitly based on bin edges
ax_scatter.set_xlim(xedges[0], xedges[-1])
ax_scatter.set_ylim(yedges[0], yedges[-1])

# Enable all four spines
for loc in ["left", "right", "top", "bottom"]:
    ax_scatter.spines[loc].set_visible(True)
ax_scatter.set_frame_on(True)

# Draw quantile grid overlay
# Vertical lines at mean quantile boundaries (skip outer edges)
for mq in mean_quantiles[1:-1]:
    ax_scatter.axvline(mq, color='k', lw=0.5, ls='--', alpha=0.7)

# Horizontal segments for variance quantiles within each mean bin
for i, var_q in enumerate(var_quantiles_per_bin):
    x_left = mean_quantiles[i]
    x_right = mean_quantiles[i + 1]
    for vq in var_q[1:-1]:  # skip outer edges
        ax_scatter.hlines(vq, x_left, x_right, color='k', lw=0.5, ls='--', alpha=0.7)

# Colorbar
cb = fig.colorbar(im, cax=ax_cbar)
cb.ax.tick_params(labelsize=8)

# Save without labels
plt.savefig('../../paper_figures/mean_variance_histogram.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Mean incoming weight')
ax_scatter.set_ylabel('Variance in weight')
cb.set_label('Probability')

plt.show()

