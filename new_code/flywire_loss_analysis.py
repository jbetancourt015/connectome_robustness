"""
    This script analyzes the loss simulated from FlyWire data .
-------------------------------------------------------------------------------
created on:
    Fri 21 Nov 2024
-------------------------------------------------------------------------------
last change:
    Tue 20 Jan 2026
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
import logging
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from tqdm import tqdm

plt.rcParams.update({
    'text.usetex': False,  # keep LaTeX off globally
    'mathtext.fontset': 'cm',   # or 'stixsans' for sans-serif
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,   # thin axis lines for publication
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 1.7
height = 1.7

alpha_min = 0.

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

data_idx = 5
thresholded = False
scheme = 'remove'

suffix = '_thresholded' if thresholded else ''

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

def general_loss(mean, var):
    """Compute predicted loss from mean and variance."""
    rob = np.sqrt(mean + var/mean)
    return (1/np.pi)*np.arccos((1.+1./rob**2)**(-1/2))

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

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

#------------------------------------------------------------------------------
# LOSS BY NEURON STATISTICS
#------------------------------------------------------------------------------
# Get range of statistics
nonneg = neuron_df['std'] > 1e-5
l_min, l_max = np.min(neuron_df['sim_loss']), np.max(neuron_df['sim_loss'])
mu_min, mu_max = np.min(neuron_df['mean']), np.max(neuron_df['mean'])
std_min, std_max = np.min(neuron_df[nonneg]['std']), np.max(neuron_df[nonneg]['std'])
rob_min, rob_max = np.min(neuron_df['robustness']), np.max(neuron_df['robustness'])

# Bins for histogram
n_bins = 100

# FIGURE: LOSS VS INCOMING WEIGHT (NO COLOR)----------------------------------
# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])

x_max = 15.

# Create histogram
xbins = np.linspace(mu_min, min(mu_max, x_max), n_bins)
ybins = np.linspace(l_min, l_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['mean'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color")
)

ax_scatter.set_xlim(0.,x_max)
ax_scatter.set_ylim(0.,None)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)
cb.ax.tick_params(labelsize=8)

# Save without labels
plt.savefig('../../paper_figures/simulated_loss/loss_vs_mean.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Average incoming weight')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

plt.show()

# FIGURE: LOSS VS INCOMING WEIGHT (LOG)----------------------------------
# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])

x_max = 15.

# Create histogram
xbins = np.logspace(np.log10(mu_min), np.log10(mu_max), n_bins)
ybins = np.linspace(l_min, l_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['mean'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color")
)

ax_scatter.set_xscale('log')
ax_scatter.set_ylim(0.,None)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)
cb.ax.tick_params(labelsize=8)

# Save without labels
# plt.savefig('../../paper_figures/simulated_loss/loss_vs_mean.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Average incoming weight')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

plt.show()

# FIGURE: LOSS VS MEAN (BINNED SCATTER, COLORED BY VARIANCE)------------------
# Filter to neurons with positive variance
df_nonneg = neuron_df[nonneg].copy()

# Set up bins
n_mean_bins = 10
n_var_bins = 5
n_pred = 200

# Create bins for mean (linear) and variance (log)
mean_bins = np.linspace(mu_min, mu_max, n_mean_bins + 1)
var_bins = np.logspace(np.log10(std_min), np.log10(std_max), n_var_bins + 1)

# Assign bin indices
df_nonneg['mean_bin'] = pd.cut(df_nonneg['mean'], bins=mean_bins, labels=False, include_lowest=True)
df_nonneg['var_bin'] = pd.cut(df_nonneg['std'], bins=var_bins, labels=False, include_lowest=True)

# Get prediction values
mean_pred = np.linspace(mu_min, mu_max, n_pred)

# Set up figure
cmap = plt.get_cmap('plasma', n_var_bins)
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

for i in range(n_var_bins):
    # Get variance bin midpoint (geometric mean for log-spaced bins)
    var_mid = np.sqrt(var_bins[i] * var_bins[i+1])
    
    # Filter data for this variance bin
    mask = df_nonneg['var_bin'] == i
    
    # Compute median loss for each mean bin
    grouped = df_nonneg[mask].groupby('mean_bin')['sim_loss'].median()
    
    # Get mean bin midpoints
    mean_midpoints = (mean_bins[:-1] + mean_bins[1:]) / 2
    
    # Plot prediction line
    ax.plot(mean_pred, general_loss(mean_pred, var_mid), c=cmap(i), lw=2, 
            label=f"Var$\\in[{var_bins[i]:.1f}, {var_bins[i+1]:.1f}]$", zorder=0)
    
    # Plot scatter for bins with data
    valid_bins = grouped.index.dropna().astype(int)
    ax.scatter(mean_midpoints[valid_bins], grouped[valid_bins], 
               c='white', edgecolors=cmap(i), s=10, rasterized=True)

# ax.legend(fontsize=6)
ax.set_xlabel('Mean')
ax.set_ylabel('Simulated loss')

plt.savefig('../../paper_figures/simulated_loss/loss_vs_mean_binned.pdf', dpi=600)

plt.show()

# FIGURE: LOSS VS VARIANCE (BINNED SCATTER, COLORED BY MEAN)-----------------
# Set up bins (reuse df_nonneg from above)
n_var_bins_x = 10
n_mean_bins_color = 5

# Create bins for variance (log, x-axis) and mean (linear, color)
var_bins_x = np.logspace(np.log10(std_min), np.log10(std_max), n_var_bins_x + 1)
mean_bins_color = np.linspace(mu_min, mu_max, n_mean_bins_color + 1)

# Assign bin indices
df_nonneg['var_bin_x'] = pd.cut(df_nonneg['std'], bins=var_bins_x, labels=False, include_lowest=True)
df_nonneg['mean_bin_color'] = pd.cut(df_nonneg['mean'], bins=mean_bins_color, labels=False, include_lowest=True)

# Get prediction values
var_pred = np.logspace(np.log10(std_min), np.log10(std_max), n_pred)

# Set up figure
cmap = plt.get_cmap('plasma', n_mean_bins_color)
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

for i in range(n_mean_bins_color):
    # Get mean bin midpoint (arithmetic mean for linear bins)
    mean_mid = (mean_bins_color[i] + mean_bins_color[i+1]) / 2
    
    # Filter data for this mean bin
    mask = df_nonneg['mean_bin_color'] == i
    
    # Compute median loss for each variance bin
    grouped = df_nonneg[mask].groupby('var_bin_x')['sim_loss'].median()
    
    # Get variance bin midpoints (geometric mean for log-spaced bins)
    var_midpoints = np.sqrt(var_bins_x[:-1] * var_bins_x[1:])
    
    # Plot prediction line
    ax.plot(var_pred, general_loss(mean_mid, var_pred), c=cmap(i), lw=2, 
            label=f"Mean$\\in[{mean_bins_color[i]:.1f}, {mean_bins_color[i+1]:.1f}]$", zorder=0)
    
    # Plot scatter for bins with data
    valid_bins = grouped.index.dropna().astype(int)
    ax.scatter(var_midpoints[valid_bins], grouped[valid_bins], 
               c='white', edgecolors=cmap(i), s=10, rasterized=True)

# ax.legend(fontsize=6)
ax.set_xscale('log')
ax.set_xlabel('Variance')
ax.set_ylabel('Simulated loss')

plt.savefig('../../paper_figures/simulated_loss/loss_vs_variance_binned.pdf', dpi=600)

plt.show()

# FIGURE: LOSS VS INCOMING VARIANCE (NO COLOR)-------------------------------
# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])

# Create histogram
xbins = np.logspace(np.log10(std_min), np.log10(std_max), n_bins)
ybins = np.linspace(l_min, l_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['std'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    cmap=fade_to_color_cmap(con_colors[1], alpha_min=alpha_min, name="fade_to_color")
)

# Set axis scales
ax_scatter.set_xscale('log')

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

ax_scatter.set_ylim(0.,None)

plt.savefig('../../paper_figures/simulated_loss/loss_vs_variance.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Variance in weight')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')



plt.show()

#------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
#------------------------------------------------------------------------------
# Compute predicted loss
neuron_df['pred_loss'] = (1/np.pi)*np.arccos((1.+(1./neuron_df['robustness'])**2)**(-1/2))

# Get loss range
min_pred, max_pred = min(neuron_df['pred_loss']), max(neuron_df['pred_loss'])
min_loss, max_loss = min([l_min, min_pred]), max([l_max, max_pred])

# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.15, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.05/1.3, 0.15, 0.03, 0.8])

# Create histogram
xbins = np.linspace(min_loss, max_loss, n_bins)
ybins = np.linspace(min_loss, max_loss, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['pred_loss'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    # norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[2], alpha_min=alpha_min, name="fade_to_color")
)

# ax_scatter.set_aspect('equal')

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

ax_scatter.set_xlim(0.,None)
ax_scatter.set_ylim(0.,None)

# Use the same locator for x and y so ticks coincide
locator = ax_scatter.yaxis.get_major_locator()
ax_scatter.xaxis.set_major_locator(locator)

# Plot y=x line
ax_scatter.plot([0.,max_loss], [0.,max_loss], c='k', ls='--', lw=1, zorder=0, alpha=.5)

plt.savefig('../../paper_figures/simulated_loss/loss_vs_prediction.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Predicted loss')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

plt.show()

#------------------------------------------------------------------------------
# LOSS VS ROBUSTNESS
#------------------------------------------------------------------------------
# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])


# Create histogram
xbins = np.linspace(rob_min, rob_max, n_bins)
ybins = np.linspace(min_loss, max_loss, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['robustness'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    # norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[4], alpha_min=alpha_min, name="fade_to_color")
)

# Plot prediction line
r_vals = np.linspace(rob_min, rob_max, 100)
ax_scatter.plot(r_vals, (1./np.pi)*np.arccos((1.+r_vals**(-2))**(-0.5)), c='k', ls='--', lw=1, label='Prediction', zorder=0, alpha=.5)
ax_scatter.legend()

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

ax_scatter.set_xlim(0.,None)
ax_scatter.set_ylim(0.,None)

plt.savefig('../../paper_figures/simulated_loss/loss_vs_robustness.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Robustness')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

plt.show()

# FIGURE: LOG ROBUSTNESS-------------------------------------------------------
# Set up figure
fig = plt.figure(figsize=(1.3*width,height))

ax_scatter = fig.add_axes([0.2, 0.15, 0.8/1.3, 0.8]) # [left, bottom, width, height]
ax_cbar = fig.add_axes([1.12/1.3, 0.15, 0.03, 0.8])


# Create histogram
xbins = np.logspace(np.log10(rob_min), np.log10(rob_max), n_bins)
ybins = np.linspace(min_loss, max_loss, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['robustness'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    # norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[4], alpha_min=alpha_min, name="fade_to_color")
)

# Plot prediction line
r_vals = np.linspace(rob_min, rob_max, 100)
ax_scatter.plot(r_vals, (1./np.pi)*np.arccos((1.+r_vals**(-2))**(-0.5)), c='k', ls='--', lw=1, label='Prediction', zorder=0, alpha=.5)
ax_scatter.legend()

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# ax_scatter.set_xlim(0.,None)
ax_scatter.set_xscale('log')
ax_scatter.set_ylim(0.,None)

# plt.savefig('../../paper_figures/simulated_loss/loss_vs_robustness.pdf', dpi=600)

# Add labels
ax_scatter.set_xlabel('Robustness')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

plt.show()