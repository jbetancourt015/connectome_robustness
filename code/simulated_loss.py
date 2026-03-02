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
import matplotlib.colors as mcolors
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
    'axes.spines.top': False,    # remove top spine
    'axes.spines.right': False,  # remove right spine
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator5
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
mm_to_in = 25.4
width = 55./mm_to_in
height = 55./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)

alpha_min = 0.

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

data_idx = 5
thresholded = False
scheme = 'remove'

eps = 1.

suffix = '_thresholded' if thresholded else ''

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255;

dark_cool = mcolors.LinearSegmentedColormap.from_list(
    "dark_cool",
    ["#0b3c49", "#1f5c8b", "#4a4aa8", "#7a3fa0", "#9b2f8a"]
)
if "dark_cool" not in plt.colormaps:
    plt.colormaps.register(dark_cool)

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

def general_loss(mean, var):
    """Compute predicted loss from mean and variance."""
    rob = np.sqrt(mean + var/mean)
    return (1/np.pi)*np.arccos((1.+(eps/rob)**2)**(-1/2))

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'
sim_dir = '../simulation_results/'

# Import neuron data
neuron_df = pd.read_parquet(processed_dir+'neuron_data.parquet')
loss_df = pd.read_parquet(sim_dir+'loss_data.parquet')

# Append data
neuron_df = neuron_df.merge(loss_df, on='root_id', how='outer')

# Keep only high in-degree neurons
k_min = 10
neuron_df = neuron_df[neuron_df['in_deg'] >= k_min]

# Compute relevant moments
neuron_df['mean'] = neuron_df['in_strength']/neuron_df['in_deg']
neuron_df['var'] = (neuron_df['sum_w2']/neuron_df['in_deg']) - neuron_df['mean']**2

#------------------------------------------------------------------------------
# LOSS BY NEURON STATISTICS
#------------------------------------------------------------------------------
# Get range of statistics
nonneg = neuron_df['var'] > 1e-5
l_min, l_max = np.min(neuron_df['sim_loss']), np.max(neuron_df['sim_loss'])
mu_min, mu_max = np.min(neuron_df['mean']), np.max(neuron_df['mean'])
var_min, var_max = np.min(neuron_df[nonneg]['var']), np.max(neuron_df[nonneg]['var'])
rob_min, rob_max = np.min(neuron_df['robustness']), np.max(neuron_df['robustness'])

# Bins for histogram
n_bins = 100

# FIGURE: LOSS VS VARIANCE (BINNED SCATTER, COLORED BY MEAN)-----------------
# Filter to neurons with positive variance
df_nonneg = neuron_df[nonneg].copy()

# Binning parameters
n_mean_bins = 5
n_var_bins = 5
n_pred = 200

# Compute log-uniform mean bin boundaries
mean_bins = np.logspace(np.log10(df_nonneg['mean'].min()),
                        np.log10(df_nonneg['mean'].max()), n_mean_bins + 1)

# Assign mean bin indices
df_nonneg['mean_bin'] = pd.cut(df_nonneg['mean'], bins=mean_bins, labels=False, include_lowest=True)

# Assign variance bin indices using quantiles within each mean bin
df_nonneg['var_bin'] = -1
for i in range(n_mean_bins):
    if i < n_mean_bins - 1:
        mean_mask = (df_nonneg['mean'] >= mean_bins[i]) & (df_nonneg['mean'] < mean_bins[i+1])
    else:
        mean_mask = (df_nonneg['mean'] >= mean_bins[i]) & (df_nonneg['mean'] <= mean_bins[i+1])
    
    subset_var = df_nonneg.loc[mean_mask, 'var']
    if len(subset_var) == 0:
        continue
    
    var_q = np.percentile(subset_var, np.linspace(0, 100, n_var_bins + 1))
    var_bin_idx = pd.cut(df_nonneg.loc[mean_mask, 'var'], bins=var_q, labels=False, include_lowest=True)
    df_nonneg.loc[mean_mask, 'var_bin'] = var_bin_idx

# Get prediction values spanning the full variance range
var_pred = np.logspace(np.log10(var_min), np.log10(var_max), n_pred)

# Set up figure
cmap = plt.get_cmap('dark_cool', n_mean_bins)
fig, ax = plt.subplots(figsize=(width, height))

for i in range(n_mean_bins):
    mean_mask = df_nonneg['mean_bin'] == i
    if mean_mask.sum() == 0:
        continue
    
    # Get representative mean (median of neurons in this mean bin)
    mean_mid = df_nonneg.loc[mean_mask, 'mean'].median()
    
    # Compute median loss and median variance for each variance bin
    grouped_loss = df_nonneg[mean_mask].groupby('var_bin')['sim_loss'].median()
    grouped_var = df_nonneg[mean_mask].groupby('var_bin')['var'].median()
    
    # Plot prediction line
    ax.plot(var_pred, general_loss(mean_mid, var_pred), c=cmap(i), lw=2, zorder=0)
    
    # Plot scatter for bins with data (using median variance as x-position)
    valid_bins = grouped_loss.index.dropna().astype(int)
    ax.scatter(grouped_var[valid_bins], grouped_loss[valid_bins], c='white', 
               edgecolors=cmap(i), s=20, rasterized=True)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-2,.2])
ax.set_xlim([5e-1,1e4])

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/simulated_loss/loss_vs_variance_binned.svg', dpi=600)

ax.set_xlabel('Variance')
ax.set_ylabel('Simulated loss')

plt.show()

#------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
#------------------------------------------------------------------------------
# Figure parameters
log_axes = False
log_prob = False
n_bins = 30

# Compute predicted loss
neuron_df['pred_loss'] = (1/np.pi)*np.arccos((1.+(eps/neuron_df['robustness'])**2)**(-1/2))



# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Create histogram
if log_axes:
    # Get loss range
    min_loss, max_loss = 1e-2, .2
    
    xbins = np.logspace(np.log10(min_loss), np.log10(max_loss), n_bins)
    ybins = np.copy(xbins)
    ax.set_xscale('log')
    ax.set_yscale('log')
else:
    # Get loss range
    min_loss, max_loss = 0., .2
    
    xbins = np.linspace(min_loss, max_loss, n_bins)
    ybins = np.linspace(min_loss, max_loss, n_bins)

counts, xedges, yedges = np.histogram2d(
    neuron_df['pred_loss'], neuron_df['sim_loss'],
    bins=[xbins, ybins]
)
prob = counts / counts.sum()
prob_masked = np.ma.masked_where(prob == 0, prob)
im = ax.pcolormesh(
    xedges, yedges, prob_masked.T, norm=LogNorm() if log_prob else None,
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color")
)

ax.set_xlim(min_loss,max_loss)
ax.set_ylim(min_loss,max_loss)

# Use the same locator for x and y so ticks coincide
locator = ax.yaxis.get_major_locator()
ax.xaxis.set_major_locator(locator)

# Plot y=x line
ax.plot([0.,max_loss], [0.,max_loss], c='k', ls='--', lw=1, zorder=0)

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/simulated_loss/loss_vs_prediction.svg', dpi=600)

# Add labels
ax.set_xlabel('Predicted loss')
ax.set_ylabel('Simulated loss')

# Add colorbar after saving (only shows in plt.show(), not in PDF)
cb = fig.colorbar(im, ax=ax)
cb.set_label('Probability')

plt.show()

#------------------------------------------------------------------------------
# LOSS VS ROBUSTNESS (BINNED SCATTER)
#------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Plot scatter points for each mean bin (same bins as first plot)
for i in range(n_mean_bins):
    # Filter data for this mean bin
    mean_mask = df_nonneg['mean_bin'] == i
    
    # Get representative mean (median of neurons in this mean bin)
    mean_mid = df_nonneg.loc[mean_mask, 'mean'].median()
    
    # Compute median loss and median variance for each variance bin
    grouped_loss = df_nonneg[mean_mask].groupby('var_bin')['sim_loss'].median()
    grouped_var = df_nonneg[mean_mask].groupby('var_bin')['var'].median()
    
    # Compute robustness for each variance bin
    grouped_rob = np.sqrt(mean_mid + grouped_var / mean_mid)
    
    # Plot scatter for bins with data
    valid_bins = grouped_loss.index.dropna().astype(int)
    ax.scatter(grouped_rob[valid_bins], grouped_loss[valid_bins], c='white', 
               edgecolors=cmap(i), s=20, zorder=2, rasterized=True)

# Plot prediction line
r_vals = np.linspace(rob_min, rob_max, 100)
ax.plot(r_vals, (1./np.pi)*np.arccos((1.+(eps/r_vals)**2)**(-0.5)), c='k', ls='--', lw=1, zorder=0)

# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.,None])
ax.set_ylim([1e-2,.2])

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/simulated_loss/loss_vs_robustness_binned.svg', dpi=600)

# Add labels
ax.set_xlabel('Robustness')
ax.set_ylabel('Simulated loss')

plt.show()