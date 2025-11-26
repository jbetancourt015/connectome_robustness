"""
    This script analyzes the loss simulated from FlyWire data .
-------------------------------------------------------------------------------
created on:
    Fri 21 Nov 2024
-------------------------------------------------------------------------------
last change:
    Fri 21 Nov 2025
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
width = 3.5
height = 3.2

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
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))

x_max = 15.

# Create histogram
xbins = np.linspace(mu_min, min(mu_max, x_max), n_bins)
ybins = np.linspace(l_min, l_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['mean'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    # norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color")
)

# Use divider to attach KDE axes to the scatter's box
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# Add labels
ax_scatter.set_xlabel('Average incoming weight')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

ax_scatter.set_xlim(0.,x_max)
# ax_scatter.set_xlim(0.,None)
ax_scatter.set_ylim(0.,None)

plt.show()

# FIGURE: LOSS VS INCOMING VARIANCE (NO COLOR)-------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))

# Create histogram
xbins = np.logspace(np.log10(std_min), np.log10(std_max), n_bins)
ybins = np.linspace(l_min, l_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['std'], neuron_df['sim_loss'],
    bins=[xbins, ybins],
    density=True,
    # norm=LogNorm(),
    cmap=fade_to_color_cmap(con_colors[1], alpha_min=alpha_min, name="fade_to_color")
)

# Set axis scales
ax_scatter.set_xscale('log')

# Use divider to attach KDE axes to the scatter's box
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# Add labels
ax_scatter.set_xlabel('Variance in weight')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

ax_scatter.set_ylim(0.,None)

# plt.savefig(f"../../figures/candidate_figures/fig_2{suffix}.pdf", dpi=600, bbox_inches='tight')

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
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))

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

ax_scatter.set_aspect('equal')

# Use divider to attach colorbar
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# Add labels
ax_scatter.set_xlabel('Predicted loss')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

ax_scatter.set_xlim(0.,None)
ax_scatter.set_ylim(0.,None)

# Plot y=x line
ax_scatter.plot([0.,max_loss], [0.,max_loss], c='k', ls='--', lw=1, zorder=0, alpha=.5)

# plt.savefig(f"../../figures/candidate_figures/fig_3{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()

#------------------------------------------------------------------------------
# LOSS VS ROBUSTNESS
#------------------------------------------------------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))

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

# Use divider to attach colorbar
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# Add labels
ax_scatter.set_xlabel('Robustness')
ax_scatter.set_ylabel('Simulated loss')
cb.set_label('Density')

ax_scatter.set_xlim(0.,None)
ax_scatter.set_ylim(0.,None)

# plt.savefig(f"../../figures/candidate_figures/fig_4{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()