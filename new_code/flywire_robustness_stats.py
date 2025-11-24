"""
    This script plots some statistics of robustness for the FlyWire FAFB 
    dataset.
-------------------------------------------------------------------------------
created on:
    Mon 24 Nov 2025
-------------------------------------------------------------------------------
last change:
    Mon 24 Nov 2025
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
import logging
import re
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

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


con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

# Nature figure size
width = 3.5
height = 3.2

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron data
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

# Add peripherality data
neuron_df = neuron_df.merge(peri_df, on=("root_id"))

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# NORMALIZED VS SCALED ROBUSTNESS
#------------------------------------------------------------------------------
# Get range of statistics
norm_min, norm_max = np.min(neuron_df['norm_robustness']), np.max(neuron_df['norm_robustness'])
scaled_min, scaled_max = np.min(neuron_df['scaled_robustness']), np.max(neuron_df['scaled_robustness'])

# Bins for histogram
n_bins = 100

# FIGURE: LOSS VS INCOMING WEIGHT (NO COLOR)----------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))

scaled_max, norm_max = 3, 50

# Create histogram
xbins = np.linspace(0., scaled_max, n_bins)
ybins = np.linspace(0., norm_max, n_bins)
counts, xedges, yedges, im = ax_scatter.hist2d(
    neuron_df['scaled_robustness'], neuron_df['norm_robustness'],
    bins=[xbins, ybins],
    density=True,
    norm = LogNorm(),
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=0., name="fade_to_color")
)

# Use divider to attach KDE axes to the scatter's box
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Colorbar in its own column
cb = fig.colorbar(im, cax=ax_cbar)

# Add labels
ax_scatter.set_xlabel('Scaled robustness')
ax_scatter.set_ylabel('Normalized robustness')
cb.set_label('Density')

plt.show()

#------------------------------------------------------------------------------
# DISTRIBUTION OF NORMALIZED ROBUSTNESS
#------------------------------------------------------------------------------
rob_vals, rob_cdf = compute_cdf(neuron_df['norm_robustness'])

median_rob = neuron_df['norm_robustness'].median()

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(rob_vals, rob_cdf, where='post', lw=2, color=con_colors[1])
ax.plot([median_rob, median_rob],[0.,1.], lw=1, c='k', ls='--')

# ax.set_xscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# PLOT ROBUSTNESS STATISTICS BY BRAIN REGION
#------------------------------------------------------------------------------
# Sort regions by median
region_order = (
    neuron_df
    .groupby('brain_region')['norm_robustness']
    .median()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: distribution of normalized robustness by brain region----------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

# Get robustness distribution by region
for r in region_order:
    # Compute distribution of robustness
    mask = neuron_df['brain_region']==r
    rob_cdf_index, rob_cdf_values = compute_cdf(neuron_df[mask]['norm_robustness'])
    ax.step(rob_cdf_index, rob_cdf_values, where='post', lw=2, c=region_colors[r], label=r)

ax.plot([median_rob, median_rob],[0.,1.], lw=1, c='k', ls='--')

ax.legend()

ax.set_xscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()






