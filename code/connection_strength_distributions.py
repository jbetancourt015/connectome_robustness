"""
    This script plots the distribution of strengths for Drosophila datasets.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Fri 23 Jan 2026
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
import matplotlib.ticker as mticker
import matplotlib as mpl
import logging
import re
import sys
import os
import pickle
from tqdm import tqdm
from scipy.sparse import load_npz
from scipy.stats import poisson as poisson_dist

sys.path.append('../new_code')
import network_processing

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
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)


con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]])/255

# Connectomes list 
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc']

# Figure size (in mm)
mm_to_in = 25.4
width = 40./mm_to_in
height = 40./mm_to_in
width_large = 62./mm_to_in
height_large = 62./mm_to_in
width_grid = 102./mm_to_in
height_grid = 62./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
fig_margins_large = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)
fig_margins_grid = dict(left=0.1, right=0.97, bottom=0.15, top=0.95)

data_dir = '../../data/'
processed_dir = '../processed_data/'
sim_dir = '../simulation_results/'
output_dir = '../../figures/connection_strength_distributions/'
os.makedirs(output_dir, exist_ok=True)

# Import connections data
conn_df = pd.read_parquet(processed_dir+'connections_data.parquet')

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def empirical_hist(data):
    # Initialize bins
    bins = np.arange(int(np.ceil(np.nanmax(data))))
    s_unique = np.zeros(len(bins)-1)
    Ps = np.zeros(len(bins)-1)
    # Fill arrays
    for i in range(len(bins)-1):
        inds_bin = (data > bins[i])*(data <= bins[i+1])
        if np.sum(inds_bin) > 0:
            s_unique[i] = np.mean(data[inds_bin])
            Ps[i] = np.sum(inds_bin)/len(data)
    # Remove absent events
    s_unique = s_unique[Ps > 0];
    Ps = Ps[Ps > 0];
    return s_unique, Ps

def empirical_hist_pd(s):
    # keep finite values only
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.array([]), np.array([])

    # Match your bin logic: (i, i+1]  <=>  ceil(x)-1
    # Example: x=0.2 -> bin 0, x=1.0 -> bin 0, x=1.7 -> bin 1
    bin_idx = np.ceil(x).astype("int64") - 1
    bin_idx = pd.Series(bin_idx, index=x.index)

    g = x.groupby(bin_idx, sort=True)

    s_unique = g.mean().to_numpy()                 # mean value in each occupied bin
    Ps = (g.size() / len(x)).to_numpy()            # mass in each occupied bin

    return s_unique, Ps

def empirical_hist_np(data):
    """Version of empirical_hist for numpy arrays (sparse matrix data)."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.array([]), np.array([])
    
    rounded_data = np.round(data).astype(int)
    s_unique, counts = np.unique(rounded_data, return_counts=True)
    
    # Get non-zero strengths
    mask = s_unique > 0
    s_unique, counts = s_unique[mask], counts[mask]
    
    Ps = counts / np.sum(counts)
    
    return s_unique, Ps

def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

def compute_cdf_np(data):
    """Version of compute_cdf for numpy arrays (sparse matrix data)."""
    data = data[np.isfinite(data)]
    sorted_data = np.sort(data)
    unique_vals, counts = np.unique(sorted_data, return_counts=True)
    cum_counts = np.cumsum(counts)
    cdf = cum_counts / cum_counts[-1]
    return unique_vals, cdf

def log_format(ax, format_x=True, format_y=True):
    ax.set_xscale('log')
    ax.set_yscale('log')
    if format_x:
        # --- Force major ticks at every decade ---
        ax.xaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
        )
        # --- Force minor ticks between decades ---
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0,
                              subs=np.arange(2, 10) * 0.1,
                              numticks=100)
        )
    if format_y:
        # --- Force major ticks at every decade ---
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
        )
        # --- Force minor ticks between decades ---
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0,
                              subs=np.arange(2, 10) * 0.1,
                              numticks=100)
        )

#------------------------------------------------------------------------------
# FAFB DISTRIBUTIONS
#------------------------------------------------------------------------------
# Get distribution of weights
s_fafb, P_fafb = empirical_hist_pd(conn_df['syn_count'])

# Get distribution of multinomial shuffled weights
fafb_shuffled_weights = np.load(sim_dir + 'drosophila_whole_brain_shuffled_weights_multinomial.npz')['weights']
s_shuffled, P_shuffled = empirical_hist_np(fafb_shuffled_weights)

# Get distribution of global shuffled weights
fafb_global_shuffled_weights = np.load(sim_dir + 'drosophila_whole_brain_global_shuffled_weights_multinomial.npz')['weights']
s_global, P_global = empirical_hist_np(fafb_global_shuffled_weights)

# Poisson PMF with lambda = total synapses / total connections
lam = conn_df['syn_count'].sum() / len(conn_df)
k_max = int(lam + 10 * np.sqrt(lam))          # a few std devs out is sufficient
k_vals = np.arange(1, k_max + 1)
pmf_vals = poisson_dist.pmf(k_vals, lam)

# FIGURE: Distribution of all connections in the FAFB dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width_large, height_large))

ax.plot(k_vals, pmf_vals, color='black', lw=1, zorder=1)
ax.scatter(s_global,   P_global,   color='lightgrey', s=10, rasterized=True, clip_on=False, zorder=3)
ax.scatter(s_shuffled, P_shuffled, color='grey',      s=10, rasterized=True, clip_on=False, zorder=4)
ax.scatter(s_fafb,     P_fafb,     color=con_colors[0], s=10, rasterized=True, clip_on=False, zorder=5)
log_format(ax)

ax.set_xlim(1.,None)
ax.set_ylim(1e-8,1.)

plt.subplots_adjust(**fig_margins_large)
plt.savefig(output_dir + 'fafb_distribution.svg', dpi=600)
plt.show()
plt.close()

# Distribution of connections by brain region (using plurality-assigned regions)
region_df = conn_df[['syn_count', 'conn_brain_region']].copy()
region_df = region_df.rename(columns={'conn_brain_region': 'brain_region'})
region_df = region_df[region_df['brain_region'] != 'Other Regions']

# Get region order
with open(processed_dir + 'region_order.pkl', 'rb') as f:
    region_order = pickle.load(f)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: Distribution of connections by brain region (Grid)--------------------
n_cols = 5
n_rows = 3

excluded = [(0,4),(1,4)]
excluded_indices = [exc[0]*n_cols+exc[1] for exc in excluded]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_grid, height_grid), 
                          sharex=True, sharey=True)

region_idx = 0

for idx in tqdm(range(n_cols*n_rows)):
    ax = axes.flat[idx]
    
    if np.isin(idx, excluded_indices):
        ax.set_visible(False)
    else:
        r = region_order[region_idx]
        mask = region_df['brain_region'] == r
        s_region, P_region = empirical_hist_pd(region_df[mask]['syn_count'])
        ax.scatter(s_region, P_region, color=region_colors[r], s=10, rasterized=True, clip_on=False, zorder=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Split two-word region names into two lines
        region_label = r.replace(' ', '\n')
        loc, va, ha = (0.05, 'bottom', 'left') if r=='Periesophageal' else (0.95, 'top', 'right')
        ax.text(loc, loc, region_label, transform=ax.transAxes, fontsize=5,
                ha=ha, va=va)
        # Advance region index
        region_idx += 1

plt.setp(axes, xlim=(1.,None), ylim=(None,1.))
plt.subplots_adjust(**fig_margins_grid, wspace=0.08, hspace=0.08)
plt.savefig(output_dir + 'brain_region_grid.svg', dpi=600)
plt.show()
plt.close()

#------------------------------------------------------------------------------
# OTHER LARGE DATASETS (BANC, MANC)
#------------------------------------------------------------------------------
banc_file = 'banc_connections.csv.gz'
banc_conn_df = pd.read_csv(data_dir + banc_file, compression='gzip')

banc_conn_df = (
    banc_conn_df.groupby(["pre_root_id", "post_root_id"])["syn_count"]
      .sum()
      .reset_index()
)

# Get distribution of weights
s_banc, P_banc = empirical_hist_pd(banc_conn_df['syn_count'])

# FIGURE: Distribution of all connections in the BANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.scatter(s_banc, P_banc, color=con_colors[1], s=10, rasterized=True, clip_on=False, zorder=5)
log_format(ax)
ax.set_xlim(1.,None)
ax.set_ylim(None,1.)
plt.subplots_adjust(**fig_margins)

plt.savefig(output_dir + 'banc_distribution.svg', dpi=600)
plt.show()
plt.close()

manc_file = 'manc_connections.csv'
manc_conn_df = pd.read_csv(data_dir + manc_file)

# Get distribution of weights
s_manc, P_manc = empirical_hist_pd(manc_conn_df['weight'])

# FIGURE: Distribution of all connections in the MANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.scatter(s_manc, P_manc, color=con_colors[2], s=10, rasterized=True, clip_on=False, zorder=5)
log_format(ax)
ax.set_xlim(1.,None)
ax.set_ylim(None,1.)

plt.subplots_adjust(**fig_margins)
plt.savefig(output_dir + 'manc_distribution.svg', dpi=600)
plt.show()
plt.close()

#------------------------------------------------------------------------------
# OTHER SPECIES DISTRIBUTIONS (from processed connectome data)
#------------------------------------------------------------------------------
# Species to plot: c_elegans (idx 2), platynereis_sensory_motor (idx 3), mouse_retina (idx 4)
species_indices = [2, 3, 4]
species_filenames = {
    2: 'c_elegans_distribution.svg',
    3: 'platynereis_distribution.svg',
    4: 'mouse_retina_distribution.svg'
}
# Map species data indices to color indices (c_elegans->3, platynereis->5, mouse_retina->4)
species_color_idx = {2: 3, 3: 5, 4: 4}

for data_idx in species_indices:
    # Load connectome and extract synapse counts
    A = network_processing.load_connectome(data_idx)
    conn_strength = A.data
    
    # Get distribution of weights
    s_species, P_species = empirical_hist_np(conn_strength)
    
    # FIGURE: Distribution of all connections----------------------------------
    fig, ax = plt.subplots(figsize=(width, height))
    
    ax.scatter(s_species, P_species, color=con_colors[species_color_idx[data_idx]], s=10, rasterized=True, clip_on=False, zorder=5)
    log_format(ax)
    ax.set_xlim(1.,None)
    ax.set_ylim(None,1.)
    
    plt.subplots_adjust(**fig_margins)
    plt.savefig(output_dir + species_filenames[data_idx], dpi=600)
    plt.show()
    plt.close()
