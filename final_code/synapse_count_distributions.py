"""
    This script plots the distribution of strengths for Drosophila datasets.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Thu 22 Jan 2026
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
import pickle
import re
import sys
import os
from tqdm import tqdm
from scipy.sparse import load_npz

sys.path.append('../new_code')
import network_functions

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
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]])/255

# Connectomes list 
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc']

# Nature figure size
width = 2
height = 2

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.20, right=0.95, bottom=0.18, top=0.95)

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'
output_dir = '../../paper_figures/synapse_count_distributions/'
os.makedirs(output_dir, exist_ok=True)

conn_file = 'flywire_connections.csv.gz'

# Import connections data
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
raw_conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')

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
    
    bin_idx = np.ceil(data).astype(int) - 1
    unique_bins, counts = np.unique(bin_idx, return_counts=True)
    
    s_unique = np.array([data[bin_idx == b].mean() for b in unique_bins])
    Ps = counts / len(data)
    
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

def empirical_heterogeneity(data):
    """Compute empirical heterogeneity: <|x_i - x_j|> / <x>."""
    x = np.sort(data[np.isfinite(data)])
    n = len(x)
    if n < 2:
        return np.nan
    # Efficient mean absolute difference via sorted formula
    i = np.arange(1, n + 1)
    mad = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * x)
    return mad / np.mean(x)

#------------------------------------------------------------------------------
# FAFB DISTRIBUTIONS
#------------------------------------------------------------------------------
# Get distribution of weights
s_fafb, P_fafb = empirical_hist_pd(conn_df['syn_count'])

# FIGURE: Distribution of all connections in the FAFB dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.scatter(s_fafb, P_fafb, color=con_colors[5], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

plt.subplots_adjust(**fig_margins)
plt.savefig(output_dir + 'fafb_distribution.pdf')
plt.close()

# Distribution of connections by brain region
region_df = (
    raw_conn_df.groupby(["pre_root_id", "post_root_id", "neuropil"])["syn_count"]
      .sum()
      .reset_index()
)

# Append brain region
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)

region_df['brain_region'] = region_df['neuropil'].map(region_map)

# Drop "Other Regions"
region_df = region_df[region_df['brain_region'] != 'Other Regions']

# Sort regions by empirical heterogeneity
region_order = (
    region_df
    .groupby('brain_region')['syn_count']
    .apply(lambda x: empirical_heterogeneity(x.values))
    .sort_values()
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: Distribution of connections by brain region (Grid)--------------------
fig, axes = plt.subplots(3, 5, figsize=(.7*5*width, .7*3*height), 
                          sharex=True, sharey=True)

for idx, r in enumerate(tqdm(region_order)):
    ax = axes.flat[idx]
    mask = region_df['brain_region'] == r
    s_region, P_region = empirical_hist_pd(region_df[mask]['syn_count'])
    ax.scatter(s_region, P_region, color=region_colors[r], s=10, rasterized=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Split two-word region names into two lines
    region_label = r.replace(' ', '\n')
    ax.text(0.95, 0.95, region_label, transform=ax.transAxes, fontsize=8,
            ha='right', va='top')

# Hide unused panels (14-16, indices 13-15)
for idx in range(len(region_order), 15):
    axes.flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir + 'brain_region_grid.pdf', bbox_inches='tight')
plt.show()
plt.close()

#------------------------------------------------------------------------------
# OTHER LARGE DATASETS (BANC, MANC)
#------------------------------------------------------------------------------
banc_file = 'banc_connections.csv.gz'
banc_conn_df = pd.read_csv(data_dir + banc_file, compression='gzip')

banc_conn_df = (
    banc_conn_df.groupby(["pre_root_id", "post_root_id", "neuropil"])["syn_count"]
      .sum()
      .reset_index()
)

# Get distribution of weights
s_banc, P_banc = empirical_hist_pd(banc_conn_df['syn_count'])

# FIGURE: Distribution of all connections in the BANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.scatter(s_banc, P_banc, color=con_colors[6], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

plt.subplots_adjust(**fig_margins)
plt.savefig(output_dir + 'banc_distribution.pdf')
plt.show()
plt.close()

manc_file = 'manc_connections.csv'
manc_conn_df = pd.read_csv(data_dir + manc_file)

# Get distribution of weights
s_manc, P_manc = empirical_hist_pd(manc_conn_df['weight'])

# FIGURE: Distribution of all connections in the MANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.scatter(s_manc, P_manc, color=con_colors[7], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

plt.subplots_adjust(**fig_margins)
plt.savefig(output_dir + 'manc_distribution.pdf')
plt.show()
plt.close()

#------------------------------------------------------------------------------
# OTHER SPECIES DISTRIBUTIONS (from processed connectome data)
#------------------------------------------------------------------------------
# Species to plot: c_elegans (idx 2), platynereis_sensory_motor (idx 3), mouse_retina (idx 4)
species_indices = [2, 3, 4]
species_filenames = {
    2: 'c_elegans_distribution.pdf',
    3: 'platynereis_distribution.pdf',
    4: 'mouse_retina_distribution.pdf'
}

for data_idx in species_indices:
    # Load connectome and extract synapse counts
    A = network_functions.load_connectome(data_idx)
    synapse_counts = A.data
    
    # Get distribution of weights
    s_species, P_species = empirical_hist_np(synapse_counts)
    
    # FIGURE: Distribution of all connections----------------------------------
    fig, ax = plt.subplots(figsize=(width, height))
    
    ax.scatter(s_species, P_species, color=con_colors[data_idx], s=10, rasterized=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.subplots_adjust(**fig_margins)
    plt.savefig(output_dir + species_filenames[data_idx])
    plt.show()
    plt.close()
