"""
    This script plots the distribution of strengths for Drosophila datasets.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Mon 15 Dec 2025
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


con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

# Nature figure size
width = 3.5
height = 3.2

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

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

def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# FAFB DISTRIBUTIONS
#------------------------------------------------------------------------------
# Get distribution of weights
s_fafb, P_fafb = empirical_hist_pd(conn_df['syn_count'])
vals_fafb, cdf_fafb = compute_cdf(conn_df['syn_count'])

# FIGURE: Distribution of all connections in the FAFB dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_fafb, P_fafb, color=con_colors[0], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('Probability')

plt.show()

# FIGURE: Distribution of all connections in the FAFB dataset (CDF)------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(vals_fafb, 1-cdf_fafb, color=con_colors[0], where='post')

ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('1 - CDF')

plt.show()

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

# Sort regions by median
region_order = (
    region_df
    .groupby('brain_region')['syn_count']
    .median()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: Distribution of connections by brain region--------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

for r in tqdm(region_order):
    # Get distribution of weights
    mask = region_df['brain_region'] == r
    s_region, P_region = empirical_hist_pd(region_df[mask]['syn_count'])
    ax.scatter(s_region, P_region, color=region_colors[r], s=10, label=r, rasterized=True)

ax.legend()

ax.set_xscale('log')
ax.set_yscale('log')
    
# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('Probability')

plt.show()

# FIGURE: Distribution of connections by brain region (CDF)--------------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

for r in tqdm(region_order):
    # Get distribution of weights
    mask = region_df['brain_region'] == r
    vals_region, cdf_region = compute_cdf(region_df[mask]['syn_count'])
    ax.step(vals_region, 1-cdf_region, color=region_colors[r], where='post', label=r)

ax.legend()

ax.set_xscale('log')
ax.set_yscale('log')
    
# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('1 - CDF')

plt.show()

#------------------------------------------------------------------------------
# OTHER DISTRIBUTIONS
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
vals_banc, cdf_banc = compute_cdf(banc_conn_df['syn_count'])

# FIGURE: Distribution of all connections in the BANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_banc, P_banc, color=con_colors[2], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('Probability')

plt.show()

# FIGURE: Distribution of all connections in the MANC dataset (CDF)------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(vals_banc, 1-cdf_banc, color=con_colors[2], where='post')

ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('1 - CDF')

plt.show()


manc_file = 'manc_connections.csv'
manc_conn_df = pd.read_csv(data_dir + manc_file)

# Get distribution of weights
s_manc, P_manc = empirical_hist_pd(manc_conn_df['weight'])
vals_manc, cdf_manc = compute_cdf(manc_conn_df['weight'])

# FIGURE: Distribution of all connections in the MANC dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_manc, P_manc, color=con_colors[4], s=10, rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('Probability')

plt.show()

# FIGURE: Distribution of all connections in the MANC dataset (CDF)------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(vals_manc, 1-cdf_manc, color=con_colors[4], where='post')

ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('1 - CDF')

plt.show()