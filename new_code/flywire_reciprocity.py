"""
    This script analyzes the statistics of reciprocity in the FlyWire dataset.
-------------------------------------------------------------------------------
created on:
    Mon 8 Dec 2025
-------------------------------------------------------------------------------
last change:
    Mon 8 Dec 2025
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
# RECIPROCAL PAIRS
#------------------------------------------------------------------------------
# Create a reciprocal dataframe
reciprocal_df = conn_df.merge(
    conn_df,
    left_on=["pre_root_id", "post_root_id"],
    right_on=["post_root_id", "pre_root_id"],
    suffixes=("_ab", "_ba"),
    how="inner"
)

reciprocal_pairs = reciprocal_df.drop_duplicates(
    subset=["pre_root_id_ab", "post_root_id_ab"]
)
n_reciprocal = len(reciprocal_pairs)

# Ratio between max and min of reciprocal connections
reciprocal_df['w_max'] = reciprocal_df[['syn_count_ab', 'syn_count_ba']].max(axis=1)
reciprocal_df['w_min'] = reciprocal_df[['syn_count_ab', 'syn_count_ba']].min(axis=1)

reciprocal_df['ratio'] = reciprocal_df['w_max']/reciprocal_df['w_min']

# Statistics
print('Mean ratio:', reciprocal_df['ratio'].mean())
print('Std dev ratio:', reciprocal_df['ratio'].std())

#------------------------------------------------------------------------------
# OVERLAP STATISTICS (RECIPROCITY)
#------------------------------------------------------------------------------
# FIGURE: Distribution of reciprocity------------------------------------------
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.hist(neuron_df['reciprocity'], bins=20, density=True, histtype='step', color=con_colors[0])

ax.set_xlabel('Reciprocity')
ax.set_ylabel('Density')

plt.show()

# FIGURE: Robustness by reciprocity quartile-----------------------------------
n_quantiles = 10

# Add deciles to dataset
neuron_df['reciprocity_q'] = pd.qcut(neuron_df['reciprocity'], q=n_quantiles, labels=False)

# Plot robustness CDF by quartile
fig, ax = plt.subplots(figsize=(.9*width, .9*height))
cmap = plt.get_cmap('plasma', n_quantiles)

for i in range(n_quantiles):
    mask = neuron_df['reciprocity_q'] == i
    rob, cdf = compute_cdf(neuron_df[mask]['norm_robustness'])
    
    # Plot CDF
    ax.step(rob, cdf, where='post', lw=2, c=cmap(i), label=f"Reciprocity D{i+1}")
    
# ax.legend()

ax.set_xscale('log')
    
ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()






