"""
    This script analyzes the distribution of excitatory and inhibitory 
    synapses at the neuron level.
-------------------------------------------------------------------------------
created on:
    Tue 18 Nov 2025
-------------------------------------------------------------------------------
last change:
    Tue 18 Nov 2025
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
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

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

#------------------------------------------------------------------------------
# LOAD DATASETS
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron and connections data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# BUILD NEURON-LEVEL DATASET
#------------------------------------------------------------------------------
# 1. Numerator: sum_i w_ij^2 over *all* inputs to each postsynaptic neuron j
conn_df['syn_count_sq'] = conn_df['syn_count']**2
num = (conn_df
       .groupby('post_root_id')['syn_count_sq']
       .sum()
       .rename('num'))

num_exc = (conn_df.loc[conn_df['nt_class'] == 'exc']
         .groupby('post_root_id')['syn_count_sq']
         .sum()
         .rename('num_exc'))

num_inh = (conn_df.loc[conn_df['nt_class'] == 'inh']
         .groupby('post_root_id')['syn_count_sq']
         .sum()
         .rename('num_inh'))

# 2. Denominator: sum_{i in J} w_ij over *excitatory* inputs only
den_exc = (conn_df.loc[conn_df['nt_class'] == 'exc']
         .groupby('post_root_id')['syn_count']
         .sum()
         .rename('den_exc'))

n_exc = (conn_df.loc[conn_df['nt_class'] == 'exc']
         .groupby('post_root_id')['syn_count']
         .count()
         .rename('n_exc'))

den_inh = (conn_df.loc[conn_df['nt_class'] == 'inh']
         .groupby('post_root_id')['syn_count']
         .sum()
         .rename('den_inh'))

n_inh = (conn_df.loc[conn_df['nt_class'] == 'inh']
         .groupby('post_root_id')['syn_count']
         .count()
         .rename('n_inh'))

# 3. Put together and compute r^J
stats = pd.concat([num, den_exc, den_inh, n_exc, n_inh, num_exc, num_inh], axis=1)

# handle neurons with no excitatory input (den = NaN or 0)
stats['r_exc'] = np.sqrt(stats['num'] / stats['den_exc'])
stats['r_inh'] = np.sqrt(stats['num'] / stats['den_inh'])
stats['r_uni_exc'] = np.sqrt(stats['num_exc']/stats['den_exc'] + stats['den_exc']/stats['n_exc'])
stats['r_uni_inh'] = np.sqrt(stats['num_inh']/stats['den_inh'] + stats['den_inh']/stats['n_inh'])
stats['r_poi_exc'] = np.sqrt(stats['num_exc']/stats['den_exc'] + 1. + stats['den_exc']/stats['n_exc'])
stats['r_poi_inh'] = np.sqrt(stats['num_inh']/stats['den_inh'] + 1. + stats['den_inh']/stats['n_inh'])

stats.loc[(stats['den_exc'].isna()) | (stats['den_exc'] == 0), 'r_exc'] = np.nan
stats.loc[(stats['den_inh'].isna()) | (stats['den_inh'] == 0), 'r_inh'] = np.nan

# `stats` index = post_root_id, column `r_exc` is your r^J(w) for J = "exc"
stats = stats.reset_index()

stats['r_norm_exc'] = (stats['r_exc'] - stats['r_uni_exc'])/(stats['r_poi_exc'] - stats['r_uni_exc'])
stats['r_norm_inh'] = (stats['r_inh'] - stats['r_uni_inh'])/(stats['r_poi_inh'] - stats['r_uni_inh'])

# Plot CDF for both 

# FIGURE: Distribution of robustness by neuron type----------------------------
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

rob_exc, cdf_exc = compute_cdf(stats['r_norm_exc'])
rob_inh, cdf_inh = compute_cdf(stats['r_norm_inh'])

ax.step(rob_exc, cdf_exc, where='post', color=con_colors[0], lw=2, label='Excitatory')
ax.step(rob_inh, cdf_inh, where='post', color=con_colors[1], lw=2, label='Inhibitory')

ax.legend()

ax.set_xscale('log')
ax.set_xlim([1e-1,1e3])

ax.set_xlabel('Normalized fractional robustness')
ax.set_ylabel('CDF')

plt.show()