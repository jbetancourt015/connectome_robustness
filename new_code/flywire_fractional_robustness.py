"""
    This script computes the fractional robustness for excitatory and 
    inhibitory synapses in the FlyWire connectome
-------------------------------------------------------------------------------
created on:
    Mon 10 Nov 2025
-------------------------------------------------------------------------------
last change:
    Mon 11 Nov 2025
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
import pickle
import logging
from tqdm import tqdm

plt.rcParams.update({
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

#------------------------------------------------------------------------------
# COMPUTE FRACTIONAL ROBUSTNESS
#------------------------------------------------------------------------------
# Import neuron data
conn_df = pd.read_csv(data_dir+'flywire_connections.csv.gz', compression='gzip')

# Map of neurotransmitter types
nt_to_class = {
    'ACH': 'exc',
    'GLUT': 'inh',
    'GABA': 'inh',
    'DA':  'mod',
    'SER': 'mod',
    'OCT': 'mod'
}

conn_df['nt_class'] = conn_df['nt_type'].map(nt_to_class)

def compute_subnetwork_robustness(mask):
    # Get constrained dataframe
    const_df = conn_df[mask]
    # Compute weights
    weights = (
        const_df
        .groupby(['pre_root_id','post_root_id'])['syn_count']
        .sum()
        .rename('weight')
        .reset_index()
    )
    # Compute normalized robustness
    robustness_df = (
        weights
        .groupby('post_root_id')
        .agg(
            in_deg = ('pre_root_id', 'nunique'),
            in_strength  = ('weight', 'sum'),
            sum_w2 = ('weight', lambda s: (s**2).sum())
        )
        .assign(
            norm_robustness = lambda d: np.sqrt((d['in_deg'] * d['sum_w2'])/d['in_strength']**2)
        )
        .reset_index()
    )
    return robustness_df['norm_robustness'].to_numpy(dtype=float)

#------------------------------------------------------------------------------
# PLOT SUBNETWORK ROBUSTNESS BY NT TYPE
#------------------------------------------------------------------------------
violin_data = []

for nt in tqdm(nt_to_class):
    norm_robustness = compute_subnetwork_robustness(conn_df['nt_type']==nt)
    violin_data.append(norm_robustness)
    
# Make violin plot
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

cmap = plt.get_cmap('viridis', len(nt_to_class))

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body to match points
for i, body in enumerate(parts['bodies']):
    color = cmap(i)
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')
    
# Labels/ticks
ax.set_ylabel("Normalized robustness")
ax.set_xticks(range(1, len(nt_to_class) + 1))
ax.set_xticklabels(nt_to_class.keys(), rotation=35, ha="right")
ax.set_ylim(None, 4.1)

plt.show()

#------------------------------------------------------------------------------
# PLOT SUBNETWORK ROBUSTNESS BY BRAIN REGION
#------------------------------------------------------------------------------
# Load brain region map
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)

# Add brain region to connections
conn_df['brain_region'] = conn_df['neuropil'].map(region_map)
region_order = np.load('../processed_data/brain_region_order.npy', allow_pickle=True)

# Obtain data for subnetworks
violin_data = []

for region in tqdm(region_order):
    norm_robustness = compute_subnetwork_robustness(conn_df['brain_region']==region)
    violin_data.append(norm_robustness)

# Make violin plot
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

cmap = plt.get_cmap('viridis', len(region_order))

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body to match points
for i, body in enumerate(parts['bodies']):
    color = cmap(i)
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')
    
# Labels/ticks
ax.set_ylabel("Normalized robustness")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")
ax.set_ylim(.9, 2.6)

plt.show()







