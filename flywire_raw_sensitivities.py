# Exploration of FlyWire dataset
import numpy as np
import pandas as pd
import network_functions
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "font.serif": ["Garamond"],
    "text.latex.preamble": r'\usepackage{amsfonts}'
})

processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

data_idx = 5

# Load connectome
A = network_functions.load_connectome(data_idx)

#------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../raw_data/'
processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

file_names = ['Drosophila_central_brain.csv','Drosophila_optic_medulla.csv','Celegans.csv',
              'Platynereis_sensory_motor.csv', 'Mouse_retina.csv', 'FlyWire.csv.gz']

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE
#------------------------------------------------------------------------------
data_idx = 5

# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir + file_names[data_idx], compression='gzip')

# Aggregate weights over different neuropils
weights = (
    df
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .rename('weight')
    .reset_index()
)

# Compute Q values
Q_df = (
    weights
    .groupby('post_root_id')
    .agg(
        in_deg = ('pre_root_id', 'nunique'),
        sum_w  = ('weight', 'sum'),
        sum_w2 = ('weight', lambda s: (s**2).sum())
    )
    .assign(
        Q = lambda d: d['in_deg'] / d['sum_w2']
    )
    .reset_index()[['post_root_id','Q']]
)

# Assign a neuropil to each neuron
neuropil_df = (
    df
    .groupby(['post_root_id','neuropil'])['syn_count']
    .sum()
    .reset_index()
    .sort_values(['post_root_id','syn_count'], ascending=[True, False])
    .drop_duplicates('post_root_id')
    .loc[:, ['post_root_id','neuropil']]
)

# Merge neuropil and sensitivity into a single dataframe
collapsed = (
    Q_df
    .merge(neuropil_df, on='post_root_id')
    .loc[:, ['post_root_id','neuropil','Q']]
)

#------------------------------------------------------------------------------
# ANALYSIS BY NEUROPIL
#------------------------------------------------------------------------------
neuropil_df = (
    collapsed
    .groupby(['neuropil'])
    .agg(
        Q_mean = ('Q','mean'),
        Q_std = ('Q', 'std'),
        n_neurons  = ('post_root_id', 'nunique')
        )
    .reset_index()
)

#------------------------------------------------------------------------------
# ANALYSIS BY BRAIN REGION
#------------------------------------------------------------------------------
# Load brain region map
with open('../processed_data/region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)

# Expand dataframe with brain region
collapsed['brain_region'] = collapsed['neuropil'].map(region_map)
collapsed = collapsed[collapsed['brain_region']!='Other Regions']

# Compute sensitivities by region
region_df = (
    collapsed
    .groupby(['brain_region'])
    .agg(
        Q_mean = ('Q','mean'),
        Q_std = ('Q', 'std'),
        n_neurons  = ('post_root_id', 'nunique')
        )
    .reset_index()
)

# Sort regions by average sensitivity
region_sorted = region_df.sort_values('Q_mean')

n_regions = region_df['brain_region'].nunique()

region_order = (
    collapsed
    .groupby('brain_region')['Q']
    .mean()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# Get data for plots
labels = region_sorted['brain_region']
means  = region_sorted['Q_mean']
stds   = region_sorted['Q_std']

# Make plots
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(n_regions), means, yerr=stds, capsize=5, color=region_colors.values())
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Sensitivity')
plt.tight_layout()
plt.savefig('../figures/flywire/raw_sensitivity_bars_region.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# ANALYSIS BY NEUROPIL - COLORED BY BRAIN REGION
#------------------------------------------------------------------------------
# Expand neuropil df to include brain_region
neuropil_df['brain_region'] = neuropil_df['neuropil'].map(region_map)
neuropil_df = neuropil_df[neuropil_df['brain_region']!='Other Regions']

# Sort by neuropil Q values
neuropil_sorted = neuropil_df.sort_values('Q_mean').reset_index(drop=True)
neuropil_colors = neuropil_sorted['brain_region'].map(region_colors)

# Set up plot
fig, ax = plt.subplots(figsize=(12,6))
ax.bar(
    range(len(neuropil_sorted)),
    neuropil_sorted['Q_mean'],
    yerr=neuropil_sorted['Q_std'],
    capsize=3,
    color=neuropil_colors
)
ax.set_xticks([])                    # no x‐labels (too cluttered)
ax.set_ylabel('Sensitivity')
ax.set_xlabel('Neuropils')

plt.tight_layout()
plt.savefig('../figures/flywire/raw_sensitivity_bars_neuropil.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# ANALYSIS BY NEUROPIL - COLORED BY BRAIN REGION
#------------------------------------------------------------------------------
# Generate a sub-dataframe for each brain region
groups = {
    region: neuropil_df[neuropil_df['brain_region'] == region]
               .sort_values('Q_mean')
    for region in region_order
}

# Compute neuropils per region
counts  = [len(groups[reg]) for reg in region_order]
group_space = 2

# Compute offsets
offsets = np.array([
    sum(counts[:i]) + i*group_space
    for i in range(len(counts))
])

# Make the figure
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.8

for i, region in enumerate(region_order):
    df_reg = groups[region]
    x = offsets[i] + np.arange(len(df_reg))
    ax.bar(
        x,
        df_reg['Q_mean'],
        yerr=df_reg['Q_std'],
        width=bar_width,
        capsize=3,
        color=region_colors[region]
    )

# 4) Label one tick per region, at the center of its group
centers = offsets + (np.array(counts) - 1) / 2
ax.set_xticks(centers)
ax.set_xticklabels(region_order, rotation=45, ha='right')

# 5) Tidy up
ax.set_ylabel('Sensitivity')
ax.set_xlabel('Brain Region')
plt.tight_layout()
plt.savefig('../figures/flywire/raw_sensitivity_bars_neuropil_region.pdf', dpi=600, bbox_inches='tight')
plt.show()
