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
# DISTIRBUTION OF SAMPLED WEIGHTS
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

#------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../raw_data/'
processed_dir = '../processed_data/'

conn_file = 'flywire_connections.csv.gz'
types_file = 'flywire_consolidated_cell_types.csv.gz'

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE
#------------------------------------------------------------------------------
# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir + conn_file, compression='gzip')

# Add neurotransmitter type
nt_to_class = {
    'ACH': 'exc',
    'GLUT': 'inh',
    'GABA': 'inh',
    'DA':  'mod',
    'SER': 'mod',
    'OCT': 'mod'
}

df['nt_class'] = df['nt_type'].map(nt_to_class)

def collapsed_df(df):
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
            Q = lambda d: d['sum_w']**2 / (d['in_deg'] * d['sum_w2'])
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
    
    return collapsed

collapsed_exc = collapsed_df(df[df['nt_class']=='exc'])
collapsed_inh = collapsed_df(df[df['nt_class']=='inh'])

#------------------------------------------------------------------------------
# ANALYSIS BY BRAIN REGION
#------------------------------------------------------------------------------
# Load brain region map
with open('../processed_data/region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)


def Q_by_region(collapsed):
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
    
    # Get data for plots
    labels = region_sorted['brain_region']
    means  = region_sorted['Q_mean']
    stds   = region_sorted['Q_std']
    
    return labels, means, stds

labels_exc, means_exc, stds_exc = Q_by_region(collapsed_exc)
labels_inh, means_inh, stds_inh = Q_by_region(collapsed_inh)

region_order = np.load('../processed_data/brain_region_order.npy', allow_pickle=True)
n_regions = len(region_order)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# Make plots
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(n_regions), means_exc, yerr=stds_exc, capsize=5, color=region_colors.values())
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels_exc, rotation=45, ha='right')
ax.set_ylabel('Normalized sensitivity')
plt.tight_layout()
# plt.savefig('../figures/flywire/norm_sensitivity_bars_region.pdf', dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(n_regions), means_inh, yerr=stds_inh, capsize=5, color=region_colors.values())
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels_inh, rotation=45, ha='right')
ax.set_ylabel('Normalized sensitivity')
plt.tight_layout()
# plt.savefig('../figures/flywire/norm_sensitivity_bars_region.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# DISTRIBUTION OF WEIGHTS BY NT TYPE
#------------------------------------------------------------------------------
# Get excitatory and inhibitory weights
exc_weights = (
    df[df['nt_class']=='exc']
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .values
)

inh_weights = (
    df[df['nt_class']=='inh']
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .values
)

# Get histograms
s_exc, P_exc = empirical_hist(exc_weights)
s_inh, P_inh = empirical_hist(inh_weights)

plt.scatter(s_exc, P_exc, color=con_colors[0], rasterized=True, label='Excitatory')
plt.scatter(s_inh, P_inh, color=con_colors[1], rasterized=True, label='Inhibitory')
plt.legend()
plt.xlabel('Connection strength')
plt.ylabel('Probability')
plt.xscale('log')
plt.yscale('log')
plt.savefig('../figures/flywire/nt_type_weight_dist.pdf', dpi=600, bbox_inches='tight')
plt.show()
