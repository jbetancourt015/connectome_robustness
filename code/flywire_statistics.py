"""
    This script computes and plots connection strength statistics for brain
    regions and neuropils of interest based on their sesitivity
-------------------------------------------------------------------------------
created on:
    Mon 2 Jul 2025
-------------------------------------------------------------------------------
last change:
    Tue 5 Aug 2025
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
import network_functions
import matplotlib.pyplot as plt
import pickle
import logging

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

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 3.5
height = 3.2


connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

data_idx = 5

#------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

fname = 'flywire_connections.csv.gz'
fname_thresh = 'flywire_connections_thresholded.csv.gz'

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
# ASSIGN BRAIN REGION TO EACH CONNECTION
#------------------------------------------------------------------------------
# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir + fname, compression='gzip')
df_thresh = pd.read_csv(data_dir + fname_thresh, compression='gzip')

# Load brain region map
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)
    
# Get brain region statistics
# NOTE: here I'm not coarse-graining neurons into regions first
df['brain_region'] = df['neuropil'].map(region_map)
df_thresh['brain_region'] = df_thresh['neuropil'].map(region_map)

#------------------------------------------------------------------------------
# GET SIMPLE CONNECTOME NETWORK STATISTICS
#------------------------------------------------------------------------------
def get_simple_statistics(df, *, normalize=False, include_zero=False, all_nodes=None):
    """
    df columns: ['pre_root_id','post_root_id','n_synapses']
    Returns dict with empirical distributions (counts or probabilities)
    for out_degree, in_degree, out_strength, in_strength.
    
    normalize=True   -> return probabilities instead of counts
    include_zero=True -> include zero-degree/zero-strength for nodes in all_nodes
    all_nodes        -> iterable of all node ids (needed to include zeros)
    """

    # 1) (Optional, but robust) collapse duplicate edges to keep one row per (pre,post)
    g = (df.groupby(['pre_root_id', 'post_root_id'], observed=True, sort=False)['syn_count']
           .sum()
           .rename('syn_count')
           .reset_index())

    # 2) Degrees = number of distinct partners
    out_degree = g.groupby('pre_root_id', observed=True)['post_root_id'].nunique()
    in_degree  = g.groupby('post_root_id', observed=True)['pre_root_id'].nunique()

    # 3) Strengths = sum of weights on incident edges
    out_strength = g.groupby('pre_root_id', observed=True)['syn_count'].sum()
    in_strength  = g.groupby('post_root_id', observed=True)['syn_count'].sum()

    # 4) Optionally add zeros for isolated (or in-/out-isolated) nodes
    if include_zero and all_nodes is not None:
        all_nodes = pd.Index(all_nodes)
        out_degree = out_degree.reindex(all_nodes, fill_value=0)
        in_degree  = in_degree.reindex(all_nodes, fill_value=0)
        out_strength = out_strength.reindex(all_nodes, fill_value=0)
        in_strength  = in_strength.reindex(all_nodes, fill_value=0)

    # 5) Empirical distributions (exact counts of each value)
    # Degrees are nonnegative ints -> use fast value_counts()
    out_deg_dist = out_degree.value_counts(normalize=normalize).sort_index()
    in_deg_dist  = in_degree.value_counts(normalize=normalize).sort_index()

    # Strengths are usually nonnegative ints in this dataset; if they are,
    # value_counts is exact. (If they were floats, consider binning.)
    out_str_dist = out_strength.value_counts(normalize=normalize).sort_index()
    in_str_dist  = in_strength.value_counts(normalize=normalize).sort_index()

    return {
        'out_degree_dist': out_deg_dist,
        'in_degree_dist':  in_deg_dist,
        'out_strength_dist': out_str_dist,
        'in_strength_dist':  in_str_dist,
        # (optionally also return the per-node series if you want them)
        'per_node': {
            'out_degree': out_degree,
            'in_degree':  in_degree,
            'out_strength': out_strength,
            'in_strength':  in_strength,
        }
    }

# Get statistics
simple_stats = get_simple_statistics(df, normalize=True)
simple_stats_thresh = get_simple_statistics(df_thresh, normalize=True)

# FIGURE: IN AND OUT DEGREE DISTRIBUTIONS--------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(simple_stats['out_degree_dist'].index, simple_stats['out_degree_dist'], color=con_colors[0], label='Out-degree', rasterized=True)
ax.scatter(simple_stats['in_degree_dist'].index, simple_stats['in_degree_dist'], color=con_colors[1], label='In-degree', rasterized=True)
ax.legend()
ax.set_xlabel('Degree')
ax.set_ylabel('Probability')
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
plt.savefig('../../figures/flywire_statistics/degree_distribution.pdf', dpi=600, bbox_inches='tight')

plt.show()

# FIGURE: IN AND OUT STRENGTH DISTRIBUTIONS------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(simple_stats['out_strength_dist'].index, simple_stats['out_strength_dist'], color=con_colors[0], label='Out-strength', rasterized=True)
ax.scatter(simple_stats['in_strength_dist'].index, simple_stats['in_strength_dist'], color=con_colors[1], label='In-strength', rasterized=True)
ax.legend()
ax.set_xlabel('Strength')
ax.set_ylabel('Probability')
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
plt.savefig('../../figures/flywire_statistics/strength_distribution.pdf', dpi=600, bbox_inches='tight')

plt.show()

# FIGURE: 1-CDF OF IN-DEGREE DISTRIBUTION--------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(simple_stats['in_degree_dist'].index, 1-np.cumsum(simple_stats['in_degree_dist']), color=con_colors[0], label='Unthresholded', rasterized=True)
ax.scatter(simple_stats_thresh['in_degree_dist'].index, 1-np.cumsum(simple_stats_thresh['in_degree_dist']), color=con_colors[1], label='Thresholded', rasterized=True)
ax.legend()
ax.set_xlabel('In-degree')
ax.set_ylabel('Survival Function')
ax.set_xscale('log')
ax.set_yscale('log')

# Add labels
plt.savefig('../../figures/flywire_statistics/in_degree_distribution_thresh.pdf', dpi=600, bbox_inches='tight')

plt.show()


# #------------------------------------------------------------------------------
# # GET STATISTICS FOR BRAIN REGION
# #------------------------------------------------------------------------------
# # Load brain region order
# region_order = np.load('../processed_data/brain_region_order.npy', allow_pickle=True)
# n_regions = len(region_order)

# cmap = plt.get_cmap('viridis', len(region_order))
# region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# # Get statistics by brain region
# counts = [len(df[df['brain_region']==region]) for region in region_order]
# means = [df[df['brain_region']==region]['syn_count'].mean() for region in region_order]
# stds = [df[df['brain_region']==region]['syn_count'].std() for region in region_order]

# # Generate a sub-dataframe for each brain region
# groups = {
#     region: df[df['brain_region'] == region] for region in region_order
# }

# # Compute neuropils per region
# counts  = [len(groups[reg]) for reg in region_order]
# group_space = 2


# # Make plots
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(range(n_regions), means, yerr=stds, capsize=5, color=region_colors.values())
# ax.set_xticks(range(n_regions))
# ax.set_xticklabels(region_order, rotation=45, ha='right')
# ax.set_ylabel('Connection strength')
# plt.tight_layout()
# plt.savefig('../../figures/flywire/conn_strength_region.pdf', dpi=600, bbox_inches='tight')
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(range(n_regions), counts, capsize=5, color=region_colors.values())
# ax.set_xticks(range(n_regions))
# ax.set_xticklabels(region_order, rotation=45, ha='right')
# ax.set_ylabel('Number of connections')
# ax.set_yscale('log')
# plt.tight_layout()
# plt.savefig('../../figures/flywire/neuron_counts_region.pdf', dpi=600, bbox_inches='tight')
# plt.show()

# #------------------------------------------------------------------------------
# # CONNECTION STRENGTH DISTRIBUTION BY BRAIN REGION
# #------------------------------------------------------------------------------
# cmap = plt.get_cmap('viridis', len(region_order))
# region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# for region in region_order:
#     # Get weights in brain region
#     weights = (
#         df[df['brain_region']==region]
#         .groupby(['pre_root_id','post_root_id'])['syn_count']
#         .sum()
#         .values
#     )
#     # Get histograms
#     s, P = empirical_hist(weights)
#     # Plot histogram
#     plt.scatter(s, P, color=region_colors[region], rasterized=True)

# plt.xlabel('Connection strength')
# plt.ylabel('Probability')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('../../figures/flywire_statistics/brain_region_weight_dist.pdf', dpi=600, bbox_inches='tight')
# plt.show()

# # Make the same plot in reverse
# for region in np.flip(region_order):
#     # Get weights in brain region
#     weights = (
#         df[df['brain_region']==region]
#         .groupby(['pre_root_id','post_root_id'])['syn_count']
#         .sum()
#         .values
#     )
#     # Get histograms
#     s, P = empirical_hist(weights)
#     # Plot histogram
#     plt.scatter(s, P, color=region_colors[region], rasterized=True)

# plt.xlabel('Connection strength')
# plt.ylabel('Probability')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('../../figures/flywire_statistics/brain_region_weight_dist_reversed.pdf', dpi=600, bbox_inches='tight')
# plt.show()

# #------------------------------------------------------------------------------
# # CONNECTION STRENGTH DISTRIBUTION BY BRAIN REGION (BY ASSIGNED NEURONS)
# #------------------------------------------------------------------------------
# # Assign a neuropil to each neuron
# neuropil_df = (
#     df
#     .groupby(['post_root_id','neuropil'])['syn_count']
#     .sum()
#     .reset_index()
#     .sort_values(['post_root_id','syn_count'], ascending=[True, False])
#     .drop_duplicates('post_root_id')
#     .loc[:, ['post_root_id','neuropil']]
# )

# categorized_df =(
#     df[['pre_root_id', 'post_root_id', 'syn_count']].merge(neuropil_df, on='post_root_id')
# )

# # Add brain regions of post neuron
# categorized_df['brain_region'] = categorized_df['neuropil'].map(region_map)

# # Get statistics by brain region
# counts = [len(categorized_df[categorized_df['brain_region']==region]) for region in region_order]
# means = [categorized_df[categorized_df['brain_region']==region]['syn_count'].mean() for region in region_order]
# stds = [categorized_df[categorized_df['brain_region']==region]['syn_count'].std() for region in region_order]

# # Generate a sub-dataframe for each brain region
# groups = {
#     region: categorized_df[categorized_df['brain_region'] == region] for region in region_order
# }

# for region in region_order:
#     # Get weights in brain region
#     weights = (
#         categorized_df[categorized_df['brain_region']==region]
#         .groupby(['pre_root_id','post_root_id'])['syn_count']
#         .sum()
#         .values
#     )
#     # Get histograms
#     s, P = empirical_hist(weights)
#     # Plot histogram
#     plt.scatter(s, P, color=region_colors[region], rasterized=True)

# plt.xlabel('Connection strength')
# plt.ylabel('Probability')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('../../figures/flywire_statistics/target_neuron_brain_region_weight_dist.pdf', dpi=600, bbox_inches='tight')
# plt.show()





