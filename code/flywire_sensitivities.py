"""
    This script computes statistics of the sensitivity of neurons by neuropil
    and brain region in the FlyWire dataset.
-------------------------------------------------------------------------------
created on:
    Tue 3 Jun 2024
-------------------------------------------------------------------------------
last change:
    Tue 9 Sep 2025
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

processed_dir = '../processed_data/'

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
# PROCESSING FLYWIRE
#------------------------------------------------------------------------------
thresholded = True

suffix = '_thresholded' if thresholded else ''

# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir+'flywire_connections'+suffix+'.csv.gz', compression='gzip')

# Aggregate weights over different neuropils
weights = (
    df
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .rename('weight')
    .reset_index()
)

if thresholded:
    # weights['weight'] = 5*(weights['weight']//5)
    weights['weight'] = weights['weight']-4

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

plt.scatter(neuropil_df['Q_mean'], neuropil_df['Q_std'], color=con_colors[0], rasterized=True)
plt.xlabel('Average sensitivity')
plt.ylabel('Sensitivity std. dev.')
# plt.gca().set_aspect('equal')
plt.savefig('../../figures/flywire/sensitivity_by_neuropil.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# ASYMMETRY IN THE NEUROPIL
#------------------------------------------------------------------------------
# Retreive the neuropils that are located on the left or right
asym_df = neuropil_df[(neuropil_df['neuropil'].str.endswith(('_L','_R')))].copy()

# Split columns into left and right
asym_df['structure'] = asym_df['neuropil'].str.rsplit('_', n=1).str[0]
asym_df['side']      = asym_df['neuropil'].str.rsplit('_', n=1).str[1]

# Make new df that has one observation per structure
asym_df = (
    asym_df
    .pivot(index='structure', columns='side', values=['Q_mean','Q_std','n_neurons'])
)

# Flatten columns
asym_df.columns = [f"{stat.lower()}_{side.lower()}" 
                   for stat,side in asym_df.columns]
asym_df = asym_df.reset_index().rename(columns={'structure':'neuropil_region'})

#------------------------------------------------------------------------------
# PLOT LEFT VS RIGHT SENSITIVITY VALUES
#------------------------------------------------------------------------------
plt.scatter(asym_df['q_mean_r'], asym_df['q_mean_l'], color=con_colors[0], rasterized=True)
# Plot the y=x line
q_min, q_max = asym_df[['q_mean_r','q_mean_l']].min().min(), asym_df[['q_mean_r','q_mean_l']].max().max()
plt.plot([q_min,q_max], [q_min,q_max], lw=2, ls='--', c='k', alpha=.5, zorder=-1)
plt.xlabel('Average sensitivity (right)')
plt.ylabel('Average sensitivity (left)')
plt.gca().set_aspect('equal')
plt.savefig('../../figures/flywire/average_left_vs_right_neuropil.pdf', dpi=600, bbox_inches='tight')
plt.show()

plt.scatter(asym_df['q_std_r'], asym_df['q_std_l'], color=con_colors[0], rasterized=True)
# Plot the y=x line
std_min, std_max = asym_df[['q_std_r','q_std_l']].min().min(), asym_df[['q_std_r','q_std_l']].max().max()
plt.plot([std_min,std_max], [std_min,std_max], lw=2, ls='--', c='k', alpha=.5, zorder=-1)
plt.xlabel('Sensitivity std. dev. (right)')
plt.ylabel('Sensitivity std. dev. (left)')
plt.gca().set_aspect('equal')
plt.savefig('../../figures/flywire/std_left_vs_right_neuropil.pdf', dpi=600, bbox_inches='tight')
plt.show()

plt.scatter(asym_df['n_neurons_r'], asym_df['n_neurons_l'], color=con_colors[0], rasterized=True)
# Plot the y=x line
n_min, n_max = asym_df[['n_neurons_r','n_neurons_l']].min().min(), asym_df[['n_neurons_r','n_neurons_l']].max().max()
plt.plot([n_min,n_max], [n_min,n_max], lw=2, ls='--', c='k', alpha=.5, zorder=-1)
plt.xlabel('Number of neurons (right)')
plt.ylabel('Number of neurons (left)')
plt.gca().set_aspect('equal')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.savefig('../../figures/flywire/number_left_vs_right_neuropil.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# ANALYSIS BY BRAIN REGION
#------------------------------------------------------------------------------
# Load brain region map
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
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

np.save('../processed_data/brain_region_order', np.array(region_order))

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# Get data for plots
labels = region_sorted['brain_region']
means  = region_sorted['Q_mean']
stds   = region_sorted['Q_std']


# FIGURE: SENSITIVITY OF BRAIN REGIONS-----------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.bar(range(n_regions), means, yerr=stds, capsize=5, color=region_colors.values())
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels, rotation=45, ha='right')
plt.savefig('../../raw_figures/flywire/norm_sensitivity_bars_region.pdf', dpi=600)

# Add labels
ax.set_ylabel('Normalized sensitivity')
plt.savefig('../../figures/flywire/norm_sensitivity_bars_region.pdf', dpi=600, bbox_inches='tight')

plt.show()

# Code for violin plot

# Group Q values per region following the chosen order
violin_data = [collapsed.loc[collapsed["brain_region"] == r, "Q"].to_numpy(dtype=float) for r in region_order]

# FIGURE: SENSITIVITY VIOLINS BY BRAIN REGION----------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=False,
    showextrema=False,
    widths=0.9
)

# Color each violin body to match points
for i, body in enumerate(parts['bodies']):
    region = region_order[i]
    color = region_colors[region]
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color([region_colors[region] for region in parts['bodies']])

for i, vals in enumerate(violin_data, start=1):
    region = region_order[i-1]
    color = region_colors[region]

    # Custom median line (colored like the points)
    med = float(np.median(vals))
    ax.hlines(med, i-0.35, i+0.35, linewidth=2.8, color=color, alpha=0.95)
    
# Labels/ticks
ax.set_ylabel("Normalized sensitivity")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")
ax.set_ylim(0, 1)

# Add labels
plt.savefig('../../figures/flywire_sensitivities/sensitivity_violins_by_region.pdf', dpi=600, bbox_inches='tight')
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

# FIGURE: SENSITIVITY OF NEUROPILS---------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(12,6))
ax.bar(
    range(len(neuropil_sorted)),
    neuropil_sorted['Q_mean'],
    yerr=neuropil_sorted['Q_std'],
    capsize=3,
    color=neuropil_colors
)
ax.set_xticks([])                    # no x‐labels (too cluttered)

# Add labels
ax.set_ylabel('Normalized sensitivity')
ax.set_xlabel('Neuropils')
plt.savefig('../../figures/flywire/norm_sensitivity_bars_neuropil.pdf', dpi=600, bbox_inches='tight')

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

# FIGURE: SENSITIVITY OF NEUROPILS, GROUPED BY REGION--------------------------
# Set up figure
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

# Add labels
ax.set_ylabel('Normalized sensitivity')
plt.savefig('../../figures/flywire/norm_sensitivity_bars_neuropil_region.pdf', dpi=600, bbox_inches='tight')

plt.show()

#------------------------------------------------------------------------------
# ANALYSIS OF THE OPTIC LOBE
#------------------------------------------------------------------------------
# Keep only neurons in the optic lobe
optic_lobe_df = collapsed[collapsed['brain_region']=='Optic Lobe']
optic_lobe_df = optic_lobe_df[~optic_lobe_df['neuropil'].isin(['AME_L', 'AME_R'])]

# Load brain region map
with open('../processed_data/optic_lobe_region_map.pkl', 'rb') as f:
    optic_lobe_map = pickle.load(f)

# Map neurons to their optic lobe region
optic_lobe_df['lobe_region'] = optic_lobe_df['neuropil'].map(optic_lobe_map)

# Compute sensitivities by region
lobe_region_sorted = (
    optic_lobe_df
    .groupby(['lobe_region'])
    .agg(
        Q_mean = ('Q','mean'),
        Q_std = ('Q', 'std'),
        n_neurons  = ('post_root_id', 'nunique')
        )
    .reset_index()
)

# Sort regions by average sensitivity
lobe_region_sorted = lobe_region_sorted.sort_values('Q_mean')

n_regions = lobe_region_sorted['lobe_region'].nunique()

lobe_region_order = (
    optic_lobe_df
    .groupby('lobe_region')['Q']
    .mean()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

np.save('../processed_data/lobe_region_order', np.array(lobe_region_order))

cmap = plt.get_cmap('viridis', len(lobe_region_order))
region_colors = {r: cmap(i) for i, r in enumerate(lobe_region_order)}

# Get data for plots
labels = lobe_region_sorted['lobe_region']
means  = lobe_region_sorted['Q_mean']
stds   = lobe_region_sorted['Q_std']

# FIGURE: SENSITIVITY OF OPTIC LOBE REGIONS------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.bar(range(n_regions), means, yerr=stds, capsize=5, color=region_colors.values())
ax.set_xticks(range(n_regions))
ax.set_xticklabels(labels, rotation=45, ha='right')
plt.savefig('../../raw_figures/flywire/norm_sensitivity_bars_lobe.pdf', dpi=600)

# Add labels
ax.set_ylabel('Normalized sensitivity')
plt.savefig('../../figures/flywire/norm_sensitivity_bars_lobe.pdf', dpi=600, bbox_inches='tight')

plt.show()