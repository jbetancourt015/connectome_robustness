"""
    This script generates four robustness plots for the FlyWire dataset:
    1. CDF of normalized robustness
    2. Violin plots of normalized robustness by brain region
    3. CDF of excitatory vs inhibitory normalized robustness
    4. CDF of normalized robustness by reciprocity decile
-------------------------------------------------------------------------------
created on:
    Tue 3 Feb 2026
-------------------------------------------------------------------------------
last change:
    Tue 3 Feb 2026
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

# Nature figure size
mm_to_in = 25.4
width = 50./mm_to_in
height = 50./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    """Compute CDF from a pandas series."""
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# LOAD DATASETS
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron and connections data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

# Add peripherality data to neuron dataframe
neuron_df = neuron_df.merge(peri_df, on="root_id")

#------------------------------------------------------------------------------
# PLOT 1: CDF OF NORMALIZED ROBUSTNESS
#------------------------------------------------------------------------------
rob_vals, rob_cdf = compute_cdf(neuron_df['norm_robustness'])
median_rob = neuron_df['norm_robustness'].median()

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.step(rob_vals, rob_cdf, where='post', lw=2, color=con_colors[1])
ax.plot([median_rob, median_rob], [0., 1.], lw=1, c='k', ls='--')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/flywire_robustness/cdf_normalized_robustness.pdf', dpi=600)
plt.show()

#------------------------------------------------------------------------------
# PLOT 2: VIOLIN PLOTS BY BRAIN REGION
#------------------------------------------------------------------------------
# Drop "Other Regions"
neuron_df_regions = neuron_df[neuron_df['brain_region'] != "Other Regions"].copy()

# Sort regions by median
region_order = (
    neuron_df_regions
    .groupby('brain_region')['norm_robustness']
    .median()
    .sort_values()
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# Get data for violin plots
violin_data = [neuron_df_regions.loc[neuron_df_regions["brain_region"] == r, "norm_robustness"].to_numpy(dtype=float) for r in region_order]

# Set up figure (wider for region labels)
fig, ax = plt.subplots(figsize=(1.9*width, height))

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=True,
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
    parts['cmedians'].set_color('black')
    
# Labels/ticks
ax.set_ylabel("Normalized Robustness")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")

plt.tight_layout()
plt.savefig('../../paper_figures/flywire_robustness/violin_by_brain_region.pdf', dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# PLOT 3: CDF OF EXCITATORY VS INHIBITORY NEURONS
#------------------------------------------------------------------------------
# Map of neurotransmitter types
nt_to_class = {
    'ACH': 'exc',
    'GLUT': 'inh',
    'GABA': 'inh',
    'DA':  'mod',
    'SER': 'mod',
    'OCT': 'mod'
}

# Add neurotransmitter class to connections dataset
conn_df['nt_class'] = conn_df['nt_type'].map(nt_to_class)

# Get index of all neurons
all_neurons = pd.Index(
    pd.unique(
        conn_df[['pre_root_id', 'post_root_id']].values.ravel('K')
    ),
    name='root_id'
)

# Get composition of outgoing synapses
outgoing = (
    conn_df
    .groupby(['pre_root_id', 'nt_class'], as_index=False)['syn_count']
    .sum()
    .pivot(index='pre_root_id', columns='nt_class', values='syn_count')
)

outgoing = outgoing.reindex(columns=['exc', 'inh', 'mod'], fill_value=0)
outgoing_tot = outgoing.sum(axis=1)

frac_out = pd.DataFrame(index=outgoing.index)
frac_out['frac_out_exc'] = outgoing['exc'] / outgoing_tot
frac_out['frac_out_inh'] = outgoing['inh'] / outgoing_tot
frac_out = frac_out.reindex(all_neurons)

# Create synapse composition dataframe
syn_comp = (
    pd.DataFrame(index=all_neurons)
    .join(frac_out)
    .reset_index()
)

# Set threshold for classification
thresh = 0.6

# Classify neurons as excitatory/inhibitory
syn_comp['is_out_exc'] = syn_comp['frac_out_exc'] >= thresh
syn_comp['is_out_inh'] = syn_comp['frac_out_inh'] >= thresh

def classify_row(row):
    inh = row['is_out_inh']
    exc = row['is_out_exc']

    if not inh and not exc:
        return np.nan
    
    if inh and not exc:
        return 'inh'
    if exc and not inh:
        return 'exc'

syn_comp['out_class'] = syn_comp.apply(classify_row, axis=1)

# Merge with neuron-level data
neuron_df_class = neuron_df.merge(syn_comp, how='outer', on='root_id')

# Get CDFs of inhibitory and excitatory neurons
mask_exc = neuron_df_class['out_class'] == 'exc'
mask_inh = neuron_df_class['out_class'] == 'inh'

rob_exc, cdf_exc = compute_cdf(neuron_df_class[mask_exc]['norm_robustness'])
rob_inh, cdf_inh = compute_cdf(neuron_df_class[mask_inh]['norm_robustness'])

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.step(rob_exc, cdf_exc, where='post', color=con_colors[0], lw=2, label='Excitatory')
ax.step(rob_inh, cdf_inh, where='post', color=con_colors[1], lw=2, label='Inhibitory')

ax.legend()

ax.set_xscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/flywire_robustness/cdf_exc_vs_inh.pdf', dpi=600)
plt.show()

#------------------------------------------------------------------------------
# PLOT 4: CDF BY RECIPROCITY DECILE
#------------------------------------------------------------------------------
n_quantiles = 10

# Add deciles to dataset
neuron_df['reciprocity_q'] = pd.qcut(neuron_df['reciprocity'], q=n_quantiles, labels=False)

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))
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

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/flywire_robustness/cdf_by_reciprocity_decile.pdf', dpi=600)
plt.show()

