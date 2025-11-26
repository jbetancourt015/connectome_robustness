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

conn_file = 'flywire_connections.csv.gz'
types_file = 'flywire_consolidated_cell_types.csv.gz'

# Load dataset into pandas DataFrame
conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')
types_df = pd.read_csv(data_dir + types_file, compression='gzip')
neuron_df = pd.read_parquet(data_dir + 'neuron_data.parquet')

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
# Map of neurotransmitter types
nt_to_class = {
    'ACH': 'exc',
    'GLUT': 'inh',
    'GABA': 'inh',
    'DA':  'mod',
    'SER': 'mod',
    'OCT': 'mod'
}

# Add neurotransmitter to connections dataset
conn_df['nt_class'] = conn_df['nt_type'].map(nt_to_class)

# Get index of all neurons
all_neurons = pd.Index(
    pd.unique(
        conn_df[['pre_root_id', 'post_root_id']].values.ravel('K')
    ),
    name='root_id'
)

# Get composition of incoming synapses
incoming = (
    conn_df
    .groupby(['post_root_id', 'nt_class'], as_index=False)['syn_count']
    .sum()
    .pivot(index='post_root_id', columns='nt_class', values='syn_count')
)

incoming = incoming.reindex(columns=['exc', 'inh', 'mod'], fill_value=0)
incoming_tot = incoming.sum(axis=1)

frac_in = pd.DataFrame(index=incoming.index)
frac_in['frac_in_exc'] = incoming['exc'] / incoming_tot
frac_in['frac_in_inh'] = incoming['inh'] / incoming_tot

# Do the same with outgoing synapses
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


# --- 3. Combine into neuron-level dataframe ---
syn_comp = (
    pd.DataFrame(index=all_neurons)
    .join(frac_in)
    .join(frac_out)
    .reset_index()
)

#------------------------------------------------------------------------------
# COMPOSITION OF INCOMING AND OUTGOING SYNAPSES
#------------------------------------------------------------------------------
# FIGURE: Distribution of incoming composition---------------------------------
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

ax.hist(syn_comp['frac_in_exc'], bins=20, density=True, histtype='step', color=con_colors[0], label='Excitatory')
ax.hist(syn_comp['frac_in_inh'], bins=20, density=True, histtype='step', color=con_colors[1], label='Inhibitory')

ax.legend()

ax.set_xlabel('Fraction of incoming synapses')
ax.set_ylabel('Density')

plt.show()

# FIGURE: Distribution of outgoing composition---------------------------------
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

ax.hist(syn_comp['frac_out_exc'], bins=20, density=True, histtype='step', color=con_colors[0], label='Excitatory')
ax.hist(syn_comp['frac_out_inh'], bins=20, density=True, histtype='step', color=con_colors[1], label='Inhibitory')

ax.legend()

ax.set_xlabel('Fraction of outgoing synapses')
ax.set_ylabel('Density')

plt.show()

# FIGURE: Distribution of maximum between the two------------------------------
syn_comp['frac_out_max'] = syn_comp[['frac_out_exc', 'frac_out_inh']].apply(np.nanmax, axis=1)

fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

ax.hist(syn_comp['frac_out_max'], bins=20, density=True, histtype='step', color=con_colors[0])

ax.set_xlabel('Maximum of outgoing exc/inh fractions')
ax.set_ylabel('Density')

plt.show()

# FIGURE: Distribution of maximum between the two (CDF)------------------------
fracs, cdf = compute_cdf(syn_comp['frac_out_max'])

fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

ax.step(fracs, cdf, where='post', color=con_colors[0])

ax.set_yscale('log')

ax.set_xlabel('Maximum of outgoing exc/inh fractions')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# ROBUSTNESS BY NEURON TYPE
#------------------------------------------------------------------------------
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
neuron_df = neuron_df.merge(syn_comp, how='outer', on='root_id')

# Get CDFs of inhibitory and excitatory neurons
mask_exc = neuron_df['out_class'] == 'exc'
mask_inh = neuron_df['out_class'] == 'inh'

rob_exc, cdf_exc = compute_cdf(neuron_df[mask_exc]['norm_robustness'])
rob_inh, cdf_inh = compute_cdf(neuron_df[mask_inh]['norm_robustness'])

# FIGURE: Distribution of robustness by neuron type----------------------------
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(rob_exc, cdf_exc, where='post', color=con_colors[0], lw=2, label='Excitatory')
ax.step(rob_inh, cdf_inh, where='post', color=con_colors[1], lw=2, label='Inhibitory')

ax.legend()

ax.set_xscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()






