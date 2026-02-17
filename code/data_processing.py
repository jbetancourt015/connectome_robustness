"""
    This script processes the raw connectome data and generates all processed
    datasets (sparse matrices and parquet files) used by the figure-generation
    scripts.
-------------------------------------------------------------------------------
created on:
    Mon 12 May 2024
-------------------------------------------------------------------------------
last change:
    Sun 16 Feb 2026
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
import networkx as nx
import pickle
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from time import time
from tqdm import tqdm
import pyarrow.feather as feather
import io
import bz2

#------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc', 'drosophila_manc']

file_names = ['Drosophila_central_brain.csv','Drosophila_optic_medulla.csv','Celegans.csv',
              'Platynereis_sensory_motor.csv', 'Mouse_retina.csv', 'flywire_connections.csv.gz',
              'banc_connections.csv.gz', 'manc_connections.csv']

#------------------------------------------------------------------------------
# PROCESSING SMALL CONNECTOMES
#------------------------------------------------------------------------------
for data_idx in range(5):
    # Load dataset into pandas DataFrame
    colnames = ['pre', 'post', 'strength']
    df = pd.read_csv(data_dir + file_names[data_idx], names=colnames, 
                     header=None)

    # Get unique node IDs
    all_nodes = pd.unique(df[['pre','post']].values.ravel())
    N = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Apply mapping to row/col arrays
    rows = df['pre'].map(node_to_idx).values
    cols = df['post'].map(node_to_idx).values
    data = df['strength'].values

    # Create sparse matrix using SciPy
    A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
    A_csc = A_coo.tocsc()

    # Save it to disk
    save_npz(processed_dir + '%s.npz'%(connectomes[data_idx]), A_csc)

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE CONNECTOMES
#------------------------------------------------------------------------------
for data_idx in [5,6]:
    # Load dataset into pandas DataFrame
    file_name = file_names[data_idx]
    df = pd.read_csv(data_dir+file_name, compression='gzip')
    
    # Sum over neuropils
    weights = (
        df
        .groupby(['pre_root_id','post_root_id'])['syn_count']
        .sum()
        .reset_index()
    )
    
    # Get unique node IDs
    all_nodes = pd.unique(weights[['pre_root_id','post_root_id']].values.ravel())
    N = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    
    # Apply mapping to row/col arrays
    rows = weights['pre_root_id'].map(node_to_idx).values
    cols = weights['post_root_id'].map(node_to_idx).values
    data = weights['syn_count'].values
    
    # Create sparse matrix using SciPy
    A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
    A_csc = A_coo.tocsc()
    
    # Save it to disk
    save_npz(processed_dir + f"{connectomes[data_idx]}.npz", A_csc)

#------------------------------------------------------------------------------
# PROCESSING MANC CONNECTOME
#------------------------------------------------------------------------------
data_idx = 7
file_name = file_names[data_idx]

# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir + file_name)

# Get unique node IDs
all_nodes = pd.unique(df[['bodyId_pre','bodyId_post']].values.ravel())
N = len(all_nodes)
node_to_idx = {node: i for i, node in enumerate(all_nodes)}

# Apply mapping to row/col arrays
rows = df['bodyId_pre'].map(node_to_idx).values
cols = df['bodyId_post'].map(node_to_idx).values
data = df['weight'].values

# Create sparse matrix using SciPy
A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
A_csc = A_coo.tocsc()

# Save it to disk
save_npz(processed_dir + f"{connectomes[data_idx]}.npz", A_csc)

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE - THRESHOLDED
#------------------------------------------------------------------------------
data_idx = 5
file_name = 'flywire_connections_thresholded.csv.gz'

# Load dataset into pandas DataFrame
df = pd.read_csv(data_dir+file_name, compression='gzip')

# Sum over neuropils
weights = (
    df
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .reset_index()
)

# Get unique node IDs
all_nodes = pd.unique(weights[['pre_root_id','post_root_id']].values.ravel())
N = len(all_nodes)
node_to_idx = {node: i for i, node in enumerate(all_nodes)}

# Apply mapping to row/col arrays
rows = weights['pre_root_id'].map(node_to_idx).values
cols = weights['post_root_id'].map(node_to_idx).values
data = weights['syn_count'].values

# Create sparse matrix using SciPy
A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
A_csc = A_coo.tocsc()

# Save it to disk
save_npz(processed_dir + f"{connectomes[data_idx]}_thresholded.npz", A_csc)

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE NEURON-LEVEL DATASET
#------------------------------------------------------------------------------
# Load FlyWire datasets
conn_file = 'flywire_connections.csv.gz'
types_file = 'flywire_consolidated_cell_types.csv.gz'
class_file = 'flywire_classification.csv.gz'

conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')
types_df = pd.read_csv(data_dir + types_file, compression='gzip')
class_df = pd.read_csv(data_dir + class_file, compression='gzip')

# Get neuropils based on incoming connections
neuron_df = (
    conn_df
    .groupby(['post_root_id','neuropil'])['syn_count']
    .sum()
    .reset_index()
    .sort_values(['post_root_id','syn_count'], ascending=[True, False])
    .drop_duplicates('post_root_id')
    .loc[:, ['post_root_id','neuropil']]
)

neuron_df = neuron_df.rename(columns={'post_root_id': 'root_id'})

# Compute aggregate weights
weights = (
    conn_df
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .rename('weight')
    .reset_index()
)

# Compute statistics of incoming connections
in_stats = (
    weights
    .groupby('post_root_id')
    .agg(
        in_deg = ('pre_root_id', 'nunique'),
        in_strength  = ('weight', 'sum'),
        sum_w2 = ('weight', lambda s: (s**2).sum())
    )
    .assign(
        robustness = lambda d: np.sqrt(d['sum_w2']/d['in_strength']),
        norm_robustness = lambda d: (np.sqrt(d['sum_w2']/d['in_strength']) - np.sqrt(d['in_strength']/d['in_deg']))/(np.sqrt(1. + (d['in_strength']/d['in_deg'])) - np.sqrt(d['in_strength']/d['in_deg'])),
        scaled_robustness = lambda d: np.sqrt((d['in_deg'] * d['sum_w2'])/d['in_strength']**2)-1.
    )
    .reset_index()
    .rename(columns={'post_root_id': 'root_id'})
)

# Compute statistics of outgoing connections
out_stats = (
    weights
    .groupby('pre_root_id')
    .agg(
        out_deg      = ('post_root_id', 'nunique'),
        out_strength = ('weight', 'sum')
    )
    .reset_index()
    .rename(columns={'pre_root_id': 'root_id'})
)

# Compute reciprocity of neurons
in_neighbors = (
    weights
    .groupby('post_root_id')['pre_root_id']
    .agg(lambda x: set(x))
    .reset_index()
    .rename(columns={'post_root_id': 'root_id', 'pre_root_id': 'in_neighbors'})
)

out_neighbors = (
    weights
    .groupby('pre_root_id')['post_root_id']
    .agg(lambda x: set(x))
    .reset_index()
    .rename(columns={'pre_root_id': 'root_id', 'post_root_id': 'out_neighbors'})
)

recip = in_neighbors.merge(out_neighbors, on='root_id', how='outer')

recip['in_neighbors'] = recip['in_neighbors'].apply(
    lambda s: s if isinstance(s, set) else set()
)
recip['out_neighbors'] = recip['out_neighbors'].apply(
    lambda s: s if isinstance(s, set) else set()
)

def _reciprocity(row):
    inter = row['in_neighbors'] & row['out_neighbors']
    union = row['in_neighbors'] | row['out_neighbors']
    return len(inter) / len(union) if len(union) > 0 else 0.0  # 0 if completely isolated

recip['reciprocity'] = recip.apply(_reciprocity, axis=1)
recip = recip[['root_id', 'reciprocity']]

# Put all neuron-level stats together
node_stats = (
    in_stats
    .merge(out_stats, on='root_id', how='outer')
    .merge(recip,    on='root_id', how='outer')
)

# Append brain region
with open(data_dir + 'brain_region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)

neuron_df['brain_region'] = neuron_df['neuropil'].map(region_map)

# Append neuron robustness and primary type
neuron_df = neuron_df.merge(node_stats, on='root_id')
neuron_df = neuron_df.merge(types_df, on='root_id').drop('additional_type(s)',axis=1)
neuron_df = neuron_df.merge(class_df, on='root_id')

# Set network statistics to 0 if there are no neighbors
for col in ['in_deg', 'in_strength', 'out_deg', 'out_strength']:
    neuron_df[col] = neuron_df[col].fillna(0)

# Save dataset as parquet
neuron_df.to_parquet(processed_dir+'neuron_data.parquet')

# Sort regions by median (excluding Other Regions)
neuron_df_regions = neuron_df[neuron_df['brain_region'] != "Other Regions"].copy()

region_order = (
    neuron_df_regions
    .groupby('brain_region')['norm_robustness']
    .median()
    .sort_values()
    .index
)

with open(processed_dir + 'region_order.pkl', 'wb') as f:
    pickle.dump(region_order, f)

#------------------------------------------------------------------------------
# PROCESSING FLYWIRE CONNECTIONS-LEVEL DATASET
#------------------------------------------------------------------------------
# Classify connections into excitatory and inhibitory
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

# Step 1: aggregate synapses by connection × neurotransmitter
nt_df = (
    conn_df.groupby(["pre_root_id", "post_root_id", "nt_class"])["syn_count"]
      .sum()
      .reset_index()
)

# Step 2: pivot to wide format (one column per neurotransmitter)
wide_df = nt_df.pivot_table(
    index=["pre_root_id", "post_root_id"],
    columns="nt_class",
    values="syn_count",
    fill_value=0
)

# Step 3: compute total synapses per connection
wide_df["total_syn"] = wide_df.sum(axis=1)

# Step 4: compute fractions
for col in wide_df.columns:
    if col != "total_syn":
        wide_df[f"frac_{col}"] = wide_df[col] / wide_df["total_syn"]

conn_class_df = wide_df.reset_index()

# Set threshold for classification
thresh = 0.6

# Classify neurons as excitatory/inhibitory
conn_class_df['is_exc'] = conn_class_df['frac_exc'] >= thresh
conn_class_df['is_inh'] = conn_class_df['frac_inh'] >= thresh

def classify_row(row):
    inh = row['is_inh']
    exc = row['is_exc']

    if not inh and not exc:
        return np.nan
    
    if inh and not exc:
        return 'inh'
    if exc and not inh:
        return 'exc'

conn_class_df['nt_class'] = conn_class_df.apply(classify_row, axis=1)
conn_class_df = conn_class_df[['pre_root_id', 'post_root_id', 'nt_class']]

# Map neuropil to brain_region
conn_df['brain_region'] = conn_df['neuropil'].map(region_map)

# Aggregate by connection and brain_region
conn_by_region = (
    conn_df
    .groupby(['pre_root_id', 'post_root_id', 'brain_region'])['syn_count']
    .sum()
    .reset_index()
)

# Find brain_region with max synapses for each connection (plurality)
plurality_region = (
    conn_by_region
    .sort_values(['pre_root_id', 'post_root_id', 'syn_count'], ascending=[True, True, False])
    .drop_duplicates(['pre_root_id', 'post_root_id'])
    .loc[:, ['pre_root_id', 'post_root_id', 'brain_region']]
    .rename(columns={'brain_region': 'conn_brain_region'})
)

# Compute total synapse count per connection
total_syn = (
    conn_by_region
    .groupby(['pre_root_id', 'post_root_id'])['syn_count']
    .sum()
    .reset_index()
)

# Merge to get connection dataset with plurality brain_region
conn_df = total_syn.merge(plurality_region, on=['pre_root_id', 'post_root_id'])

# Append neurotransmitter type data
conn_df = conn_df.merge(conn_class_df, on=('pre_root_id', 'post_root_id'), how='outer')

# Append neuron data
neuron_vars = ['neuropil','brain_region','primary_type']

for var in neuron_vars:
    var_map = neuron_df.set_index('root_id')[var]
    conn_df['pre_'+var] = conn_df['pre_root_id'].map(var_map)
    conn_df['post_'+var] = conn_df['post_root_id'].map(var_map)

# Save dataset as parquet
conn_df.to_parquet(processed_dir+'connections_data.parquet')
