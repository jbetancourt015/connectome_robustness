"""
    This script puts together the FlyWire datasets into a single master 
    DataFrame with the relevant information. 
    
    In the original dataset, each connection had an associated neuropil. Now we 
    assign a neuropil to each neuron as the neuropil at which it has the 
    maximum number of incoming synapses.
-------------------------------------------------------------------------------
created on:
    Tue 4 Nov 2025
-------------------------------------------------------------------------------
last change:
    Tue 4 Nov 2025
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

#------------------------------------------------------------------------------
# LOAD DATASETS
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

conn_file = 'flywire_connections.csv.gz'
types_file = 'flywire_consolidated_cell_types.csv.gz'
class_file = 'flywire_classification.csv.gz'

# Load dataset into pandas DataFrame
conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')
types_df = pd.read_csv(data_dir + types_file, compression='gzip')
class_df = pd.read_csv(data_dir + class_file, compression='gzip')

#------------------------------------------------------------------------------
# BUILD NEURON-LEVEL DATASET
#------------------------------------------------------------------------------
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
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
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
neuron_df.to_parquet(data_dir+'neuron_data.parquet')

#------------------------------------------------------------------------------
# BUILD CONNECTIONS-LEVEL DATASET
#------------------------------------------------------------------------------
# Aggregate connections over neuropils
conn_df = conn_df.groupby(['pre_root_id','post_root_id'])['syn_count'].sum().reset_index()

# Append neuron data
neuron_vars = ['neuropil','brain_region','primary_type']

for var in neuron_vars:
    var_map = neuron_df.set_index('root_id')[var]
    conn_df['pre_'+var] = conn_df['pre_root_id'].map(var_map)
    conn_df['post_'+var] = conn_df['post_root_id'].map(var_map)

# Save dataset as parquet
conn_df.to_parquet(data_dir+'connections_data.parquet')