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

# Load dataset into pandas DataFrame
conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')
types_df = pd.read_csv(data_dir + types_file, compression='gzip')

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

# Compute robustness and normalized robustness
weights = (
    conn_df
    .groupby(['pre_root_id','post_root_id'])['syn_count']
    .sum()
    .rename('weight')
    .reset_index()
)

# Compute Q values
robustness_df = (
    weights
    .groupby('post_root_id')
    .agg(
        in_deg = ('pre_root_id', 'nunique'),
        in_strength  = ('weight', 'sum'),
        sum_w2 = ('weight', lambda s: (s**2).sum())
    )
    .assign(
        robustness = lambda d: np.sqrt((d['sum_w2'])/d['in_strength']),
        norm_robustness = lambda d: np.sqrt((d['in_deg'] * d['sum_w2'])/d['in_strength']**2)
    )
    .reset_index()
)

robustness_df = robustness_df.rename(columns={'post_root_id': 'root_id'})

# Append brain region
with open('../processed_data/brain_region_map.pkl', 'rb') as f:
    region_map = pickle.load(f)

neuron_df['brain_region'] = neuron_df['neuropil'].map(region_map)

# Append neuron robustness and primary type
neuron_df = neuron_df.merge(robustness_df, on='root_id')
neuron_df = neuron_df.merge(types_df, on='root_id').drop('additional_type(s)',axis=1)

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