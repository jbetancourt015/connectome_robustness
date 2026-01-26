"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
-------------------------------------------------------------------------------
created on:
    Mon 12 May 2024
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
import networkx as nx
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
