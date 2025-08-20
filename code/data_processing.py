"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
-------------------------------------------------------------------------------
created on:
    Mon 12 May 2024
-------------------------------------------------------------------------------
last change:
    Mon 12 May 2025
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

#------------------------------------------------------------------------------
# CONNECTOMES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../raw_data/'
processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

file_names = ['Drosophila_central_brain.csv','Drosophila_optic_medulla.csv','Celegans.csv',
              'Platynereis_sensory_motor.csv', 'Mouse_retina.csv', 'flywire_connections.csv.gz']

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
# PROCESSING FLYWIRE
#------------------------------------------------------------------------------
data_idx = 5

# Load dataset into pandas DataFrame
df = pd.read_csv(
    data_dir + file_names[data_idx],
    usecols=['pre_root_id','post_root_id','syn_count'],
    dtype={'pre_root_id':np.int32,'post_root_id':np.int32,'syn_count':np.float32},
    compression='gzip'
)

# Get unique node IDs
all_nodes = pd.unique(df[['pre_root_id','post_root_id']].values.ravel())
N = len(all_nodes)
node_to_idx = {node: i for i, node in enumerate(all_nodes)}

# Apply mapping to row/col arrays
rows = df['pre_root_id'].map(node_to_idx).values
cols = df['post_root_id'].map(node_to_idx).values
data = df['syn_count'].values

# Create sparse matrix using SciPy
A_coo = coo_matrix((data, (rows, cols)), shape=(N, N))
A_csc = A_coo.tocsc()

# Save it to disk
save_npz(processed_dir + '%s.npz'%(connectomes[data_idx]), A_csc)
