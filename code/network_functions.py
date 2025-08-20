"""
    This script implements auxiliary functions for analysis of connectome data
    and computation sensitivity analysis.
-------------------------------------------------------------------------------
created on:
    Tue 28 May 2024
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
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, csc_matrix
from numba import njit

#------------------------------------------------------------------------------
# IMPORTING DATA
#------------------------------------------------------------------------------
processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

def load_connectome(data_idx):
    A = load_npz('%s%s.npz'%(processed_dir, connectomes[data_idx]))
    return A

#------------------------------------------------------------------------------
# SENSITIVITY CALCULATION
#------------------------------------------------------------------------------
def compute_sensitivity(A, scheme='constant', normalized=True):
    # Compute 0th moment
    k = A.getnnz(axis=0)
    # Get higher moments
    mask = k > 1
    s = [k[mask]]
    An = A
    for n in range(1,4):
        s.append(np.array(An.sum(axis=0)).ravel()[mask])
        An = An.multiply(A)
    # Compute sensitivities
    if scheme == 'constant':
        Q = 0.5*(s[0] - 1)/s[2]
        if normalized:
            Q /= (0.5*s[0]*(s[0]-1)/s[1]**2)
    else:  # proportional
        Q = -0.5*(s[3] - s[1]*s[2])/(s[2]**2)
        if normalized:
            Q /= (0.5*(s[0]-1)/s[1])
    return np.array(Q)

#------------------------------------------------------------------------------
# NULL NETWORK GENERATION
#------------------------------------------------------------------------------
# @njit
def null_network(A, scheme='rand_weight', conn_type='disc'):
    indptr, indices, data = A.indptr, A.indices, A.data
    n = A.shape[0]
    # Create new data array
    M_data = np.empty_like(data)
    # Loop over neurons
    for col in range(n):
        start, end = indptr[col], indptr[col+1]
        # Total in-degree and strength
        k = end - start
        if k == 0: continue
        strength = 0.
        for j in range(start, end):
            strength += data[j]
        # Sample new weights
        if scheme == 'rand_weight':
            if conn_type == 'cont':
                # Sample form the L-dimensional simplex
                wts_ext = np.zeros(int(k)+1)
                r = np.random.rand(int(k)-1)
                r = np.sort(r)
                wts_ext[1:-1] = r
                wts_ext[-1] = 1.
                # Get connection strengths
                wts = np.diff(wts_ext)*strength
            else:
                # Sample weight distribution uniformly
                rand_ints = np.random.randint(0, int(strength-k) + 1, size=int(k)-1)
                rand_ints = np.append(rand_ints, [0, int(strength-k)])
                rand_ints.sort()
                wts = 1.+np.diff(rand_ints)
        else:
            if conn_type == 'cont': # Still no clear analogue of Poisson for continuous weights
                # Sample form the L-dimensional simplex
                wts_ext = np.zeros(int(k)+1)
                r = np.random.rand(int(k)-1)
                r = np.sort(r)
                wts_ext[1:-1] = r
                wts_ext[-1] = 1.
                # Get connection strengths
                wts = np.diff(wts_ext)*strength
            else:
                # Sample weight distribution uniformly
                rand_ints = np.random.poisson(strength/k - 1., size=int(k))
                wts = 1.+rand_ints
        # Store to data matrix
        M_data[start:end] = wts
    # Build sparse matrix
    A_null = csc_matrix((M_data, indices, indptr))
    return A_null







# OLD FUNCTIONS

def largest_wcc(G):
    # Find all weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(G))
    # Identify the largest weakly connected component
    largest_wcc = max(weakly_connected_components, key=len)
    # Create a subgraph from the largest weakly connected component
    largest_wcc_subgraph = G.subgraph(largest_wcc).copy()
    return largest_wcc_subgraph


def edge_weight_list(A):
    # NOTE: this can be optimized with networkx directly
    N = A.shape[0]
    # Construct non-symmetrized edgelist and conductance values
    pairs_asym = []
    g_vals_asym = []
    for i in range(N):
        for j in range(N):
            if A[i,j] > 0:
                pairs_asym.append(np.array([i,j]))
                g_vals_asym.append(A[i,j])
    # Create edge and weight arrays
    pairs_asym = np.array(pairs_asym)
    g_vals_asym = np.array(g_vals_asym)
    edges_asym = len(pairs_asym)
    return pairs_asym, g_vals_asym

#------------------------------------------------------------------------------
# GENERATION OF NULL NETWORKS
#------------------------------------------------------------------------------
@njit
def random_weight(A, data_idx):
    N = A.shape[0]
    # Get statistics
    M = np.copy(A)
    in_deg = np.sum(A>0, axis=0)
    in_strength = np.sum(A, axis=0)
    # Go through each neuron
    for i in range(N):
        if in_deg[i] > 0:
            if data_idx == 4:
                # Sample form the L-dimensional simplex
                g_ext = np.zeros(int(in_deg[i])+1)
                r = np.random.rand(int(in_deg[i])-1)
                r = np.sort(r)
                g_ext[1:-1] = r
                g_ext[-1] = 1.
                # Get connection strengths
                g = np.diff(g_ext)*in_strength[i]
            else:
                # Sample weight distribution uniformly
                rand_ints = np.random.randint(0, int(in_strength[i]-in_deg[i]) + 1, size=int(in_deg[i])-1)
                rand_ints = np.append(rand_ints, [0, int(in_strength[i]-in_deg[i])])
                rand_ints.sort()
                g = 1.+np.diff(rand_ints)
            # Construct matrix
            idx = 0
            for j in range(N):
                if A[j,i] > 0:
                    M[j,i] = g[idx]
                    idx += 1
    return M


@njit
def poisson_weight(A, data_idx):
    N = A.shape[0]
    # Get statistics
    M = np.copy(A)
    in_deg = np.sum(A>0, axis=0)
    in_strength = np.sum(A, axis=0)
    # Go through each neuron
    for i in range(N):
        if in_deg[i] > 0:
            if data_idx == 4:
                # Sample form the L-dimensional simplex
                g_ext = np.zeros(int(in_deg[i])+1)
                r = np.random.rand(int(in_deg[i])-1)
                r = np.sort(r)
                g_ext[1:-1] = r
                g_ext[-1] = 1.
                # Get connection strengths
                g = np.diff(g_ext)*in_strength[i]
            else:
                # Sample weight distribution uniformly
                rand_ints = np.random.poisson(in_strength[i]/in_deg[i] - 1., size=int(in_deg[i]))
                g = 1.+rand_ints
            # Construct matrix
            idx = 0
            for j in range(N):
                if A[j,i] > 0:
                    M[j,i] = g[idx]
                    idx += 1
    return M


@njit
def random_synapse_placement(A):
    N = A.shape[0]
    # Construct adjacency matrix
    M = np.zeros((N,N))
    # Place one synapse in each position
    for l in range(edges_asym):
        M[pairs_asym[l,0], pairs_asym[l,1]] += 1.
    # Place each synapse individually
    for s in range(int(g_tot)-edges_asym):
        l = np.random.choice(np.arange(edges_asym))
        M[pairs_asym[l,0], pairs_asym[l,1]] += 1.
    return M


@njit
def random_topology(A):
    N = A.shape[0]
    # Create an NxN matrix of zeros
    M = np.zeros((N,N))
    # Get all the positions in the upper triangular part
    full_range = np.arange(N)
    positions = [(i,j) for i in range(N) for j in full_range[full_range != i]]
    indices = np.arange(len(positions))
    # Shuffle the position
    np.random.shuffle(indices)
    selected_positions = [positions[indices[i]] for i in range(edges_asym)]
    # Assign the values from L to the selected positions
    for value, (i,j) in zip(g_vals_asym, selected_positions):
        M[i,j] += value
    return M


@njit
def weight_shuffle(A):
    N = A.shape[0]
    # Create an NxN matrix of zeros
    M = np.zeros((N,N))
    # Shuffle the order of pairs
    new_pairs = np.copy(pairs_asym)
    np.random.shuffle(new_pairs)
    # Assign the values from L to the selected positions
    for value, (i,j) in zip(g_vals_asym, new_pairs):
        M[i,j] += value
    return M