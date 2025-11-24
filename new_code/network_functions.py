"""
    This script implements auxiliary functions for analysis of connectome data
    and computation sensitivity analysis.
-------------------------------------------------------------------------------
created on:
    Tue 28 May 2024
-------------------------------------------------------------------------------
last change:
    Tue 2 Sep 2025
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
from scipy.sparse import load_npz, csc_matrix
from numba import njit

#------------------------------------------------------------------------------
# IMPORTING DATA
#------------------------------------------------------------------------------
processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc']

def load_connectome(data_idx, thresholded=False, scheme=None):
    if not thresholded:
        A = load_npz(f"{processed_dir}{connectomes[data_idx]}.npz")
    else:
        A = load_npz(f"{processed_dir}{connectomes[data_idx]}_thresholded.npz")
        if scheme == 'remove':
            A.data = A.data - 4
        elif scheme == 'clump':
            A.data = 5*(A.data//5)
    return A

#------------------------------------------------------------------------------
# SENSITIVITY CALCULATION
#------------------------------------------------------------------------------
def compute_sensitivity(A, eta=1., normalized=True):
    # Compute 0th moment
    k = A.getnnz(axis=0)
    # Get higher moments
    mask = k > 1
    
    # Get moments of columns
    def col_moment(p, mask):
        if p == 1:
            return np.asarray(A.sum(axis=0)).ravel()[mask]
        else:
            B = A.copy()
            B.data = np.power(B.data, p)
            return np.asarray(B.sum(axis=0)).ravel()[mask]
        
    # Compute moments
    s_0 = k[mask]
    s_1 = col_moment(1., mask)
    s_2 = col_moment(2., mask)
    s_eta = col_moment(eta, mask)
    
    # Compute sensitivities
    Q = s_eta/s_2
    if normalized:
        Q *= (s_1/s_0)**(2-eta)
    return Q

def compute_robustness(A, eta=1., normalized=True):
    # Compute 0th moment
    k = A.getnnz(axis=0)
    # Get higher moments
    mask = k > 1
    
    # Get moments of columns
    def col_moment(p, mask):
        if p == 1:
            return np.asarray(A.sum(axis=0)).ravel()[mask]
        else:
            B = A.copy()
            B.data = np.power(B.data, p)
            return np.asarray(B.sum(axis=0)).ravel()[mask]
        
    # Compute moments
    s_0 = k[mask]
    s_1 = col_moment(1., mask)
    s_2 = col_moment(2., mask)
    s_eta = col_moment(eta, mask)
    
    # Compute sensitivities
    Q = (s_2/s_eta)**0.5
    if normalized:
        Q *= (s_0/s_1)**(1.-eta/2)
        Q -= 1.
    return Q

#------------------------------------------------------------------------------
# NULL NETWORK GENERATION
#------------------------------------------------------------------------------
# @njit
def null_network(A, scheme='rand_weight', conn_type='disc', n_threshold=1):
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
                rand_ints = np.random.randint(0, int(strength-n_threshold*k) + 1, size=int(k)-1)
                rand_ints = np.append(rand_ints, [0, int(strength-n_threshold*k)])
                rand_ints.sort()
                wts = n_threshold+np.diff(rand_ints)
        elif scheme=='poisson':
            # Sample weight distribution uniformly
            rand_ints = np.random.poisson(strength/k - 1., size=int(k))
            wts = 1.+rand_ints
        elif scheme=='concentrated':
            # 
            wts = np.ones(int(k))
            wts[0] = strength-k+1
        else:
            print('Error')
        # Store to data matrix
        M_data[start:end] = wts
    # Build sparse matrix
    A_null = csc_matrix((M_data, indices, indptr))
    return A_null







# # OLD FUNCTIONS

# def largest_wcc(G):
#     # Find all weakly connected components
#     weakly_connected_components = list(nx.weakly_connected_components(G))
#     # Identify the largest weakly connected component
#     largest_wcc = max(weakly_connected_components, key=len)
#     # Create a subgraph from the largest weakly connected component
#     largest_wcc_subgraph = G.subgraph(largest_wcc).copy()
#     return largest_wcc_subgraph


# def edge_weight_list(A):
#     # NOTE: this can be optimized with networkx directly
#     N = A.shape[0]
#     # Construct non-symmetrized edgelist and conductance values
#     pairs_asym = []
#     g_vals_asym = []
#     for i in range(N):
#         for j in range(N):
#             if A[i,j] > 0:
#                 pairs_asym.append(np.array([i,j]))
#                 g_vals_asym.append(A[i,j])
#     # Create edge and weight arrays
#     pairs_asym = np.array(pairs_asym)
#     g_vals_asym = np.array(g_vals_asym)
#     edges_asym = len(pairs_asym)
#     return pairs_asym, g_vals_asym

# #------------------------------------------------------------------------------
# # GENERATION OF NULL NETWORKS
# #------------------------------------------------------------------------------
# @njit
# def random_weight(A, data_idx):
#     N = A.shape[0]
#     # Get statistics
#     M = np.copy(A)
#     in_deg = np.sum(A>0, axis=0)
#     in_strength = np.sum(A, axis=0)
#     # Go through each neuron
#     for i in range(N):
#         if in_deg[i] > 0:
#             if data_idx == 4:
#                 # Sample form the L-dimensional simplex
#                 g_ext = np.zeros(int(in_deg[i])+1)
#                 r = np.random.rand(int(in_deg[i])-1)
#                 r = np.sort(r)
#                 g_ext[1:-1] = r
#                 g_ext[-1] = 1.
#                 # Get connection strengths
#                 g = np.diff(g_ext)*in_strength[i]
#             else:
#                 # Sample weight distribution uniformly
#                 rand_ints = np.random.randint(0, int(in_strength[i]-in_deg[i]) + 1, size=int(in_deg[i])-1)
#                 rand_ints = np.append(rand_ints, [0, int(in_strength[i]-in_deg[i])])
#                 rand_ints.sort()
#                 g = 1.+np.diff(rand_ints)
#             # Construct matrix
#             idx = 0
#             for j in range(N):
#                 if A[j,i] > 0:
#                     M[j,i] = g[idx]
#                     idx += 1
#     return M


# @njit
# def poisson_weight(A, data_idx):
#     N = A.shape[0]
#     # Get statistics
#     M = np.copy(A)
#     in_deg = np.sum(A>0, axis=0)
#     in_strength = np.sum(A, axis=0)
#     # Go through each neuron
#     for i in range(N):
#         if in_deg[i] > 0:
#             if data_idx == 4:
#                 # Sample form the L-dimensional simplex
#                 g_ext = np.zeros(int(in_deg[i])+1)
#                 r = np.random.rand(int(in_deg[i])-1)
#                 r = np.sort(r)
#                 g_ext[1:-1] = r
#                 g_ext[-1] = 1.
#                 # Get connection strengths
#                 g = np.diff(g_ext)*in_strength[i]
#             else:
#                 # Sample weight distribution uniformly
#                 rand_ints = np.random.poisson(in_strength[i]/in_deg[i] - 1., size=int(in_deg[i]))
#                 g = 1.+rand_ints
#             # Construct matrix
#             idx = 0
#             for j in range(N):
#                 if A[j,i] > 0:
#                     M[j,i] = g[idx]
#                     idx += 1
#     return M


# @njit
# def random_synapse_placement(A):
#     N = A.shape[0]
#     # Construct adjacency matrix
#     M = np.zeros((N,N))
#     # Place one synapse in each position
#     for l in range(edges_asym):
#         M[pairs_asym[l,0], pairs_asym[l,1]] += 1.
#     # Place each synapse individually
#     for s in range(int(g_tot)-edges_asym):
#         l = np.random.choice(np.arange(edges_asym))
#         M[pairs_asym[l,0], pairs_asym[l,1]] += 1.
#     return M


# @njit
# def random_topology(A):
#     N = A.shape[0]
#     # Create an NxN matrix of zeros
#     M = np.zeros((N,N))
#     # Get all the positions in the upper triangular part
#     full_range = np.arange(N)
#     positions = [(i,j) for i in range(N) for j in full_range[full_range != i]]
#     indices = np.arange(len(positions))
#     # Shuffle the position
#     np.random.shuffle(indices)
#     selected_positions = [positions[indices[i]] for i in range(edges_asym)]
#     # Assign the values from L to the selected positions
#     for value, (i,j) in zip(g_vals_asym, selected_positions):
#         M[i,j] += value
#     return M


# @njit
# def weight_shuffle(A):
#     N = A.shape[0]
#     # Create an NxN matrix of zeros
#     M = np.zeros((N,N))
#     # Shuffle the order of pairs
#     new_pairs = np.copy(pairs_asym)
#     np.random.shuffle(new_pairs)
#     # Assign the values from L to the selected positions
#     for value, (i,j) in zip(g_vals_asym, new_pairs):
#         M[i,j] += value
#     return M