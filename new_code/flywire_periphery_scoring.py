"""
    This script scores all neurons in terms of the distance to some peripheral
    set of neurons.
-------------------------------------------------------------------------------
created on:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
last change:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
    Varun:
        name:       Varun Varanasi
        email:      varunvaranasi@g.harvard.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import network_functions
# from scipy.sparse import 

data_idx = 5

# Get connectome
A = network_functions.load_connectome(data_idx)
A = A.tocsr()

# Define set of seed neurons
k_threshold = 10
mask = np.array(A.sum(axis=0) < k_threshold).flatten()

seed_idx = [i for i,val in enumerate(mask) if val]

#------------------------------------------------------------------------------
# SIMULATION FUNCTIONS
#------------------------------------------------------------------------------
def initialize_pool(idx_list):
    pool = {
    # neuron : traversal number
    }
    for i in idx_list:
        pool[i] = 0
    return pool


def iterate_info_flow(A, pool, k, remaining_neurons, total_weights_in):
    if not remaining_neurons:
        return pool, remaining_neurons
    
    remaining_array = np.array(list(remaining_neurons))
    pool_array = np.array(list(pool.keys()))
    
    # Keep sparse - extract rows for pool neurons only
    pool_rows = A[pool_array]
    
    selected_neurons = []
    weights_scaled = total_weights_in[remaining_array] * 0.3
    
    # Process each remaining neuron to avoid dense matrix creation
    for i, neuron_idx in enumerate(remaining_array):
        # Get column for this neuron (stays sparse)
        col = pool_rows[:, neuron_idx]
        if col.nnz > 0:  # Only process if there are connections
            transition_probs = col.data / weights_scaled[i]
            # Test all connections at once
            if np.any(np.random.rand(len(transition_probs)) < transition_probs):
                selected_neurons.append(neuron_idx)
    
    # Update pool and remaining
    for neuron in selected_neurons:
        pool[neuron] = k
    remaining_neurons -= set(selected_neurons)
    
    return pool, remaining_neurons


def information_flow(A, seed, total_weights_in, repeats=1):

    pool = initialize_pool(seed)
    remaining_neurons = set(range(A.shape[0])) - set(pool.keys())
    len_pool_prev = 0
    print(f'{0}: {len(pool)}')
    k = 1
    t = 0

    while len(pool) != A.shape[0]:
        len_pool_prev = len(pool)
        pool, remaining_neurons = iterate_info_flow(A, pool, k, remaining_neurons, total_weights_in)
        print(f'{k}: {len(pool)}')
        #print(pool)
        k+=1 
        #print(f'{k}: {len(pool)}')
        if len(pool) == len_pool_prev:
            t += 1 
            if t > repeats:
                break
        else:
            t = 0

    return pool

#------------------------------------------------------------------------------
# RUN SIMULATION
#------------------------------------------------------------------------------
total_weights_in = np.array(A.astype(np.float32).sum(axis=0)).flatten() 
pool = information_flow(A.astype(np.float32), seed_idx, total_weights_in)