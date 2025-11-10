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
from scipy.sparse import csr_matrix

data_idx = 5

# Get connectome
A = network_functions.load_connectome(data_idx)
A = A.tocsr()

# Define set of seed neurons
k_threshold = 10
mask = np.array(A.sum(axis=0) < k_threshold).flatten()

seed_set = [i for i,val in enumerate(mask) if val]

#------------------------------------------------------------------------------
# SIMULATION FUNCTIONS
#------------------------------------------------------------------------------
def simulate_propagation(A, pool, threshold=0.3, seed=None, max_steps=None):
    """
    Simulate stochastic propagation on a directed weighted sparse graph.

    Parameters
    ----------
    A : (n x n) sparse matrix (CSR preferred or convertible)
        A[i, j] = w_ij is the weight of edge i -> j.
    pool : array-like of ints
        Initial set of node indices in the pool at time 0.
    threshold : float, default 0.3
        Probability scaling: p_ij = clip( w_ij / (threshold * sum_k w_kj), 0, 1 ).
    seed : int or None
        Random seed for reproducibility.
    max_steps : int or None
        Optional cap on the number of propagation steps.

    Returns
    -------
    time_added : np.ndarray of shape (n,), dtype=int
        time_added[j] is the step when node j first joined the pool (0 for initial pool).
        Nodes never reached remain -1.
    """
    # Ensure CSR for fast row slicing
    A = A.tocsr()
    n = A.shape[0]
    pool = np.asarray(pool, dtype=int)

    # Column sums: sum_k w_kj
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    # Denominator for probabilities; if column sum == 0, no incoming mass => set denom to inf
    denom = np.where(col_sums > 0, threshold * col_sums, np.inf)

    # State
    time_added = -np.zeros(n, dtype=int) - 1
    in_pool = np.zeros(n, dtype=bool)
    time_added[pool] = 0
    in_pool[pool] = True
    frontier = np.unique(pool)

    rng = np.random.default_rng(seed)
    step = 1

    while frontier.size > 0:
        print(f'{step}: {np.sum(in_pool)}')
        # Submatrix of edges from current frontier to all targets
        sub = A[frontier, :].tocoo()  # (row, col, data) sparse view

        if sub.nnz == 0:
            break

        cols = sub.col
        w = sub.data

        # Edge-wise success probabilities
        p = np.divide(w, denom[cols], out=np.zeros_like(w, dtype=float), where=np.isfinite(denom[cols]))
        # Clip to [0, 1]
        p = np.clip(p, 0.0, 1.0)

        # Bernoulli trials on all edges from frontier
        successes = rng.random(p.size) < p

        # Any successful incoming edge activates the target
        hit_cols = cols[successes]
        if hit_cols.size == 0:
            break

        candidates = np.unique(hit_cols)
        new_nodes = candidates[~in_pool[candidates]]

        if new_nodes.size == 0:
            break

        time_added[new_nodes] = step
        in_pool[new_nodes] = True
        frontier = new_nodes

        step += 1
        if max_steps is not None and step > max_steps:
            break

    return time_added

#------------------------------------------------------------------------------
# RUN SIMULATION
#------------------------------------------------------------------------------
pool_2 = simulate_propagation(A.astype(np.float32), seed_set)