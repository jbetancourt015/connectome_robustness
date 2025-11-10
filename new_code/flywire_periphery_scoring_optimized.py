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
import pandas as pd
import network_functions
from scipy.sparse import coo_matrix
from time import time
from tqdm import tqdm

#------------------------------------------------------------------------------
# TURN CONNECTOME TO SPARSE MATRIX
#------------------------------------------------------------------------------
# Load connectome
data_dir = '../../raw_data/'

conn_df = pd.read_parquet(data_dir+'connections_data.parquet')

# Get set of nodes
nodes_index = pd.Index(
    pd.unique(pd.concat([conn_df['pre_root_id'], conn_df['post_root_id']], ignore_index=True)),
    name="root_id"
)

id_to_idx = pd.Series(np.arange(nodes_index.size), index=nodes_index)
idx_to_id = nodes_index.to_numpy()

# Map edges to integer rows/cols
rows = id_to_idx.reindex(conn_df['pre_root_id']).to_numpy()
cols = id_to_idx.reindex(conn_df['post_root_id']).to_numpy()

data = conn_df['syn_count'].to_numpy()

# Create sparse matrix
A = coo_matrix((data, (rows, cols)),
                shape=(nodes_index.size, nodes_index.size)).tocsr()

#------------------------------------------------------------------------------
# SIMULATION FUNCTIONS
#------------------------------------------------------------------------------
def simulate_propagation(A, pool, threshold=0.3, repeats=3, seed=None, max_steps=None):
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
    no_growth_streak = 0
    
    while True:
        # print(f'{step}: {np.sum(in_pool)}')
        # All current pool nodes try this round
        sources = np.nonzero(in_pool)[0]
        if sources.size == 0:
            break  # degenerate, but safe

        sub = A[sources, :].tocoo()
        if sub.nnz == 0:
            # No outgoing edges from pool; count as no growth
            no_growth_streak += 1
            if no_growth_streak >= repeats:
                break
            step += 1
            if max_steps is not None and step > max_steps:
                break
            continue

        cols = sub.col
        w = sub.data

        # Edge-wise success probabilities
        p = np.divide(w, denom[cols], out=np.zeros_like(w, dtype=float), where=np.isfinite(denom[cols]))
        p = np.clip(p, 0.0, 1.0)

        # Bernoulli trials
        successes = rng.random(p.size) < p
        hit_cols = np.unique(cols[successes])

        # Newly activated nodes this round
        new_nodes = hit_cols[~in_pool[hit_cols]]

        if new_nodes.size > 0:
            time_added[new_nodes] = step
            in_pool[new_nodes] = True
            no_growth_streak = 0
        else:
            no_growth_streak += 1

        if no_growth_streak >= repeats:
            break

        step += 1
        if max_steps is not None and step > max_steps:
            break

        # Early exit if everything is already in the pool
        if in_pool.all():
            break

    return time_added


def average_propagation(A, pool, n_sim=100, threshold=0.3, repeats=3, seed=None, max_steps=None):
    """
    Run simulate_propagation_all_pool() n_sim times and return the average
    time_added array (ignoring -1 entries).

    Parameters
    ----------
    A : sparse matrix
    pool : list[int]
    n_sim : int
        Number of Monte Carlo runs.
    threshold, repeats, seed, max_steps : passed to simulate_propagation_all_pool

    Returns
    -------
    avg_time : np.ndarray (float)
        Average time_added across runs (excluding never-reached nodes).
        Nodes never reached in all runs remain np.nan.
    frac_reached : np.ndarray (float)
        Fraction of runs where each node was reached.
    """
    n = A.shape[0]
    avg_time = np.zeros(n, dtype=float)
    count_reached = np.zeros(n, dtype=int)

    base_rng = np.random.default_rng(seed)

    for s in tqdm(range(n_sim)):
        run_seed = base_rng.integers(1e9)
        t = simulate_propagation(A, pool, threshold, repeats, seed=run_seed, max_steps=max_steps)
        reached = (t >= 0)
        count_reached += reached
        # online mean for numerical stability
        avg_time[reached] += (t[reached] - avg_time[reached]) / count_reached[reached]

    # Convert to NaN where node was never reached
    avg_time[count_reached == 0] = np.nan
    frac_reached = count_reached / n_sim
    return avg_time, frac_reached

#------------------------------------------------------------------------------
# RUN SIMULATION
#------------------------------------------------------------------------------
n_sim = 100
rng_seed = 1764
k_vals = [5,10,20]

# Build output DataFrame
sim_df = pd.DataFrame({'root_id': idx_to_id})

for k_threshold in k_vals:
    # Define set of seed neurons
    mask = np.array(A.sum(axis=0) < k_threshold).flatten()
    
    seed_set = [i for i,val in enumerate(mask) if val]
    
    # Run simulation
    avg_dist, frac_reached = average_propagation(A, seed_set, n_sim, seed=rng_seed)
    
    # Store in DataFrame
    sim_df[f"distance_{k_threshold}"] = avg_dist
    
# Save dataset as parquet
sim_df.to_parquet(data_dir+'periphery_data.parquet')