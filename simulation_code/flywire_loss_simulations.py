"""
    This script simulates and stores loss for the FlyWire FAFB dataset.
-------------------------------------------------------------------------------
created on:
    Tue 3 Jun 2024
-------------------------------------------------------------------------------
last change:
    Tue 11 Nov 2025
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
from tqdm import tqdm
from scipy.sparse import coo_matrix

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'
#------------------------------------------------------------------------------
# GET SPARSE MATRIX OF CONNECTIONS
#------------------------------------------------------------------------------
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')

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

A = A.tocsc()
N = A.shape[0]

#------------------------------------------------------------------------------
# OBTAIN SUBSET OF NEURONS
#------------------------------------------------------------------------------
incoming_weights = [
    A.data[A.indptr[j] : A.indptr[j+1]]
    for j in range(N)
]

#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
n_draws = int(1e3)          # number of input draws
n_perturb = int(1e3)        # number of weight perturbation draws

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def average_error_fast(
    w,
    eta,
    eps,
    n_draws,
    n_perturb,
    block_perturb=128,
    rng=None,
):
    """
    Monte Carlo estimate of the loss, but vectorized + blocked
    to reduce memory and speed things up.
    """
    if rng is None:
        rng = np.random.default_rng()

    w = np.asarray(w, dtype=float)
    n_inputs = w.size
    if n_inputs == 0:
        return np.nan

    # Draw inputs: n_inputs × n_draws
    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))

    # Draw base Gaussian noise: n_inputs × n_perturb
    base_noise = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb))
    # Scale by w**(eta/2)
    w_hat = base_noise * (w**(eta / 2.0))[:, None]

    # Baseline output: shape (n_draws,)
    z = w @ x

    total_pairs = n_draws * n_perturb
    error_sum = 0.0

    # Process perturbations in blocks to keep memory small
    for start in range(0, n_perturb, block_perturb):
        end = min(start + block_perturb, n_perturb)

        # (block_size × n_draws)
        delta = eps * (w_hat[:, start:end].T @ x)

        # Broadcast z over rows
        zztilde = z[None, :] * (z[None, :] + delta)

        # (1 - sign)/2 is 1 if sign flips, 0 otherwise
        error_sum += ((1.0 - np.sign(zztilde)) * 0.5).sum()

    return error_sum / total_pairs

#------------------------------------------------------------------------------
# LOSS BY NEURON STATISTICS
#------------------------------------------------------------------------------
eta = 1.0
eps = 1.0

n_neurons = len(incoming_weights)

loss = np.full(n_neurons, np.nan, dtype=float)

rng = np.random.default_rng(1764)

for i, w in enumerate(tqdm(incoming_weights)):
    w = np.asarray(w, dtype=float)
    n_inputs = w.size

    if n_inputs == 0:
        continue  # leave NaNs for loss/pred_loss

    # Monte Carlo loss (fast, blocked)
    l_hat = average_error_fast(
        w,
        eta=eta,
        eps=eps,
        n_draws=n_draws,
        n_perturb=n_perturb,
        block_perturb=128,
        rng=rng,
    )

    loss[i] = l_hat

# Build pandas DataFrame
sim_df = pd.DataFrame({
    "root_id": idx_to_id,
    "sim_loss": loss,
})

# Save dataset as parquet
sim_df.to_parquet(data_dir+'loss_data.parquet')


