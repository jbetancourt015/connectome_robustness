"""
    This script simulates loss for parametric distributions.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Mon 15 Dec 2025
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
import matplotlib.pyplot as plt

sim_dir = '../simulation_results/'

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
# POISSON SIMULATION
#------------------------------------------------------------------------------
eta = 1.0
eps = 1.0

n_inputs = 100

n_neurons = int(1e3)
n_w = 10
w0_vals = np.logspace(0.,3.,n_w)

rng = np.random.default_rng(1764)

poisson_loss = []

for w0 in tqdm(w0_vals):
    loss = np.full(n_neurons, np.nan, dtype=float)
    for i in range(n_neurons):
        # Draw neurons
        w = np.random.poisson(w0, n_inputs)
    
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
    
    poisson_loss.append(np.mean(loss))

# Build pandas DataFrame
poisson_df = pd.DataFrame({
    "w0": w0_vals,
    "sim_loss": np.array(poisson_loss),
})

# Save dataset as parquet
poisson_df.to_parquet(sim_dir+f"poisson_sim_{n_inputs}.parquet")

#------------------------------------------------------------------------------
# LOGNORMAL SIMULATION
#------------------------------------------------------------------------------
eta = 1.0
eps = 1.0

n_inputs = 100

n_neurons = int(1e3)
n_sigma = 10
mu_vals = np.arange(5)
sigma_vals = np.linspace(1.,5.,n_sigma)

lognormal_loss = []
mu_ar = []
sigma_ar = []

for mu in tqdm(mu_vals):
    for sigma in sigma_vals:
        loss = np.full(n_neurons, np.nan, dtype=float)
        for i in range(n_neurons):
            # Draw neurons
            w = np.random.lognormal(mu,sigma, n_inputs)
        
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
        
        lognormal_loss.append(np.mean(loss))
        mu_ar.append(mu)
        sigma_ar.append(sigma)

# Build pandas DataFrame
lognormal_df = pd.DataFrame({
    "mu": np.array(mu_ar),
    "sigma": np.array(sigma_ar),
    "sim_loss": np.array(lognormal_loss),
})

# Save dataset as parquet
lognormal_df.to_parquet(sim_dir+f"lognormal_sim_{n_inputs}.parquet")
