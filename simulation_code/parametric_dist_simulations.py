"""
    This script simulates loss for parametric distributions.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Tue 20 Jan 2026
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

sim_dir = '../simulation_results/'

#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
n_draws = int(1e3)          # number of input draws
n_perturb = int(1e3)        # number of weight perturbation draws

rng = np.random.default_rng(1764)

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

    # Draw inputs: n_inputs × n_draws (faster than rng.choice)
    x = 2.0 * rng.integers(0, 2, size=(n_inputs, n_draws)) - 1.0

    # Draw base Gaussian noise: n_inputs × n_perturb
    # Pre-scale by eps and w**(eta/2) to avoid repeated multiplication
    w_scale = eps * (w ** (eta / 2.0))
    w_hat = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb)) * w_scale[:, None]

    # Baseline output: shape (n_draws,)
    z = w @ x

    total_pairs = n_draws * n_perturb
    error_count = 0

    # Process perturbations in blocks to keep memory small
    for start in range(0, n_perturb, block_perturb):
        end = min(start + block_perturb, n_perturb)

        # (block_size × n_draws)
        delta = w_hat[:, start:end].T @ x

        # z * (z + delta) < 0 means sign flip (faster than sign arithmetic)
        zztilde = z * (z + delta)
        error_count += (zztilde < 0).sum()

    return error_count / total_pairs


def get_dist_params(distribution, mean, var):
    """
    Convert mean/variance to native distribution parameters.
    
    Parameters
    ----------
    distribution : str
        One of 'lognormal', 'lomax', or 'gamma'.
    mean : float
        Desired mean of the distribution.
    var : float
        Desired variance of the distribution.
    
    Returns
    -------
    params : tuple
        Native parameters for the distribution.
    """
    if distribution == 'lognormal':
        sigma = np.sqrt(np.log(1. + var / mean**2))
        mu = np.log(mean) - sigma**2 / 2.
        return (mu, sigma)
    
    elif distribution == 'lomax':
        if var <= mean**2:
            raise ValueError("Lomax requires var > mean^2 for finite variance (alpha > 2)")
        a = 2. * var / (var - mean**2)
        w0 = (a - 1.) * mean
        return (a, w0)
    
    elif distribution == 'gamma':
        shape = mean**2 / var
        scale = var / mean
        return (shape, scale)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}. "
                         f"Choose from 'lognormal', 'lomax', 'gamma'.")


def sample_weights(distribution, params, n_inputs, rng):
    """
    Sample weights using pre-computed distribution parameters.
    
    Parameters
    ----------
    distribution : str
        One of 'lognormal', 'lomax', or 'gamma'.
    params : tuple
        Native parameters from get_dist_params().
    n_inputs : int
        Number of samples to draw.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    w : ndarray
        Array of sampled weights.
    """
    if distribution == 'lognormal':
        mu, sigma = params
        return rng.lognormal(mu, sigma, n_inputs)
    
    elif distribution == 'lomax':
        a, w0 = params
        r = rng.random(n_inputs)
        return ((1-r)**(-1/a) - 1.) * w0
    
    elif distribution == 'gamma':
        shape, scale = params
        return rng.gamma(shape, scale, n_inputs)


def run_simulation(
    distribution,
    mean_vals,
    var_vals,
    n_inputs=100,
    n_neurons=1000,
    n_draws=1000,
    n_perturb=1000,
    eta=1.0,
    eps=1.0,
    rng=None,
    sim_dir='../simulation_results/',
):
    """
    Run Monte Carlo loss simulation for a parametric distribution.
    
    Parameters
    ----------
    distribution : str
        One of 'lognormal', 'lomax', or 'gamma'.
    mean_vals : array-like
        Array of mean values to simulate.
    var_vals : array-like
        Array of variance values to simulate.
    n_inputs : int
        Number of input weights per neuron.
    n_neurons : int
        Number of neurons to simulate per (mean, var) pair.
    n_draws : int
        Number of input draws for Monte Carlo.
    n_perturb : int
        Number of weight perturbation draws.
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    rng : np.random.Generator, optional
        Random number generator.
    sim_dir : str
        Directory to save output parquet file.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 'mean', 'var', 'sim_loss'.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    loss_list = []
    mean_list = []
    var_list = []
    
    for mean in tqdm(mean_vals, desc=distribution):
        for var in var_vals:
            # Compute distribution parameters once per (mean, var) pair
            params = get_dist_params(distribution, mean, var)
            
            loss = np.full(n_neurons, np.nan, dtype=float)
            for i in range(n_neurons):
                # Sample weights using pre-computed parameters
                w = sample_weights(distribution, params, n_inputs, rng)
                
                # Monte Carlo loss estimate
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
            
            loss_list.append(np.mean(loss))
            mean_list.append(mean)
            var_list.append(var)
    
    # Build DataFrame
    df = pd.DataFrame({
        "mean": np.array(mean_list),
        "var": np.array(var_list),
        "sim_loss": np.array(loss_list),
    })
    
    # Save to parquet with distribution name
    output_path = f"{sim_dir}{distribution}_sim_{n_inputs}.parquet"
    df.to_parquet(output_path)
    print(f"Saved: {output_path}")
    
    return df


#------------------------------------------------------------------------------
# MAIN EXECUTION
#------------------------------------------------------------------------------
# Simulation parameters
n_inputs = int(1e2)
mean_vals = np.linspace(10., 50, 5)
var_vals = 10**np.arange(5)

# For lomax, variance must be > mean^2, so use larger variances
lomax_var_vals = 10**np.arange(1, 5)

# Run simulations for each distribution
# run_simulation('lognormal', mean_vals, var_vals, n_inputs=n_inputs,
#                 n_draws=n_draws, n_perturb=n_perturb, rng=rng, sim_dir=sim_dir)

# run_simulation('lomax', mean_vals, lomax_var_vals, n_inputs=n_inputs,
#                 n_draws=n_draws, n_perturb=n_perturb, rng=rng, sim_dir=sim_dir)

run_simulation('gamma', mean_vals, var_vals, n_inputs=n_inputs,
                n_draws=n_draws, n_perturb=n_perturb, rng=rng, sim_dir=sim_dir)