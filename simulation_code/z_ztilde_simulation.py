"""
    This script computes z and ztilde values for a single weight vector
    across all input and perturbation instances.
-------------------------------------------------------------------------------
created on:
    Mon 12 Jan 2026
-------------------------------------------------------------------------------
last change:
    Mon 12 Jan 2026
-------------------------------------------------------------------------------
notes:
    For a single lognormal weight vector w, saves raw z (baseline local field)
    and ztilde (perturbed local field) values to parquet format.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

sim_dir = '../simulation_results/'


def compute_z_ztilde(
    mean,
    var,
    n_inputs,
    eta,
    eps,
    n_draws,
    n_perturb,
    rng=None,
):
    """
    Compute z and ztilde values for a single lognormal weight vector.
    
    Parameters
    ----------
    mean : float
        Mean of the lognormal distribution for weights.
    var : float
        Variance of the lognormal distribution for weights.
    n_inputs : int
        Number of input dimensions (size of weight vector).
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    n_draws : int
        Number of input draws (x instances).
    n_perturb : int
        Number of perturbation draws (w_hat instances).
    rng : numpy.random.Generator, optional
        Random number generator. If None, creates a new one.
    
    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        - draw_idx: Index of the input x instance (0 to n_draws-1)
        - perturb_idx: Index of the perturbation instance (0 to n_perturb-1)
        - z: Baseline output value
        - ztilde: Perturbed output value
    w : np.ndarray
        The generated weight vector (for reference).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate lognormal weights
    mu = np.log(mean**2 / np.sqrt(mean**2 + var))
    sigma = np.sqrt(np.log(1. + var / mean**2))
    w = rng.lognormal(mu, sigma, n_inputs)
    
    # Generate gamma weights
    # theta = var/mean
    # alpha = mean/theta
    # w = rng.gamma(alpha, theta, n_inputs)

    # Draw inputs: n_inputs × n_draws
    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))

    # Draw base Gaussian noise: n_inputs × n_perturb
    base_noise = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb))
    # Scale by w**(eta/2)
    w_hat = base_noise * (w**(eta / 2.0))[:, None]

    # Baseline output: shape (n_draws,)
    z = w @ x

    # Perturbed output: shape (n_perturb, n_draws)
    # ztilde[j, i] = z[i] + eps * (w_hat[:, j].T @ x[:, i])
    delta = eps * (w_hat.T @ x)  # (n_perturb, n_draws)
    ztilde = z[None, :] + delta  # (n_perturb, n_draws)

    # Build long-form DataFrame
    # Create index arrays
    draw_idx = np.tile(np.arange(n_draws), n_perturb)
    perturb_idx = np.repeat(np.arange(n_perturb), n_draws)
    z_flat = np.tile(z, n_perturb)
    ztilde_flat = ztilde.ravel()

    df = pd.DataFrame({
        'draw_idx': draw_idx,
        'perturb_idx': perturb_idx,
        'z': z_flat,
        'ztilde': ztilde_flat,
    })

    return df, w


def save_z_ztilde(
    mean,
    var,
    n_inputs,
    eta,
    eps,
    n_draws,
    n_perturb,
    fname=None,
    rng=None,
):
    """
    Compute and save z and ztilde values to parquet.
    
    Parameters
    ----------
    mean : float
        Mean of the lognormal distribution for weights.
    var : float
        Variance of the lognormal distribution for weights.
    n_inputs : int
        Number of input dimensions (size of weight vector).
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    n_draws : int
        Number of input draws (x instances).
    n_perturb : int
        Number of perturbation draws (w_hat instances).
    output_path : str, optional
        Path to save the parquet file. If None, generates default path.
    rng : numpy.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    pd.DataFrame
        The computed DataFrame.
    w : np.ndarray
        The generated weight vector.
    """
    df, w = compute_z_ztilde(
        mean=mean,
        var=var,
        n_inputs=n_inputs,
        eta=eta,
        eps=eps,
        n_draws=n_draws,
        n_perturb=n_perturb,
        rng=rng,
    )

    output_path = sim_dir + f"z_ztilde_{fname}.parquet"

    df.to_parquet(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

    return df, w


#------------------------------------------------------------------------------
# RUN SIMULATION
#------------------------------------------------------------------------------
rng = np.random.default_rng(1764)

# Parameters
means = [1., 9.5, 1.]
second_moments = [100., 100., 400.]
n_inputs = int(1e4)
eta = 1.0
eps = 10.0
n_draws = int(1e3)
n_perturb = int(1e3)
fnames = ['baseline', 'high_w', 'high_w2']

for i in range(3):
    fname = fnames[i]
    mean = means[i]
    var = second_moments[i] - mean**2
    print(f"Simulating round {fname}")
    # Compute and save
    df, w = save_z_ztilde(
        mean=mean,
        var=var,
        n_inputs=n_inputs,
        eta=eta,
        eps=eps,
        n_draws=n_draws,
        n_perturb=n_perturb,
        fname=fname,
        rng=rng,
    )

