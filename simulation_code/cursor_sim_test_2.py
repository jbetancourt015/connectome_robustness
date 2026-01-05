# %%
"""
    Variance-Fixed Parametric Distribution Simulations
    
    This script simulates loss for lognormal and lomax distributions where
    variance is fixed and distribution shape parameters are varied directly.
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import brentq

sim_dir = '../simulation_results/'

# %%
#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
n_draws = int(1e3)          # number of input draws
n_perturb = int(1e3)        # number of weight perturbation draws
n_neurons = int(1e3)        # neurons per parameter combination
n_inputs = int(1e3)         # inputs per neuron

eta = 1.0
eps = 1.0

rng = np.random.default_rng(1764)

# %%
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

# %%
#------------------------------------------------------------------------------
# LOGNORMAL: PARAMETER SOLVING
#------------------------------------------------------------------------------
def lognormal_variance(mu, sigma):
    """
    Compute variance of lognormal distribution given mu and sigma.
    Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    """
    s2 = sigma**2
    return (np.exp(s2) - 1) * np.exp(2*mu + s2)

def solve_sigma_for_variance(mu, target_var, sigma_bounds=(1e-6, 10.0)):
    """
    Given mu and target variance, solve for sigma using root finding.
    """
    def objective(sigma):
        return lognormal_variance(mu, sigma) - target_var
    
    # Check if solution exists in bounds
    f_low = objective(sigma_bounds[0])
    f_high = objective(sigma_bounds[1])
    
    if f_low * f_high > 0:
        # No root in interval - return NaN
        return np.nan
    
    return brentq(objective, sigma_bounds[0], sigma_bounds[1])

# %%
#------------------------------------------------------------------------------
# LOGNORMAL SIMULATION (Fixed Variance, Varying mu)
#------------------------------------------------------------------------------
print("Running Lognormal Simulation...")

# Parameter ranges
n_mu = 10
mu_vals = np.linspace(0., 3., n_mu)
var_vals = 10**np.arange(5)  # [1, 10, 100, 1000, 10000]

lognormal_loss = []
mu_ar = []
var_ar = []
sigma_ar = []

for var in tqdm(var_vals, desc="Variance"):
    for mu in mu_vals:
        # Solve for sigma given mu and variance
        sigma = solve_sigma_for_variance(mu, var)
        
        if np.isnan(sigma):
            # Skip invalid parameter combinations
            continue
        
        loss = np.full(n_neurons, np.nan, dtype=float)
        for i in range(n_neurons):
            # Draw weights from lognormal
            w = np.random.lognormal(mu, sigma, n_inputs)
        
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
        var_ar.append(var)
        sigma_ar.append(sigma)

# Build pandas DataFrame
lognormal_df = pd.DataFrame({
    "mu": np.array(mu_ar),
    "var": np.array(var_ar),
    "sigma": np.array(sigma_ar),
    "sim_loss": np.array(lognormal_loss),
})

# Save dataset as parquet
lognormal_df.to_parquet(sim_dir + f"lognormal_varfixed_sim_{n_inputs}.parquet")
print(f"Lognormal simulation saved to {sim_dir}lognormal_varfixed_sim_{n_inputs}.parquet")

# %%
#------------------------------------------------------------------------------
# LOMAX: PARAMETER SOLVING
#------------------------------------------------------------------------------
def lomax_variance(w0, a):
    """
    Compute variance of Lomax distribution given w0 and a.
    Var[X] = a * w0^2 / ((a-1)^2 * (a-2)) for a > 2
    """
    if a <= 2:
        return np.inf
    return a * w0**2 / ((a - 1)**2 * (a - 2))

def solve_a_for_variance(w0, target_var, a_bounds=(2.001, 1000.0)):
    """
    Given w0 and target variance, solve for shape parameter a using root finding.
    Note: a must be > 2 for finite variance.
    """
    def objective(a):
        return lomax_variance(w0, a) - target_var
    
    # Check if solution exists in bounds
    f_low = objective(a_bounds[0])
    f_high = objective(a_bounds[1])
    
    if f_low * f_high > 0:
        # No root in interval - return NaN
        return np.nan
    
    return brentq(objective, a_bounds[0], a_bounds[1])

def invert_lomax(r, w0, a):
    """Inverse CDF sampling for Lomax distribution."""
    return ((1 - r)**(-1/a) - 1.) * w0

# %%
#------------------------------------------------------------------------------
# LOMAX SIMULATION (Fixed Variance, Varying w0)
#------------------------------------------------------------------------------
print("Running Lomax Simulation...")

# Parameter ranges
n_w0 = 10
w0_vals = np.logspace(0., 3., n_w0)  # 1 to 1000, log scale
var_vals_lomax = 10**np.arange(1, 5)  # [10, 100, 1000, 10000]

lomax_loss = []
w0_ar = []
var_ar_lomax = []
a_ar = []

for var in tqdm(var_vals_lomax, desc="Variance"):
    for w0 in w0_vals:
        # Solve for a given w0 and variance
        a = solve_a_for_variance(w0, var)
        
        if np.isnan(a):
            # Skip invalid parameter combinations
            continue
        
        loss = np.full(n_neurons, np.nan, dtype=float)
        for i in range(n_neurons):
            # Draw weights from Lomax using inverse CDF
            r = np.random.rand(n_inputs)
            w = invert_lomax(r, w0, a)
        
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
        
        lomax_loss.append(np.mean(loss))
        w0_ar.append(w0)
        var_ar_lomax.append(var)
        a_ar.append(a)

# Build pandas DataFrame
lomax_df = pd.DataFrame({
    "w0": np.array(w0_ar),
    "var": np.array(var_ar_lomax),
    "a": np.array(a_ar),
    "sim_loss": np.array(lomax_loss),
})

# Save dataset as parquet
lomax_df.to_parquet(sim_dir + f"lomax_varfixed_sim_{n_inputs}.parquet")
print(f"Lomax simulation saved to {sim_dir}lomax_varfixed_sim_{n_inputs}.parquet")

# %%
print("All simulations complete!")
print(f"\nLognormal results shape: {lognormal_df.shape}")
print(lognormal_df.head(10))
print(f"\nLomax results shape: {lomax_df.shape}")
print(lomax_df.head(10))

