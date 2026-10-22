"""
    Master script for running all simulations for the robust communication paper.

    Can be run directly to execute all simulations, or imported by figure scripts
    to access individual simulation functions without triggering execution.
-------------------------------------------------------------------------------
created on:
    Tue 4 Feb 2026
-------------------------------------------------------------------------------
last change:
    Thu 22 Oct 2026
-------------------------------------------------------------------------------
notes:
    Run this script once to generate all simulation data:

    1. FAFB error simulation -> simulation_results/fafb_error_data.parquet
    2. Periphery scoring -> simulation_results/fafb_periphery_data.parquet
    3. z/zhat simulation -> simulation_results/z_zhat_simulations.parquet
    4. Parametric distributions -> simulation_results/{dist}_sim.parquet
    4B. Parametric distributions (p_fire sweep) -> simulation_results/gamma_sim_sparse.parquet
    5. Network shuffling -> simulation_results/{name}_shuffled.parquet

    Estimated total runtime: 4-8 hours (dominated by FAFB simulations)

    Set SKIP_EXISTING_SIMULATIONS = True to skip simulations if output files exist.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, load_npz
from scipy.special import erfinv
from tqdm import tqdm
from params import (
    rng_seed,
    block_perturb,
    error_sigma,
    error_n_draws,
    error_n_perturb,
    error_p_fire_vals,
    error_sigma_vals,
    error_sweep_n_draws,
    error_sweep_n_perturb,
    periphery_n_sim,
    periphery_threshold,
    periphery_repeats,
    periphery_max_steps,
    zztilde_param_sets,
    zztilde_n_inputs,
    zztilde_sigma,
    zztilde_n_draws,
    zztilde_n_perturb,
    parametric_n_neurons,
    parametric_n_inputs,
    parametric_n_draws,
    parametric_n_perturb,
    parametric_mean_vals,
    parametric_n_var,
    sparse_p_fire_vals,
    parametric_sparse_n_draws,
    parametric_sparse_n_perturb,
    shuffle_k_min,
    shuffle_n_threshold_default,
    shuffle_n_threshold_banc,
    connectomes,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to True to skip simulations if output files already exist
SKIP_EXISTING_SIMULATIONS = True

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------
processed_dir = "../processed_data/"
sim_dir = "../simulation_results/"

# Create output directories if they don't exist
os.makedirs(sim_dir, exist_ok=True)

# ==============================================================================
# CONNECTOME DATA
# ==============================================================================


def load_connectome(name):
    return load_npz(f"{processed_dir}{name}.npz")


def compute_robustness(A, k_min, normalized):
    k = A.getnnz(axis=0)
    mask = k >= k_min

    B = A.copy()
    B.data = B.data**2

    mu_1 = np.asarray(A.sum(axis=0)).ravel()[mask] / k[mask]  # E[w]
    mu_2 = np.asarray(B.sum(axis=0)).ravel()[mask] / k[mask]  # E[w²]

    r = np.sqrt(mu_2 / mu_1)
    if normalized:
        robustness = (r - np.sqrt(mu_1)) / (np.sqrt(1.0 + mu_1) - np.sqrt(mu_1))
    else:
        robustness = r
    return robustness


# ==============================================================================
# SIMULATION FUNCTIONS
# ==============================================================================
def average_error_fast(w, sigma, n_draws, n_perturb, block_perturb, rng):
    """
    Monte Carlo estimate of the error, vectorized and blocked to reduce memory.

    Parameters
    ----------
    w : array-like
        Weight vector.
    sigma : float
        Perturbation scale (noise magnitude).
    n_draws : int
        Number of input draws.
    n_perturb : int
        Number of weight perturbation draws.
    block_perturb : int
        Block size for memory-efficient computation.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        Estimated error (fraction of sign flips).
    """
    w = np.asarray(w, dtype=float)
    n_inputs = w.size
    if n_inputs == 0:
        return np.nan

    # Draw inputs: n_inputs × n_draws
    x = 2.0 * rng.integers(0, 2, size=(n_inputs, n_draws)) - 1.0

    # Draw base Gaussian noise and scale
    w_scale = sigma * np.sqrt(w)
    w_hat = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb)) * w_scale[:, None]

    # Baseline output: shape (n_draws,)
    z = w @ x

    total_pairs = n_draws * n_perturb
    error_count = 0

    # Process perturbations in blocks
    for start in range(0, n_perturb, block_perturb):
        end = min(start + block_perturb, n_perturb)
        delta = w_hat[:, start:end].T @ x
        zztilde = z * (z + delta)
        error_count += (zztilde < 0).sum()

    return error_count / total_pairs


def _phi_inv(p):
    """Standard normal quantile (probit) corresponding to firing probability p."""
    return -np.sqrt(2) * erfinv(1.0 - 2 * p)


def average_error_fast_biased(
    w, sigma, p_fire, n_draws, n_perturb, block_perturb, rng
):
    """
    Monte Carlo estimate of the error for a neuron biased to a target firing
    probability p_fire, vectorized and blocked to reduce memory.

    The local field is z = w @ x + b, where the bias b = sqrt(sum(w**2)) *
    Phi^-1(p_fire) is chosen analytically (assuming z is asymptotically Gaussian,
    since inputs are i.i.d. symmetric) so that P(z > 0) ~= p_fire. At p_fire=0.5
    this reduces to average_error_fast (b=0).

    Parameters
    ----------
    w : array-like
        Weight vector.
    sigma : float
        Perturbation scale (noise magnitude).
    p_fire : float
        Target firing probability in (0, 1).
    n_draws : int
        Number of input draws.
    n_perturb : int
        Number of weight perturbation draws.
    block_perturb : int
        Block size for memory-efficient computation.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        Estimated error (fraction of sign flips).
    """
    w = np.asarray(w, dtype=float)
    n_inputs = w.size
    if n_inputs == 0:
        return np.nan

    # Draw inputs: n_inputs × n_draws
    x = 2.0 * rng.integers(0, 2, size=(n_inputs, n_draws)) - 1.0

    # Draw base Gaussian noise and scale
    w_scale = sigma * np.sqrt(w)
    w_hat = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb)) * w_scale[:, None]

    # Analytical bias to hit the target firing probability, then baseline output
    bias = np.sqrt(np.sum(w**2)) * _phi_inv(p_fire)
    z = w @ x + bias

    total_pairs = n_draws * n_perturb
    error_count = 0

    # Process perturbations in blocks
    for start in range(0, n_perturb, block_perturb):
        end = min(start + block_perturb, n_perturb)
        delta = w_hat[:, start:end].T @ x
        zztilde = z * (z + delta)
        error_count += (zztilde < 0).sum()

    return error_count / total_pairs


def simulate_propagation(A, pool, threshold, repeats, seed, max_steps):
    """
    Simulate stochastic propagation on a directed weighted sparse graph.

    Parameters
    ----------
    A : sparse matrix (CSR preferred)
        Adjacency matrix where A[i,j] = weight of edge i -> j.
    pool : array-like
        Initial set of node indices in the pool at time 0.
    threshold : float
        Probability scaling factor.
    repeats : int
        Number of consecutive no-growth steps before stopping.
    seed : int, optional
        Random seed.
    max_steps : int, optional
        Maximum number of propagation steps.

    Returns
    -------
    time_added : ndarray
        Step when each node was first reached (-1 if never reached).
    """
    A = A.tocsr()
    n = A.shape[0]
    pool = np.asarray(pool, dtype=int)

    col_sums = np.asarray(A.sum(axis=0)).ravel()
    denom = np.where(col_sums > 0, threshold * col_sums, np.inf)

    time_added = -np.zeros(n, dtype=int) - 1
    in_pool = np.zeros(n, dtype=bool)
    time_added[pool] = 0
    in_pool[pool] = True

    rng = np.random.default_rng(seed)
    step = 1
    no_growth_streak = 0

    while True:
        sources = np.nonzero(in_pool)[0]
        if sources.size == 0:
            break

        sub = A[sources, :].tocoo()
        if sub.nnz == 0:
            no_growth_streak += 1
            if no_growth_streak >= repeats:
                break
            step += 1
            if max_steps is not None and step > max_steps:
                break
            continue

        cols = sub.col
        w = sub.data

        p = np.divide(
            w,
            denom[cols],
            out=np.zeros_like(w, dtype=float),
            where=np.isfinite(denom[cols]),
        )
        p = np.clip(p, 0.0, 1.0)

        successes = rng.random(p.size) < p
        hit_cols = np.unique(cols[successes])
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

        if in_pool.all():
            break

    return time_added


def average_propagation(
    A, pool, n_sim, threshold, repeats, seed, max_steps
):
    """
    Run simulate_propagation n_sim times and return the average time_added array.

    Parameters
    ----------
    A : sparse matrix
        Adjacency matrix.
    pool : array-like
        Initial seed set.
    n_sim : int
        Number of Monte Carlo runs.
    threshold, repeats, seed, max_steps :
        Passed to simulate_propagation.

    Returns
    -------
    avg_time : ndarray
        Average time to reach each node (NaN if never reached).
    frac_reached : ndarray
        Fraction of runs where each node was reached.
    """
    n = A.shape[0]
    avg_time = np.zeros(n, dtype=float)
    count_reached = np.zeros(n, dtype=int)

    base_rng = np.random.default_rng(seed)

    for s in tqdm(range(n_sim), desc="Propagation simulations"):
        run_seed = base_rng.integers(1e9)
        t = simulate_propagation(
            A, pool, threshold, repeats, seed=run_seed, max_steps=max_steps
        )
        reached = t >= 0
        count_reached += reached
        avg_time[reached] += (t[reached] - avg_time[reached]) / count_reached[reached]

    avg_time[count_reached == 0] = np.nan
    frac_reached = count_reached / n_sim
    return avg_time, frac_reached


# ==============================================================================
# SIMULATION 1: FAFB ERROR
# ==============================================================================
def _error_target_precision(p_fire, sigma):
    """n_draws/n_perturb that a given (p_fire, sigma) pair should currently use:
    the full-precision baseline settings at (0.5, error_sigma), the reduced
    sweep settings otherwise."""
    if (p_fire, sigma) == (0.5, error_sigma):
        return error_n_draws, error_n_perturb
    return error_sweep_n_draws, error_sweep_n_perturb


def run_fafb_error_simulation():
    """
    Run Monte Carlo error-rate simulation for all FAFB neurons, swept over
    two dimensions from a shared baseline point (p_fire=0.5, sigma=1.0), one
    at a time: target firing probability (error_p_fire_vals, at sigma=1.0)
    and perturbation scale (error_sigma_vals, at p_fire=0.5). Each neuron's
    local field is analytically biased to hit each target p_fire (see
    average_error_fast_biased). (p_fire, sigma) pairs already present in the
    output file are reused rather than recomputed when
    SKIP_EXISTING_SIMULATIONS is True.
    Cached rows whose n_draws/n_perturb don't match the current target
    precision (error_n_draws/error_n_perturb at baseline, error_sweep_n_draws/
    error_sweep_n_perturb otherwise) are wiped and re-simulated.

    Input: processed_data/fafb_connections.parquet
    Output: simulation_results/fafb_error_data.parquet
    """
    print("=" * 60)
    print("SIMULATION 1: FAFB Error")
    print("=" * 60)

    output_file = sim_dir + "fafb_error_data.parquet"

    # Reuse cached results for (p_fire, sigma) pairs already simulated, so
    # resuming doesn't repeat the expensive Monte Carlo loop over all
    # FAFB neurons.
    cached_df = None
    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        cached_df = pd.read_parquet(output_file)

    # Wipe any cached rows whose n_draws/n_perturb no longer match the current
    # target precision for their (p_fire, sigma) pair, so a precision change
    # in params.py triggers a re-run instead of silently mixing precisions.
    if cached_df is not None and len(cached_df) > 0:
        expected_draws, expected_perturb = zip(
            *(
                _error_target_precision(pf, s)
                for pf, s in zip(cached_df["p_fire"], cached_df["sigma"])
            )
        )
        stale = (cached_df["n_draws"].to_numpy() != np.array(expected_draws)) | (
            cached_df["n_perturb"].to_numpy() != np.array(expected_perturb)
        )
        if stale.any():
            stale_pairs = sorted(
                set(zip(cached_df.loc[stale, "p_fire"], cached_df.loc[stale, "sigma"]))
            )
            print(
                f"Outdated n_draws/n_perturb found for (p_fire, sigma) = {stale_pairs}; "
                "wiping and re-running."
            )
            cached_df = cached_df.loc[~stale].reset_index(drop=True)

    targets = {(pf, error_sigma) for pf in error_p_fire_vals} | {
        (0.5, s) for s in error_sigma_vals
    }
    cached_pairs = (
        set(zip(cached_df["p_fire"], cached_df["sigma"]))
        if cached_df is not None
        else set()
    )
    pending = sorted(targets - cached_pairs)

    if not pending:
        print(f"Skipping: all (p_fire, sigma) pairs already present in {output_file}")
        return

    # Load data
    print("Loading connections data...")
    conn_df = pd.read_parquet(processed_dir + "fafb_connections.parquet")

    # Build sparse matrix
    print("Building sparse matrix...")
    nodes_index = pd.Index(
        pd.unique(
            pd.concat(
                [conn_df["pre_root_id"], conn_df["post_root_id"]], ignore_index=True
            )
        ),
        name="root_id",
    )

    id_to_idx = pd.Series(np.arange(nodes_index.size), index=nodes_index)
    idx_to_id = nodes_index.to_numpy()

    rows = id_to_idx.reindex(conn_df["pre_root_id"]).to_numpy()
    cols = id_to_idx.reindex(conn_df["post_root_id"]).to_numpy()
    data = conn_df["syn_count"].to_numpy()

    A = coo_matrix(
        (data, (rows, cols)), shape=(nodes_index.size, nodes_index.size)
    ).tocsc()
    N = A.shape[0]

    # Get incoming weights for each neuron
    print("Extracting incoming weights...")
    incoming_weights = [A.data[A.indptr[j] : A.indptr[j + 1]] for j in range(N)]

    rng = np.random.default_rng(rng_seed)
    new_dfs = []

    for p_fire, sigma in pending:
        n_draws, n_perturb = _error_target_precision(p_fire, sigma)

        error = np.full(N, np.nan, dtype=float)
        print(
            f"Simulating error for {N} neurons at p_fire={p_fire}, sigma={sigma} "
            f"(n_draws={n_draws}, n_perturb={n_perturb})..."
        )
        for i, w in enumerate(
            tqdm(incoming_weights, desc=f"FAFB error (p_fire={p_fire}, sigma={sigma})")
        ):
            w = np.asarray(w, dtype=float)
            if w.size == 0:
                continue

            e_hat = average_error_fast_biased(
                w,
                sigma=sigma,
                p_fire=p_fire,
                n_draws=n_draws,
                n_perturb=n_perturb,
                block_perturb=block_perturb,
                rng=rng,
            )
            error[i] = e_hat

        new_dfs.append(
            pd.DataFrame(
                {
                    "root_id": idx_to_id,
                    "p_fire": p_fire,
                    "sim_error": error,
                    "sigma": sigma,
                    "n_draws": n_draws,
                    "n_perturb": n_perturb,
                }
            )
        )

    # Save results
    result_df = pd.concat(
        ([cached_df] if cached_df is not None else []) + new_dfs, ignore_index=True
    )
    result_df.to_parquet(output_file)
    print(f"Saved: {output_file}")


# ==============================================================================
# SIMULATION 2: FAFB PERIPHERY SCORING
# ==============================================================================
def run_fafb_periphery_scoring():
    """
    Run stochastic propagation to compute peripherality scores.

    Input: processed_data/fafb_connections.parquet, processed_data/fafb_neurons.parquet
    Output: simulation_results/fafb_periphery_data.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 2: FAFB Periphery Scoring")
    print("=" * 60)

    output_file = sim_dir + "fafb_periphery_data.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"Skipping: {output_file} already exists")
        return

    # Load data
    print("Loading data...")
    conn_df = pd.read_parquet(processed_dir + "fafb_connections.parquet")
    neuron_df = pd.read_parquet(processed_dir + "fafb_neurons.parquet")

    # Build sparse matrix
    print("Building sparse matrix...")
    nodes_index = pd.Index(
        pd.unique(
            pd.concat(
                [conn_df["pre_root_id"], conn_df["post_root_id"]], ignore_index=True
            )
        ),
        name="root_id",
    )

    id_to_idx = pd.Series(np.arange(nodes_index.size), index=nodes_index)
    idx_to_id = nodes_index.to_numpy()

    rows = id_to_idx.reindex(conn_df["pre_root_id"]).to_numpy()
    cols = id_to_idx.reindex(conn_df["post_root_id"]).to_numpy()
    data = conn_df["syn_count"].to_numpy()

    A = coo_matrix(
        (data, (rows, cols)), shape=(nodes_index.size, nodes_index.size)
    ).tocsr()

    # Define seed sets
    visual_list = ["R1-6", "R7", "R8"]
    olfactory_list = ["ORN"]

    visual_re = "|".join(map(re.escape, visual_list))
    olfactory_re = "|".join(map(re.escape, olfactory_list))

    visual_mask = neuron_df["primary_type"].str.contains(visual_re)
    olfactory_mask = neuron_df["primary_type"].str.contains(olfactory_re)

    masks = [visual_mask, olfactory_mask]
    labels = ["visual", "olfactory"]

    sim_df = pd.DataFrame({"root_id": idx_to_id})

    for i, mask in enumerate(masks):
        print(f"\nProcessing {labels[i]} seed set...")
        selected_ids = neuron_df.loc[mask, "root_id"].values
        seed_set = np.array(
            [id_to_idx[rid] for rid in selected_ids if rid in id_to_idx]
        )

        avg_dist, frac_reached = average_propagation(
            A,
            seed_set,
            periphery_n_sim,
            threshold=periphery_threshold,
            repeats=periphery_repeats,
            seed=rng_seed,
            max_steps=periphery_max_steps,
        )
        print(f"Fraction reached from {labels[i]} seed: {np.mean(frac_reached):.3f}")

        sim_df[f"distance_{labels[i]}"] = avg_dist

    # Save results
    sim_df.to_parquet(output_file)
    print(f"Saved: {output_file}")


# ==============================================================================
# SIMULATION 3: Z/ZTILDE
# ==============================================================================
def generate_weights_zztilde(distribution, mean, n_inputs, rng, var):
    """
    Generate weights from the specified distribution for z/zhat simulation.

    Parameters
    ----------
    distribution : str
        One of 'dirac' or 'gamma'.
    mean : float
        Mean of the distribution.
    n_inputs : int
        Number of weights to generate.
    rng : numpy.random.Generator
        Random number generator.
    var : float, optional
        Variance (required only for 'gamma').

    Returns
    -------
    w : ndarray
        Generated weight vector.
    computed_var : float
        The variance of the distribution.
    """
    if distribution == "dirac":
        w = np.full(n_inputs, mean)
        computed_var = 0.0
    elif distribution == "gamma":
        if var is None:
            raise ValueError("Gamma distribution requires variance parameter")
        theta = var / mean
        alpha = mean / theta
        w = rng.gamma(alpha, theta, n_inputs)
        computed_var = var
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return w, computed_var


def compute_z_zhat(
    distribution, mean, n_inputs, sigma, n_draws, n_perturb, rng, var
):
    """
    Compute z and zhat (= ztilde - z) values for a weight vector drawn from the specified distribution.

    Parameters
    ----------
    distribution : str
        Distribution type.
    mean : float
        Mean parameter.
    n_inputs : int
        Number of input dimensions.
    sigma : float
        Perturbation scale (noise magnitude).
    n_draws : int
        Number of input draws.
    n_perturb : int
        Number of perturbation draws.
    rng : numpy.random.Generator
        Random number generator.
    var : float, optional
        Variance (required for 'gamma').

    Returns
    -------
    df : pd.DataFrame
        Long-form DataFrame with z and zhat values.
    w : ndarray
        The generated weight vector.
    """
    w, computed_var = generate_weights_zztilde(distribution, mean, n_inputs, rng, var)

    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))
    base_noise = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb))
    w_hat = base_noise * np.sqrt(w)[:, None]

    z = w @ x
    zhat = sigma * (w_hat.T @ x)

    draw_idx = np.tile(np.arange(n_draws), n_perturb)
    perturb_idx = np.repeat(np.arange(n_perturb), n_draws)
    z_flat = np.tile(z, n_perturb)
    zhat_flat = zhat.ravel()

    df = pd.DataFrame(
        {
            "distribution": distribution,
            "mean": mean,
            "variance": computed_var,
            "sigma": sigma,
            "draw_idx": draw_idx,
            "perturb_idx": perturb_idx,
            "z": z_flat,
            "zhat": zhat_flat,
        }
    )

    return df, w


def run_z_zhat_simulation():
    """
    Run z/zhat simulations for different distributions.

    Output: simulation_results/z_zhat_simulations.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 3: z/zhat")
    print("=" * 60)

    output_file = sim_dir + "z_zhat_simulations.parquet"

    rng = np.random.default_rng(rng_seed)

    # Check existing data
    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        df_all = pd.read_parquet(output_file)
        print(f"Loaded existing file with {len(df_all)} rows")
    else:
        df_all = None

    # Determine which parameters need to be simulated
    missing = []
    for param_set in zztilde_param_sets:
        distribution = param_set[0]
        mean = param_set[1]
        var = param_set[2] if len(param_set) == 3 else None

        if df_all is not None:
            if distribution == "gamma" and var is not None:
                mask = (
                    (df_all["distribution"] == distribution)
                    & (df_all["mean"] == mean)
                    & (df_all["variance"] == var)
                    & (df_all["sigma"] == zztilde_sigma)
                )
            else:
                mask = (
                    (df_all["distribution"] == distribution)
                    & (df_all["mean"] == mean)
                    & (df_all["sigma"] == zztilde_sigma)
                )
            if not mask.any():
                missing.append(param_set)
        else:
            missing.append(param_set)

    if missing:
        print(f"Running simulations for {len(missing)} parameter sets...")
        new_dfs = []
        for param_set in missing:
            distribution = param_set[0]
            mean = param_set[1]
            var = param_set[2] if len(param_set) == 3 else None

            print(f"  Simulating distribution={distribution}, mean={mean}...")
            df, w = compute_z_zhat(
                distribution=distribution,
                mean=mean,
                n_inputs=zztilde_n_inputs,
                sigma=zztilde_sigma,
                n_draws=zztilde_n_draws,
                n_perturb=zztilde_n_perturb,
                rng=rng,
                var=var,
            )
            new_dfs.append(df)
            print(f"    Generated {len(df)} rows")

        df_new = pd.concat(new_dfs, ignore_index=True)

        if df_all is not None:
            df_all = pd.concat([df_all, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_parquet(output_file)
        print(f"Saved {len(df_all)} total rows to {output_file}")
    else:
        print("All parameter sets already simulated, skipping simulation step")


# ==============================================================================
# SIMULATION 4: PARAMETRIC DISTRIBUTIONS
# ==============================================================================
def get_dist_params(distribution, mean, var):
    """
    Convert mean/variance to native distribution parameters.

    Parameters
    ----------
    distribution : str
        Only 'gamma' is supported.
    mean : float
        Desired mean of the distribution.
    var : float
        Desired variance of the distribution.

    Returns
    -------
    params : tuple
        Native parameters for the distribution.
    """
    if distribution == "gamma":
        shape = mean**2 / var
        scale = var / mean
        return (shape, scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def sample_weights_parametric(distribution, params, n_inputs, rng):
    """
    Sample weights using pre-computed distribution parameters.

    Parameters
    ----------
    distribution : str
        Only 'gamma' is supported.
    params : tuple
        Native parameters from get_dist_params().
    n_inputs : int
        Number of samples to draw.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    w : ndarray
        Array of sampled weights.
    """
    if distribution == "gamma":
        shape, scale = params
        return rng.gamma(shape, scale, n_inputs)


def _parametric_mean_var_grid():
    """
    Build the (mean, var) grid shared by all parametric simulations: manual
    mean_vals (parametric_mean_vals) crossed with a FAFB-informed var_vals
    grid obtained by quantile-median binning of real FAFB neuron variances
    within each mean bin. See run_parametric_simulations for
    provenance/details.

    Returns
    -------
    mean_vals, var_vals : ndarray, ndarray
    """
    # --- Derive the variance grid from FAFB neuron data -------------------
    print("Loading neuron data to determine the variance range...")
    neuron_df = pd.read_parquet(processed_dir + "fafb_neurons.parquet")
    neuron_df = neuron_df[neuron_df["in_deg"] >= shuffle_k_min]

    neuron_df["mean"] = neuron_df["in_strength"] / neuron_df["in_deg"]
    neuron_df["var"] = (neuron_df["sum_w2"] / neuron_df["in_deg"]) - neuron_df[
        "mean"
    ] ** 2

    # Filter to neurons with positive variance
    neuron_df = neuron_df[neuron_df["var"] > 1e-5]

    mu_min, mu_max = neuron_df["mean"].min(), neuron_df["mean"].max()

    mean_vals = np.array(parametric_mean_vals)

    # --- Derive variance grid via quantile binning within mean bins ----------
    # Bin neurons by mean using log-uniform edges matched to the data range
    mean_edges = np.logspace(np.log10(mu_min), np.log10(mu_max), len(mean_vals) + 1)
    neuron_df["mean_bin"] = pd.cut(
        neuron_df["mean"], bins=mean_edges, labels=False, include_lowest=True
    )

    # Within each mean bin, split variances into quantile bins and take medians
    all_var_medians = []
    for i in range(len(mean_vals)):
        mean_mask = neuron_df["mean_bin"] == i
        subset_var = neuron_df.loc[mean_mask, "var"]
        if len(subset_var) == 0:
            continue

        # Quantile bin edges for variance within this mean bin
        var_q = np.percentile(subset_var, np.linspace(0, 100, parametric_n_var + 1))
        var_bin_idx = pd.cut(subset_var, bins=var_q, labels=False, include_lowest=True)

        # Median variance in each quantile bin
        for j in range(parametric_n_var):
            bin_vals = subset_var[var_bin_idx == j]
            if len(bin_vals) > 0:
                all_var_medians.append(bin_vals.median())

    # Pool medians and set var_vals as n_var log-uniform points over that range
    all_var_medians = np.array(all_var_medians)
    var_pool_min, var_pool_max = all_var_medians.min(), all_var_medians.max()
    var_vals = np.logspace(
        np.log10(var_pool_min), np.log10(var_pool_max), parametric_n_var
    )

    print(
        f"  Mean  range: [{mu_min:.2f}, {mu_max:.2f}]  (data)  ->  manual values: {mean_vals}"
    )
    print(
        f"  Var   range: [{var_pool_min:.2f}, {var_pool_max:.2f}]  ->  {parametric_n_var} log-spaced points (from quantile medians)"
    )
    print(f"  mean_vals = {np.array2string(mean_vals, precision=2)}")
    print(f"  var_vals  = {np.array2string(var_vals, precision=2)}")

    return mean_vals, var_vals


def run_parametric_simulation(
    distribution,
    mean_vals,
    var_vals,
    n_inputs,
    n_neurons,
    n_draws,
    n_perturb,
    sigma,
    block_perturb,
    rng,
):
    """
    Run Monte Carlo error simulation for a parametric distribution.

    Parameters
    ----------
    distribution : str
        Only 'gamma' is supported.
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
    sigma : float
        Perturbation scale (noise magnitude).
    block_perturb : int
        Block size for memory-efficient perturbation loop.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 'mean', 'var', 'sim_error'.
    """
    output_file = f"{sim_dir}{distribution}_sim.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"  Skipping {distribution}: {output_file} already exists")
        return pd.read_parquet(output_file)

    error_list = []
    mean_list = []
    var_list = []

    for mean in tqdm(mean_vals, desc=f"  {distribution}"):
        for var in var_vals:
            try:
                params = get_dist_params(distribution, mean, var)
            except ValueError:
                continue

            error = np.full(n_neurons, np.nan, dtype=float)
            for i in range(n_neurons):
                w = sample_weights_parametric(distribution, params, n_inputs, rng)
                l_hat = average_error_fast(
                    w,
                    sigma=sigma,
                    n_draws=n_draws,
                    n_perturb=n_perturb,
                    block_perturb=block_perturb,
                    rng=rng,
                )
                error[i] = l_hat

            error_list.append(np.mean(error))
            mean_list.append(mean)
            var_list.append(var)

    df = pd.DataFrame(
        {
            "mean": np.array(mean_list),
            "var": np.array(var_list),
            "sim_error": np.array(error_list),
        }
    )

    df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    return df


def run_parametric_simulations():
    """
    Run parametric distribution simulations for gamma.

    The mean values are chosen manually (parametric_mean_vals). The variance
    grid is derived from the FAFB neuron data: log-uniform spacing over the
    range of within-bin median variances (see _parametric_mean_var_grid),
    following the same approach as the log-uniform binning in
    simulated_error.py.

    Output files:
        simulation_results/gamma_sim.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 4: Parametric Distributions")
    print("=" * 60)

    mean_vals, var_vals = _parametric_mean_var_grid()

    rng = np.random.default_rng(rng_seed)

    print("Running gamma simulations...")
    run_parametric_simulation(
        "gamma",
        mean_vals,
        var_vals,
        n_inputs=parametric_n_inputs,
        n_neurons=parametric_n_neurons,
        n_draws=parametric_n_draws,
        n_perturb=parametric_n_perturb,
        sigma=error_sigma,
        block_perturb=block_perturb,
        rng=rng,
    )


def run_parametric_simulation_sparse(
    distribution,
    mean_vals,
    var_vals,
    p_fire_vals,
    n_inputs,
    n_neurons,
    n_draws,
    n_perturb,
    sigma,
    block_perturb,
    rng,
):
    """
    Run Monte Carlo error-rate simulation for a parametric distribution swept
    over firing probability p_fire, using the same FAFB-inspired synthetic
    weight sampling as run_parametric_simulation but with
    average_error_fast_biased in place of average_error_fast so each
    neuron's local field is biased to hit a target p_fire.

    Parameters
    ----------
    distribution : str
        Only 'gamma' is supported.
    mean_vals : array-like
        Array of mean values to simulate.
    var_vals : array-like
        Array of variance values to simulate.
    p_fire_vals : array-like
        Target firing probabilities to simulate.
    n_inputs : int
        Number of input weights per neuron.
    n_neurons : int
        Number of neurons to simulate per (mean, var, p_fire) triple.
    n_draws : int
        Number of input draws for Monte Carlo.
    n_perturb : int
        Number of weight perturbation draws.
    sigma : float
        Perturbation scale (noise magnitude).
    block_perturb : int
        Block size for memory-efficient perturbation loop.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 'mean', 'var', 'p_fire', 'sim_error'.
    """
    output_file = f"{sim_dir}{distribution}_sim_sparse.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"  Skipping {distribution} (sparse): {output_file} already exists")
        return pd.read_parquet(output_file)

    error_list = []
    mean_list = []
    var_list = []
    p_fire_list = []

    for mean in tqdm(mean_vals, desc=f"  {distribution} (sparse)"):
        for var in var_vals:
            try:
                params = get_dist_params(distribution, mean, var)
            except ValueError:
                continue

            for p_fire in p_fire_vals:
                error = np.full(n_neurons, np.nan, dtype=float)
                for i in range(n_neurons):
                    w = sample_weights_parametric(distribution, params, n_inputs, rng)
                    l_hat = average_error_fast_biased(
                        w,
                        sigma=sigma,
                        p_fire=p_fire,
                        n_draws=n_draws,
                        n_perturb=n_perturb,
                        block_perturb=block_perturb,
                        rng=rng,
                    )
                    error[i] = l_hat

                error_list.append(np.mean(error))
                mean_list.append(mean)
                var_list.append(var)
                p_fire_list.append(p_fire)

    df = pd.DataFrame(
        {
            "mean": np.array(mean_list),
            "var": np.array(var_list),
            "p_fire": np.array(p_fire_list),
            "sim_error": np.array(error_list),
        }
    )

    df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    return df


def run_parametric_simulations_sparse():
    """
    Run the gamma parametric simulation swept over firing probability
    (sparse_p_fire_vals), reusing the same mean values and FAFB-derived variance grid as
    run_parametric_simulations, for overlay on the sparse-firing error-rate-
    vs-robustness analytical figure (framework.plot_sparse_error_vs_robustness).

    Output file:
        simulation_results/gamma_sim_sparse.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 4B: Parametric Distributions (firing-probability sweep)")
    print("=" * 60)

    mean_vals, var_vals = _parametric_mean_var_grid()

    rng = np.random.default_rng(rng_seed)

    print("Running gamma sparse-firing simulations...")
    run_parametric_simulation_sparse(
        "gamma",
        mean_vals,
        var_vals,
        sparse_p_fire_vals,
        n_inputs=parametric_n_inputs,
        n_neurons=parametric_n_neurons,
        n_draws=parametric_sparse_n_draws,
        n_perturb=parametric_sparse_n_perturb,
        sigma=error_sigma,
        block_perturb=block_perturb,
        rng=rng,
    )


# ==============================================================================
# SIMULATION 5: NETWORK SHUFFLING
# ==============================================================================
def shuffled_neuron_robustness(
    weights,
    scheme,
    conn_type,
    n_threshold,
    normalized,
    return_weights,
):
    """
    Shuffle a single neuron's input weights and return its robustness.

    Generates new weights that preserve in-degree and (approximately)
    in-strength according to the given scheme, then computes the robustness
    metric directly without constructing a full network.
    """
    k = len(weights)
    strength = weights.sum()

    if scheme == "rand_weight":
        if conn_type == "cont":
            wts_ext = np.zeros(k + 1)
            r = np.sort(np.random.rand(k - 1))
            wts_ext[1:-1] = r
            wts_ext[-1] = 1.0
            wts = np.diff(wts_ext) * strength
        else:
            rand_ints = np.random.randint(
                0, int(strength - n_threshold * k) + 1, size=k - 1
            )
            rand_ints = np.append(rand_ints, [0, int(strength - n_threshold * k)])
            rand_ints.sort()
            wts = n_threshold + np.diff(rand_ints)
    elif scheme == "multinomial":
        excess = int(strength - n_threshold * k)
        extra = np.random.multinomial(excess, np.full(k, 1.0 / k))
        wts = n_threshold + extra.astype(float)

    mu_1 = np.mean(wts)  # E[w]
    mu_2 = np.mean(wts**2)  # E[w²]
    r = np.sqrt(mu_2 / mu_1)
    if normalized:
        robustness = (r - np.sqrt(mu_1)) / (np.sqrt(1.0 + mu_1) - np.sqrt(mu_1))
    else:
        robustness = r
    if return_weights:
        return robustness, wts
    return robustness


def global_multinomial_weight_shuffle(A, n_threshold):
    """
    Redistribute all synapses globally across existing connections using
    a multinomial distribution with equal probabilities.

    Each excess synapse (above n_threshold per connection) is independently
    assigned to a uniformly random connection, equivalent to a single
    multinomial draw. Topology and total synapse count are preserved.

    Returns a new weight array of length A.nnz (same order as A.data).
    """
    E = A.nnz
    S = int(A.data.sum())
    S_adj = S - n_threshold * E  # excess synapses above minimum
    extra = np.random.multinomial(S_adj, np.full(E, 1.0 / E))
    return (n_threshold + extra).astype(float)


def run_network_shuffling():
    """
    Shuffle weights for all connectomes and save robustness comparisons.

    For each connectome, compute per-neuron robustness for the original and
    shuffled network. Each output has one row per neuron.
    For FAFB, include the FAFB brain_region metadata.

    Output files:
        simulation_results/mouse_retina_shuffled.parquet  (stars-and-bars shuffle)
        simulation_results/{connectome_name}_shuffled.parquet  (multinomial shuffle,
            all connectomes except mouse retina)
        simulation_results/fafb_shuffled_weights.npz
        simulation_results/fafb_global_shuffled_weights.npz
    """
    print("\n" + "=" * 60)
    print("SIMULATION 5: Network Shuffling")
    print("=" * 60)

    def robustness_per_neuron(A, k_min, normalized):
        """
        Return per-neuron robustness with NaN for undefined cases.
        """
        k = A.getnnz(axis=0)
        r = np.full(A.shape[0], np.nan, dtype=float)
        print("Vector lengths:", len(k), len(r))
        mask = k >= k_min
        if mask.any():
            r[mask] = compute_robustness(A, k_min=k_min, normalized=normalized)
        return r

    # ── Stars-and-bars shuffle (mouse retina only) ─────────────────────────────
    # The other connectomes use the multinomial shuffle below. The stars-and-bars
    # shuffle of BANC lives in supplement_code/shuffling_thresholded.py.
    for name in ["mouse_retina"]:
        output_file = sim_dir + name + "_shuffled.parquet"

        if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue

        conn_type = "cont"
        n_threshold = shuffle_n_threshold_default

        A = load_connectome(name)
        print(f"Shuffling {name} (N={A.shape[0]})...")

        r = robustness_per_neuron(A, k_min=shuffle_k_min, normalized=False)

        r_shuf = np.full(A.shape[0], np.nan, dtype=float)
        for col in range(A.shape[0]):
            start, end = A.indptr[col], A.indptr[col + 1]
            if end - start < shuffle_k_min:
                continue
            r_shuf[col] = shuffled_neuron_robustness(
                A.data[start:end].astype(float),
                scheme="rand_weight",
                conn_type=conn_type,
                n_threshold=n_threshold,
                normalized=False,
                return_weights=False,
            )

        sim_df = pd.DataFrame(
            {
                "robustness": r,
                "shuffled_robustness": r_shuf,
            }
        )
        sim_df.to_parquet(output_file)
        print(f"Saved: {output_file}")

    # ── Multinomial shuffle ────────────────────────────────────────────────────
    for name in connectomes:
        # Skip mouse retina
        if name == "mouse_retina":
            continue

        output_file = sim_dir + name + "_shuffled.parquet"

        if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue

        conn_type = "cont" if name == "mouse_retina" else "disc"
        n_threshold = (
            shuffle_n_threshold_banc if name == "banc" else shuffle_n_threshold_default
        )

        if name == "fafb":
            print("Loading FAFB connections data...")
            conn_df = pd.read_parquet(processed_dir + "fafb_connections.parquet")

            nodes_index = pd.Index(
                pd.unique(
                    pd.concat(
                        [conn_df["pre_root_id"], conn_df["post_root_id"]],
                        ignore_index=True,
                    )
                ),
                name="root_id",
            )
            id_to_idx = pd.Series(np.arange(nodes_index.size), index=nodes_index)
            idx_to_id = nodes_index.to_numpy()

            rows = id_to_idx.reindex(conn_df["pre_root_id"]).to_numpy()
            cols = id_to_idx.reindex(conn_df["post_root_id"]).to_numpy()
            data_vals = conn_df["syn_count"].to_numpy()

            A = coo_matrix(
                (data_vals, (rows, cols)), shape=(nodes_index.size, nodes_index.size)
            ).tocsc()

            neuron_df = pd.read_parquet(processed_dir + "fafb_neurons.parquet")
            region_by_id = (
                neuron_df[["root_id", "brain_region"]]
                .drop_duplicates(subset="root_id")
                .set_index("root_id")["brain_region"]
            )
        else:
            A = load_connectome(name)

        print(f"Shuffling (multinomial) {name} (N={A.shape[0]})...")

        r = robustness_per_neuron(A, k_min=shuffle_k_min, normalized=False)

        save_fafb_weights = name == "fafb"
        if save_fafb_weights:
            shuffled_data_mn = A.data.copy().astype(float)

        r_shuf_mn = np.full(A.shape[0], np.nan, dtype=float)
        for col in range(A.shape[0]):
            start, end = A.indptr[col], A.indptr[col + 1]
            if end - start < shuffle_k_min:
                continue
            if save_fafb_weights:
                r_shuf_mn[col], wts = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="multinomial",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    normalized=False,
                    return_weights=True,
                )
                shuffled_data_mn[start:end] = wts
            else:
                r_shuf_mn[col] = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="multinomial",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    normalized=False,
                    return_weights=False,
                )

        if save_fafb_weights:
            weights_file_mn = sim_dir + "fafb_shuffled_weights.npz"
            np.savez_compressed(weights_file_mn, weights=shuffled_data_mn)
            print(f"Saved multinomial shuffled weights: {weights_file_mn}")

            global_mn_data = global_multinomial_weight_shuffle(
                A, n_threshold=n_threshold
            )
            global_mn_file = sim_dir + "fafb_global_shuffled_weights.npz"
            np.savez_compressed(global_mn_file, weights=global_mn_data)
            print(f"Saved global multinomial shuffled weights: {global_mn_file}")

            # Compute normalized shuffled robustness using same formula as
            # drosophila_robustness_plots.py:
            #   (sqrt(sum_w2/in_strength) - sqrt(mean_w)) / (sqrt(1+mean_w) - sqrt(mean_w))
            A_mn = csc_matrix((shuffled_data_mn, A.indices, A.indptr), shape=A.shape)
            in_deg_mn = np.asarray(A_mn.getnnz(axis=0)).ravel()
            in_str_mn = np.asarray(A_mn.sum(axis=0)).ravel()
            A_mn_sq = A_mn.copy()
            A_mn_sq.data **= 2
            sum_w2_mn = np.asarray(A_mn_sq.sum(axis=0)).ravel()

            valid_mn = in_deg_mn > 1
            mean_w_mn = np.where(
                valid_mn, in_str_mn / np.where(valid_mn, in_deg_mn, 1), 0.0
            )
            norm_rob_mn = np.full(A.shape[0], np.nan)
            num_mn = np.sqrt(sum_w2_mn[valid_mn] / in_str_mn[valid_mn]) - np.sqrt(
                mean_w_mn[valid_mn]
            )
            den_mn = np.sqrt(1.0 + mean_w_mn[valid_mn]) - np.sqrt(mean_w_mn[valid_mn])
            norm_rob_mn[valid_mn] = num_mn / den_mn

        sim_df_mn = pd.DataFrame(
            {
                "robustness": r,
                "shuffled_robustness": r_shuf_mn,
            }
        )

        if name == "fafb":
            sim_df_mn["brain_region"] = region_by_id.reindex(idx_to_id).to_numpy()
            sim_df_mn["root_id"] = idx_to_id
            sim_df_mn["norm_robustness_shuffled"] = norm_rob_mn

        sim_df_mn.to_parquet(output_file)
        print(f"Saved: {output_file}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONNECTOME ROBUSTNESS: MASTER SIMULATION SCRIPT")
    print("=" * 60)
    print(f"Skip existing simulations: {SKIP_EXISTING_SIMULATIONS}")
    print(f"Random seed: {rng_seed}")
    print("=" * 60)

    # Run all simulations
    run_fafb_error_simulation()
    run_fafb_periphery_scoring()
    run_z_zhat_simulation()
    run_parametric_simulations()
    run_parametric_simulations_sparse()
    run_network_shuffling()

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 60)
    print("\nOutput files generated:")
    print(f"  - {sim_dir}fafb_error_data.parquet")
    print(f"  - {sim_dir}fafb_periphery_data.parquet")
    print(f"  - {sim_dir}z_zhat_simulations.parquet")
    print(f"  - {sim_dir}gamma_sim.parquet")
    print(f"  - {sim_dir}gamma_sim_sparse.parquet")
    print(f"  - {sim_dir}*_shuffled.parquet (one per connectome)")
    print(f"  - {sim_dir}fafb_global_shuffled_weights.npz")
    print(f"  - {sim_dir}fafb_shuffled_weights.npz")
    print("=" * 60)
