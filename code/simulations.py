"""
    Master script for running all simulations for the connectome robustness paper.

    Can be run directly to execute all simulations, or imported by figure scripts
    to access individual simulation functions without triggering execution.
-------------------------------------------------------------------------------
created on:
    Tue 4 Feb 2026
-------------------------------------------------------------------------------
last change:
    Thu 17 Apr 2026
-------------------------------------------------------------------------------
notes:
    Run this script once to generate all simulation data:

    1. FlyWire loss simulation -> simulation_results/loss_data.parquet
    2. Periphery scoring -> simulation_results/periphery_data.parquet
    3. z/ztilde simulation -> simulation_results/z_ztilde_simulations.parquet
                              simulation_results/zhat_simulations.parquet
    4. Parametric distributions -> simulation_results/{dist}_sim_{n}.parquet
    5. Network shuffling -> simulation_results/{name}_shuffled.parquet

    Estimated total runtime: 4-8 hours (dominated by FlyWire simulations)

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
from tqdm import tqdm
from params import (
    rng_seed, eta, block_perturb,
    loss_eps, loss_n_draws, loss_n_perturb,
    periphery_n_sim, periphery_threshold, periphery_repeats,
    zztilde_param_sets, zztilde_n_inputs, zztilde_eps, zztilde_n_draws, zztilde_n_perturb,
    parametric_n_neurons, parametric_n_inputs, parametric_n_draws, parametric_n_perturb,
    parametric_n_mean, parametric_n_var,
    shuffle_k_min, shuffle_n_threshold_default, shuffle_n_threshold_banc,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to True to skip simulations if output files already exist
SKIP_EXISTING_SIMULATIONS = True

# ------------------------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------------------------
data_dir = "../../data/"
processed_dir = "../processed_data/"
sim_dir = "../simulation_results/"

# Create output directories if they don't exist
os.makedirs(sim_dir, exist_ok=True)

# ==============================================================================
# CONNECTOME DATA
# ==============================================================================
connectomes = [
    "drosophila_central_brain",
    "drosophila_optic_medulla",
    "c_elegans",
    "platynereis_sensory_motor",
    "mouse_retina",
    "drosophila_whole_brain",
    "drosophila_banc",
    "drosophila_manc",
]


def load_connectome(data_idx, thresholded=False, scheme=None):
    if not thresholded:
        A = load_npz(f"{processed_dir}{connectomes[data_idx]}.npz")
    else:
        A = load_npz(f"{processed_dir}{connectomes[data_idx]}_thresholded.npz")
        if scheme == "remove":
            A.data = A.data - 4
        elif scheme == "clump":
            A.data = 5 * (A.data // 5)
    return A


def compute_robustness(A, eta=1.0, k_min=2, normalized=True):
    k = A.getnnz(axis=0)
    mask = k >= k_min

    s_0 = k[mask]
    s_1 = np.asarray(A.sum(axis=0)).ravel()[mask]

    B = A.copy()
    B.data = B.data**2
    s_2 = np.asarray(B.sum(axis=0)).ravel()[mask]

    if eta == 1.0:
        s_eta = s_1
    elif eta == 2.0:
        s_eta = s_2
    else:
        C = A.copy()
        C.data = C.data**eta
        s_eta = np.asarray(C.sum(axis=0)).ravel()[mask]

    Q = (s_2 / s_eta) ** 0.5
    if normalized:
        Q *= (s_0 / s_1) ** (1.0 - eta / 2)
        Q -= 1.0
    return Q


# ==============================================================================
# SIMULATION FUNCTIONS
# ==============================================================================
def average_error_fast(w, eta, eps, n_draws, n_perturb, block_perturb=128, rng=None):
    """
    Monte Carlo estimate of the loss, vectorized and blocked to reduce memory.

    Parameters
    ----------
    w : array-like
        Weight vector.
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    n_draws : int
        Number of input draws.
    n_perturb : int
        Number of weight perturbation draws.
    block_perturb : int
        Block size for memory-efficient computation.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Estimated loss (fraction of sign flips).
    """
    if rng is None:
        rng = np.random.default_rng()

    w = np.asarray(w, dtype=float)
    n_inputs = w.size
    if n_inputs == 0:
        return np.nan

    # Draw inputs: n_inputs × n_draws
    x = 2.0 * rng.integers(0, 2, size=(n_inputs, n_draws)) - 1.0

    # Draw base Gaussian noise and scale
    w_scale = eps * (w ** (eta / 2.0))
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


def simulate_propagation(A, pool, threshold=0.3, repeats=3, seed=None, max_steps=None):
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
    A, pool, n_sim=100, threshold=0.3, repeats=3, seed=None, max_steps=None
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
# SIMULATION 1: FLYWIRE LOSS
# ==============================================================================
def run_flywire_loss_simulation():
    """
    Run Monte Carlo loss simulation for all FlyWire neurons.

    Input: data/connections_data.parquet
    Output: data/loss_data.parquet
    """
    print("=" * 60)
    print("SIMULATION 1: FlyWire Loss")
    print("=" * 60)

    output_file = sim_dir + "loss_data.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"Skipping: {output_file} already exists")
        return

    # Load data
    print("Loading connections data...")
    conn_df = pd.read_parquet(processed_dir + "connections_data.parquet")

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

    loss = np.full(N, np.nan, dtype=float)
    rng = np.random.default_rng(rng_seed)

    print(f"Simulating loss for {N} neurons...")
    for i, w in enumerate(tqdm(incoming_weights, desc="FlyWire loss")):
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            continue

        l_hat = average_error_fast(
            w,
            eta=eta,
            eps=loss_eps,
            n_draws=loss_n_draws,
            n_perturb=loss_n_perturb,
            block_perturb=block_perturb,
            rng=rng,
        )
        loss[i] = l_hat

    # Save results
    sim_df = pd.DataFrame(
        {
            "root_id": idx_to_id,
            "sim_loss": loss,
        }
    )
    sim_df.to_parquet(output_file)
    print(f"Saved: {output_file}")


# ==============================================================================
# SIMULATION 2: FLYWIRE PERIPHERY SCORING
# ==============================================================================
def run_flywire_periphery_scoring():
    """
    Run stochastic propagation to compute peripherality scores.

    Input: data/connections_data.parquet, data/neuron_data.parquet
    Output: data/periphery_data.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 2: FlyWire Periphery Scoring")
    print("=" * 60)

    output_file = sim_dir + "periphery_data.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"Skipping: {output_file} already exists")
        return

    # Load data
    print("Loading data...")
    conn_df = pd.read_parquet(processed_dir + "connections_data.parquet")
    neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")

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
    optic_list = ["R1-6", "R7", "R8"]
    olfactory_list = ["ORN"]

    optic_re = "|".join(map(re.escape, optic_list))
    olfactory_re = "|".join(map(re.escape, olfactory_list))

    optic_mask = neuron_df["primary_type"].str.contains(optic_re)
    olfactory_mask = neuron_df["primary_type"].str.contains(olfactory_re)
    joint_mask = optic_mask | olfactory_mask

    masks = [optic_mask, olfactory_mask, joint_mask]
    labels = ["optic", "olfactory", "joint"]

    sim_df = pd.DataFrame({"root_id": idx_to_id})

    for i, mask in enumerate(masks):
        print(f"\nProcessing {labels[i]} seed set...")
        selected_ids = neuron_df.loc[mask, "root_id"].values
        seed_set = np.array(
            [id_to_idx[rid] for rid in selected_ids if rid in id_to_idx]
        )

        avg_dist, frac_reached = average_propagation(
            A, seed_set, periphery_n_sim,
            threshold=periphery_threshold, repeats=periphery_repeats, seed=rng_seed,
        )
        print(f"Fraction reached from {labels[i]} seed: {np.mean(frac_reached):.3f}")

        sim_df[f"distance_{labels[i]}"] = avg_dist

    # Save results
    sim_df.to_parquet(output_file)
    print(f"Saved: {output_file}")


# ==============================================================================
# SIMULATION 3: Z/ZTILDE
# ==============================================================================
def generate_weights_zztilde(distribution, mean, n_inputs, rng, var=None):
    """
    Generate weights from the specified distribution for z/ztilde simulation.

    Parameters
    ----------
    distribution : str
        One of 'dirac', 'poisson', 'pareto', 'gamma', or 'lognormal'.
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
    elif distribution == "poisson":
        w = rng.poisson(mean, n_inputs).astype(float)
        computed_var = mean
    elif distribution == "pareto":
        if mean <= 1:
            raise ValueError(f"Pareto distribution requires mean > 1, got {mean}")
        alpha = mean / (mean - 1)
        w = rng.pareto(alpha, n_inputs) + 1
        if alpha > 2:
            computed_var = alpha / ((alpha - 1) ** 2 * (alpha - 2))
        else:
            computed_var = np.inf
    elif distribution == "gamma":
        if var is None:
            raise ValueError("Gamma distribution requires variance parameter")
        theta = var / mean
        alpha = mean / theta
        w = rng.gamma(alpha, theta, n_inputs)
        computed_var = var
    elif distribution == "lognormal":
        if var is None:
            raise ValueError("Lognormal distribution requires variance parameter")
        sigma2 = np.log(1.0 + var / mean**2)
        mu = np.log(mean) - sigma2 / 2.0
        w = rng.lognormal(mu, np.sqrt(sigma2), n_inputs)
        computed_var = var
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return w, computed_var


def compute_z_ztilde(
    distribution, mean, n_inputs, eta, eps, n_draws, n_perturb, rng=None, var=None
):
    """
    Compute z and ztilde values for a weight vector drawn from the specified distribution.

    Parameters
    ----------
    distribution : str
        Distribution type.
    mean : float
        Mean parameter.
    n_inputs : int
        Number of input dimensions.
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    n_draws : int
        Number of input draws.
    n_perturb : int
        Number of perturbation draws.
    rng : numpy.random.Generator, optional
        Random number generator.
    var : float, optional
        Variance (required for 'gamma').

    Returns
    -------
    df : pd.DataFrame
        Long-form DataFrame with z and ztilde values.
    w : ndarray
        The generated weight vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    w, computed_var = generate_weights_zztilde(distribution, mean, n_inputs, rng, var)

    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))
    base_noise = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb))
    w_hat = base_noise * (w ** (eta / 2.0))[:, None]

    z = w @ x
    delta = eps * (w_hat.T @ x)
    ztilde = z[None, :] + delta

    draw_idx = np.tile(np.arange(n_draws), n_perturb)
    perturb_idx = np.repeat(np.arange(n_perturb), n_draws)
    z_flat = np.tile(z, n_perturb)
    ztilde_flat = ztilde.ravel()
    zhat_flat = ztilde_flat - z_flat

    df = pd.DataFrame(
        {
            "distribution": distribution,
            "mean": mean,
            "variance": computed_var,
            "eps": eps,
            "draw_idx": draw_idx,
            "perturb_idx": perturb_idx,
            "z": z_flat,
            "ztilde": ztilde_flat,
            "zhat": zhat_flat,
        }
    )

    return df, w


def run_zztilde_simulation():
    """
    Run z/ztilde simulations for different distributions.

    Output: simulation_results/z_ztilde_simulations.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 3: z/ztilde")
    print("=" * 60)

    output_file = sim_dir + "z_ztilde_simulations.parquet"

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
                    & (df_all["eps"] == zztilde_eps)
                )
            else:
                mask = (
                    (df_all["distribution"] == distribution)
                    & (df_all["mean"] == mean)
                    & (df_all["eps"] == zztilde_eps)
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
            df, w = compute_z_ztilde(
                distribution=distribution,
                mean=mean,
                n_inputs=zztilde_n_inputs,
                eta=eta,
                eps=zztilde_eps,
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

    zhat_file = sim_dir + "zhat_simulations.parquet"
    df_zhat = df_all[["distribution", "mean", "variance", "eps", "draw_idx", "perturb_idx", "zhat"]]
    df_zhat.to_parquet(zhat_file)
    print(f"Saved zhat to {zhat_file}")


# ==============================================================================
# SIMULATION 4: PARAMETRIC DISTRIBUTIONS
# ==============================================================================
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
    if distribution == "lognormal":
        sigma = np.sqrt(np.log(1.0 + var / mean**2))
        mu = np.log(mean) - sigma**2 / 2.0
        return (mu, sigma)
    elif distribution == "lomax":
        if var <= mean**2:
            raise ValueError(
                "Lomax requires var > mean^2 for finite variance (alpha > 2)"
            )
        a = 2.0 * var / (var - mean**2)
        w0 = (a - 1.0) * mean
        return (a, w0)
    elif distribution == "gamma":
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
        One of 'lognormal', 'lomax', or 'gamma'.
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
    if distribution == "lognormal":
        mu, sigma = params
        return rng.lognormal(mu, sigma, n_inputs)
    elif distribution == "lomax":
        a, w0 = params
        r = rng.random(n_inputs)
        return ((1 - r) ** (-1 / a) - 1.0) * w0
    elif distribution == "gamma":
        shape, scale = params
        return rng.gamma(shape, scale, n_inputs)


def run_parametric_simulation(
    distribution,
    mean_vals,
    var_vals,
    n_inputs=1000,
    n_neurons=1000,
    n_draws=1000,
    n_perturb=1000,
    eta=1.0,
    eps=1.0,
    rng=None,
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
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns 'mean', 'var', 'sim_loss'.
    """
    if rng is None:
        rng = np.random.default_rng()

    output_file = f"{sim_dir}{distribution}_sim_{n_inputs}.parquet"

    if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
        print(f"  Skipping {distribution}: {output_file} already exists")
        return pd.read_parquet(output_file)

    loss_list = []
    mean_list = []
    var_list = []

    for mean in tqdm(mean_vals, desc=f"  {distribution}"):
        for var in var_vals:
            try:
                params = get_dist_params(distribution, mean, var)
            except ValueError:
                continue

            loss = np.full(n_neurons, np.nan, dtype=float)
            for i in range(n_neurons):
                w = sample_weights_parametric(distribution, params, n_inputs, rng)
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

    df = pd.DataFrame(
        {
            "mean": np.array(mean_list),
            "var": np.array(var_list),
            "sim_loss": np.array(loss_list),
        }
    )

    df.to_parquet(output_file)
    print(f"  Saved: {output_file}")

    return df


def run_parametric_simulations():
    """
    Run parametric distribution simulations for lognormal, lomax, and gamma.

    Mean and variance grids are derived from the FlyWire neuron data using a
    log-uniform spacing (excluding endpoints) over the observed data range,
    following the same approach as the log-uniform binning in
    flywire_loss_analysis.py.

    Output files:
        simulation_results/lognormal_sim_100.parquet
        simulation_results/lomax_sim_100.parquet
        simulation_results/gamma_sim_100.parquet
    """
    print("\n" + "=" * 60)
    print("SIMULATION 4: Parametric Distributions")
    print("=" * 60)

    # --- Derive mean/variance grids from FlyWire neuron data -----------------
    print("Loading neuron data to determine mean/variance ranges...")
    neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")
    neuron_df = neuron_df[neuron_df["in_deg"] >= shuffle_k_min]

    neuron_df["mean"] = neuron_df["in_strength"] / neuron_df["in_deg"]
    neuron_df["var"] = (neuron_df["sum_w2"] / neuron_df["in_deg"]) - neuron_df[
        "mean"
    ] ** 2

    # Filter to neurons with positive variance
    neuron_df = neuron_df[neuron_df["var"] > 1e-5]

    mu_min, mu_max = neuron_df["mean"].min(), neuron_df["mean"].max()

    # Log-uniform mean bin edges (n_mean + 1 edges -> n_mean bins),
    # matching the binning in flywire_loss_analysis.py
    mean_edges = np.logspace(np.log10(mu_min), np.log10(mu_max), parametric_n_mean + 1)

    # --- Derive variance grid via quantile binning within mean bins ----------
    # Bin neurons by mean using log-uniform edges
    neuron_df["mean_bin"] = pd.cut(
        neuron_df["mean"], bins=mean_edges, labels=False, include_lowest=True
    )

    # Representative mean for each bin: median of neurons in that bin
    mean_vals = np.array(
        [
            neuron_df.loc[neuron_df["mean_bin"] == i, "mean"].median()
            for i in range(parametric_n_mean)
        ]
    )

    # Within each mean bin, split variances into quantile bins and take medians
    all_var_medians = []
    for i in range(parametric_n_mean):
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
    var_vals = np.logspace(np.log10(var_pool_min), np.log10(var_pool_max), parametric_n_var)

    print(
        f"  Mean  range: [{mu_min:.2f}, {mu_max:.2f}]  ->  {parametric_n_mean} log-spaced points"
    )
    print(
        f"  Var   range: [{var_pool_min:.2f}, {var_pool_max:.2f}]  ->  {parametric_n_var} log-spaced points (from quantile medians)"
    )
    print(f"  mean_vals = {np.array2string(mean_vals, precision=2)}")
    print(f"  var_vals  = {np.array2string(var_vals, precision=2)}")

    rng = np.random.default_rng(rng_seed)

    # print("Running lognormal simulations...")
    # run_parametric_simulation('lognormal', mean_vals, var_vals, n_inputs=parametric_n_inputs,
    #                            n_neurons=parametric_n_neurons, n_draws=parametric_n_draws, n_perturb=parametric_n_perturb, rng=rng)

    # print("Running lomax simulations...")
    # run_parametric_simulation('lomax', mean_vals, var_vals, n_inputs=parametric_n_inputs,
    #                            n_neurons=parametric_n_neurons, n_draws=parametric_n_draws, n_perturb=parametric_n_perturb, rng=rng)

    print("Running gamma simulations...")
    run_parametric_simulation(
        "gamma",
        mean_vals,
        var_vals,
        n_inputs=parametric_n_inputs,
        n_neurons=parametric_n_neurons,
        n_draws=parametric_n_draws,
        n_perturb=parametric_n_perturb,
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
    eta=1.0,
    normalized=False,
    return_weights=False,
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
    elif scheme == "poisson":
        wts = 1.0 + np.random.poisson(strength / k - 1.0, size=k)
    elif scheme == "concentrated":
        wts = np.ones(k)
        wts[0] = strength - k + 1

    s_2 = np.sum(wts**2)
    s_eta = np.sum(wts**eta)
    Q = (s_2 / s_eta) ** 0.5
    if normalized:
        Q *= (k / np.sum(wts)) ** (1.0 - eta / 2)
        Q -= 1.0
    if return_weights:
        return Q, wts
    return Q


def global_weight_shuffle(A, n_threshold=0):
    """
    Redistribute all synapses globally across existing connections.

    Unlike the per-neuron shuffle, this places synapses without any per-neuron
    constraint: each of the A.nnz connections competes for the full synapse pool.
    Topology (sparsity pattern) is preserved; total synapse count is preserved.
    Uses the same stars-and-bars sampling as the per-neuron scheme.

    Returns a new weight array of length A.nnz (same order as A.data).
    """
    E = A.nnz
    S = int(A.data.sum())
    S_adj = S - n_threshold * E  # synapses beyond the per-connection minimum
    rand_ints = np.random.randint(0, S_adj + 1, size=E - 1)
    rand_ints = np.append(rand_ints, [0, S_adj])
    rand_ints.sort()
    return (n_threshold + np.diff(rand_ints)).astype(float)


def global_multinomial_weight_shuffle(A, n_threshold=0):
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
    For FAFB (data_idx=5), include the FAFB brain_region metadata.

    Output files:
        simulation_results/{connectome_name}_shuffled.parquet  (all 8 connectomes)
    """
    print("\n" + "=" * 60)
    print("SIMULATION 5: Network Shuffling")
    print("=" * 60)

    def robustness_per_neuron(A, eta=eta, k_min=shuffle_k_min, normalized=False):
        """
        Return per-neuron robustness with NaN for undefined cases.
        """
        k = A.getnnz(axis=0)
        q = np.full(A.shape[0], np.nan, dtype=float)
        print("Vector lengths:", len(k), len(q))
        mask = k >= k_min
        if mask.any():
            q[mask] = compute_robustness(A, eta=eta, k_min=k_min, normalized=normalized)
        return q

    for data_idx in range(len(connectomes)):
        output_file = sim_dir + connectomes[data_idx] + "_shuffled.parquet"

        if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue

        conn_type = "cont" if data_idx == 4 else "disc"
        n_threshold = shuffle_n_threshold_banc if data_idx == 6 else shuffle_n_threshold_default

        if data_idx == 5:
            print("Loading FAFB connections data...")
            conn_df = pd.read_parquet(processed_dir + "connections_data.parquet")

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

            neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")
            region_by_id = (
                neuron_df[["root_id", "brain_region"]]
                .drop_duplicates(subset="root_id")
                .set_index("root_id")["brain_region"]
            )
        else:
            A = load_connectome(data_idx)

        print(f"Shuffling {connectomes[data_idx]} (N={A.shape[0]})...")

        q = robustness_per_neuron(A)

        save_fafb_weights = data_idx == 5
        if save_fafb_weights:
            shuffled_data = A.data.copy().astype(float)

        q_rand = np.full(A.shape[0], np.nan, dtype=float)
        for col in range(A.shape[0]):
            start, end = A.indptr[col], A.indptr[col + 1]
            if end - start < shuffle_k_min:
                continue
            if save_fafb_weights:
                q_rand[col], wts = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="rand_weight",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    eta=eta,
                    normalized=False,
                    return_weights=True,
                )
                shuffled_data[start:end] = wts
            else:
                q_rand[col] = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="rand_weight",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    eta=eta,
                    normalized=False,
                )

        if save_fafb_weights:
            weights_file = sim_dir + "drosophila_whole_brain_shuffled_weights.npz"
            np.savez_compressed(weights_file, weights=shuffled_data)
            print(f"Saved shuffled weights: {weights_file}")

            global_data = global_weight_shuffle(A, n_threshold=n_threshold)
            global_file = sim_dir + "drosophila_whole_brain_global_shuffled_weights.npz"
            np.savez_compressed(global_file, weights=global_data)
            print(f"Saved global shuffled weights: {global_file}")

            global_mn_data = global_multinomial_weight_shuffle(
                A, n_threshold=n_threshold
            )
            global_mn_file = (
                sim_dir
                + "drosophila_whole_brain_global_shuffled_weights_multinomial.npz"
            )
            np.savez_compressed(global_mn_file, weights=global_mn_data)
            print(f"Saved global multinomial shuffled weights: {global_mn_file}")

        sim_df = pd.DataFrame(
            {
                "robustness": q,
                "shuffled_robustness": q_rand,
            }
        )

        if data_idx == 5:
            sim_df["brain_region"] = region_by_id.reindex(idx_to_id).to_numpy()

        sim_df.to_parquet(output_file)
        print(f"Saved: {output_file}")

    # ── Multinomial shuffle ────────────────────────────────────────────────────
    for data_idx in range(len(connectomes)):
        # Skip mouse retina
        if data_idx == 4:
            continue

        output_file = sim_dir + connectomes[data_idx] + "_shuffled_multinomial.parquet"

        if SKIP_EXISTING_SIMULATIONS and os.path.exists(output_file):
            print(f"Skipping: {output_file} already exists")
            continue

        conn_type = "cont" if data_idx == 4 else "disc"
        n_threshold = shuffle_n_threshold_banc if data_idx == 6 else shuffle_n_threshold_default

        if data_idx == 5:
            print("Loading FAFB connections data...")
            conn_df = pd.read_parquet(processed_dir + "connections_data.parquet")

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

            neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")
            region_by_id = (
                neuron_df[["root_id", "brain_region"]]
                .drop_duplicates(subset="root_id")
                .set_index("root_id")["brain_region"]
            )
        else:
            A = load_connectome(data_idx)

        print(f"Shuffling (multinomial) {connectomes[data_idx]} (N={A.shape[0]})...")

        q = robustness_per_neuron(A)

        save_fafb_weights = data_idx == 5
        if save_fafb_weights:
            shuffled_data_mn = A.data.copy().astype(float)

        q_rand_mn = np.full(A.shape[0], np.nan, dtype=float)
        for col in range(A.shape[0]):
            start, end = A.indptr[col], A.indptr[col + 1]
            if end - start < shuffle_k_min:
                continue
            if save_fafb_weights:
                q_rand_mn[col], wts = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="multinomial",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    eta=eta,
                    normalized=False,
                    return_weights=True,
                )
                shuffled_data_mn[start:end] = wts
            else:
                q_rand_mn[col] = shuffled_neuron_robustness(
                    A.data[start:end].astype(float),
                    scheme="multinomial",
                    conn_type=conn_type,
                    n_threshold=n_threshold,
                    eta=eta,
                    normalized=False,
                )

        if save_fafb_weights:
            weights_file_mn = (
                sim_dir + "drosophila_whole_brain_shuffled_weights_multinomial.npz"
            )
            np.savez_compressed(weights_file_mn, weights=shuffled_data_mn)
            print(f"Saved multinomial shuffled weights: {weights_file_mn}")

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
                "robustness": q,
                "shuffled_robustness": q_rand_mn,
            }
        )

        if data_idx == 5:
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
    run_flywire_loss_simulation()
    run_flywire_periphery_scoring()
    run_zztilde_simulation()
    run_parametric_simulations()
    run_network_shuffling()

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 60)
    print("\nOutput files generated:")
    print(f"  - {sim_dir}loss_data.parquet")
    print(f"  - {sim_dir}periphery_data.parquet")
    print(f"  - {sim_dir}z_ztilde_simulations.parquet")
    print(f"  - {sim_dir}zhat_simulations.parquet")
    print(f"  - {sim_dir}lognormal_sim_100.parquet")
    print(f"  - {sim_dir}lomax_sim_100.parquet")
    print(f"  - {sim_dir}gamma_sim_100.parquet")
    print(f"  - {sim_dir}*_shuffled.parquet (8 connectomes)")
    print(f"  - {sim_dir}*_shuffled_multinomial.parquet (8 connectomes)")
    print(f"  - {sim_dir}drosophila_whole_brain_global_shuffled_weights.npz")
print(f"  - {sim_dir}drosophila_whole_brain_global_shuffled_weights_multinomial.npz")
print(f"  - {sim_dir}drosophila_whole_brain_shuffled_weights_multinomial.npz")
print("=" * 60)
