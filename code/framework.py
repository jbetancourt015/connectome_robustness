"""
    Generates all framework figures for the manuscript.
-------------------------------------------------------------------------------
created on:
    Sun 13 Apr 2026
-------------------------------------------------------------------------------
last change:
    Tue 29 Apr 2026
-------------------------------------------------------------------------------
notes:
    Generates the six main framework figures:
      1. Decision boundary cartoon (classification_plane.svg)
      2. Ellipse cartoon with principal axes (2d_local_field_distribution.svg)
      3. Weight distribution PDFs (pdf_*.svg)
      4. Analytical Gaussian heatmaps (gaussian_*.svg)
      5. Simulated 2D histograms (hist_*.svg)
      6. Loss vs variance plot (*_simulation_var.svg)

    Outputs to figures/framework/.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm as lognorm_dist
import logging
from params import (
    rng_seed, eta,
    zztilde_param_sets, zztilde_n_inputs, zztilde_eps, zztilde_n_draws, zztilde_n_perturb,
    shuffle_k_min,
)
from simulations import run_zztilde_simulation

# ------------------------------------------------------------------------------
# MATPLOTLIB CONFIGURATION
# ------------------------------------------------------------------------------
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "font.family": "Helvetica",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

mpl.rcParams["figure.dpi"] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# SHARED CONFIGURATION
# ------------------------------------------------------------------------------
con_colors = (
    np.array(
        [
            [0, 77, 128],
            [181, 23, 0],
            [1, 113, 0],
            [242, 112, 0],
            [120, 0, 150],
            [0, 168, 157],
            [203, 41, 123],
            [0, 0, 0],
        ]
    )
    / 255
)

# Directory paths
sim_dir       = "../simulation_results/"
processed_dir = "../processed_data/"
fig_dir       = "../../figures/framework/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Single consolidated simulation file for z/ztilde
sim_file = sim_dir + "z_ztilde_simulations.parquet"
zhat_sim_file = sim_dir + "zhat_simulations.parquet"

# Reference size (page width in mm) for scaling figure dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from mm reference)
width_sm = 0.15 * pg_width / mm_to_in
height_sm = 0.15 * pg_width / mm_to_in
width_md = (1/3.) * pg_width / mm_to_in
height_md = (1/3.) * pg_width / mm_to_in
width_lg = 0.45 * pg_width / mm_to_in
height_lg = 0.45 * pg_width / mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins_sm = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
fig_margins_md = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_lg = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)

# Dark-cool colormap for parametric loss plots
dark_cool = mcolors.LinearSegmentedColormap.from_list(
    "dark_cool", ["#0b3c49", "#1f5c8b", "#4a4aa8", "#7a3fa0", "#9b2f8a"]
)
if "dark_cool" not in plt.colormaps:
    plt.colormaps.register(dark_cool)

# z/ztilde simulation parameters (defined in params.py)
param_sets = zztilde_param_sets
n_inputs   = zztilde_n_inputs
eps        = zztilde_eps
n_draws    = zztilde_n_draws
n_perturb  = zztilde_n_perturb

# Histogram parameters
alpha_min = 0.2
n_bins = 20

# Distribution PDF plot parameters — set to a float to fix the y-axis for continuous PDFs
pdf_y_max = 1.

# Parametric loss parameters
n_pred = 200

# Colors for each parameter set (equally spaced from viridis, capped at 80%)
n_colors = len(param_sets)
cmap = plt.get_cmap("viridis")
viridis_max = 0.8
sim_colors = [
    plt.cm.viridis(i * viridis_max / (n_colors - 1))[:3] for i in range(n_colors)
]


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================
def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    """Create a colormap that fades from transparent to a solid color."""
    bottom = (*rgb, alpha_min)
    top = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)


def _outer_tick(lim):
    """Sig fig just below 75% of lim (e.g. lim=23.4 → 0.75*23.4=17.6 → 10)."""
    val = 0.75 * lim
    exp = int(np.floor(np.log10(val)))
    return int(np.floor(val / 10**exp) * 10**exp)


def mean_bin_median_norm(n_mean_bins=4):
    """
    Compute LogNorm anchored to log-midpoint mean values from FlyWire neuron data.
    Matches the colorbar range used in simulated_loss.py.
    """
    neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")
    neuron_df = neuron_df[neuron_df["in_deg"] >= shuffle_k_min]
    neuron_df["mean"] = neuron_df["in_strength"] / neuron_df["in_deg"]
    neuron_df["var"] = (neuron_df["sum_w2"] / neuron_df["in_deg"]) - neuron_df["mean"] ** 2
    df_nonneg = neuron_df[neuron_df["var"] > 1e-5].copy()

    mean_bins = np.logspace(
        np.log10(df_nonneg["mean"].min()), np.log10(df_nonneg["mean"].max()), n_mean_bins + 1
    )
    mean_mids = np.sqrt(mean_bins[:-1] * mean_bins[1:])
    print("Mean bin log-midpoints used for colors:")
    for i, m in enumerate(mean_mids):
        print(f"  Bin {i}: {m:.4f}")
    return mcolors.LogNorm(vmin=1, vmax=mean_mids.max())


def draw_ellipse(R, rho, ax, color, lim=3.0):
    """
    Draw an ellipse satisfying x^2 + y^2 - 2*rho*x*y = R^2*sqrt(1 - rho^2).

    Parameters
    ----------
    R : float
        Radius scaling factor.
    rho : float
        Correlation parameter controlling ellipse shape.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    color : np.ndarray
        RGB color array.
    lim : float
        Plot limits.
    """
    xx = np.linspace(-lim, lim, 400)
    yy = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(xx, yy)
    Z = X**2 + Y**2 - 2 * rho * X * Y
    level = (R**2) * np.sqrt(1 - rho**2)
    ax.contourf(X, Y, Z, levels=[0, level], colors=[color], alpha=0.3)
    ax.contour(X, Y, Z, levels=[level], colors=[color], linewidths=2)


def general_loss(mean, var):
    """Predicted loss from mean and variance."""
    rob = np.sqrt(mean + var / mean)
    return (1 / np.pi) * np.arccos((1.0 + 1.0 / rob**2) ** (-1 / 2))


# ==============================================================================
# SIMULATION FUNCTIONS (z/ztilde)
# ==============================================================================
def generate_weights(distribution, mean, n_inputs, rng, var=None):
    """
    Generate weights from the specified distribution.

    Parameters
    ----------
    distribution : str
        Distribution type: 'dirac', 'poisson', 'pareto', 'gamma', or 'lognormal'.
    mean : float
        Mean of the distribution.
    n_inputs : int
        Number of weights to generate.
    rng : numpy.random.Generator
        Random number generator.
    var : float, optional
        Variance (required for 'gamma' and 'lognormal').

    Returns
    -------
    w : np.ndarray
        Generated weight vector.
    computed_var : float
        The variance of the distribution (computed or provided).
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
        sigma2 = np.log(1 + var / mean**2)
        mu = np.log(mean) - sigma2 / 2
        w = rng.lognormal(mu, np.sqrt(sigma2), n_inputs)
        computed_var = var

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return w, computed_var


def compute_z_ztilde(
    distribution,
    mean,
    n_inputs,
    eta,
    eps,
    n_draws,
    n_perturb,
    rng=None,
    var=None,
):
    """
    Compute z and ztilde values for a weight vector drawn from the specified distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    w, computed_var = generate_weights(distribution, mean, n_inputs, rng, var)
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


def parse_param_set(param_set):
    """Parse a parameter set tuple into (distribution, mean, variance)."""
    if len(param_set) == 2:
        distribution, mean = param_set
        var = None
    elif len(param_set) == 3:
        distribution, mean, var = param_set
    else:
        raise ValueError(f"Invalid param_set format: {param_set}")
    return distribution, mean, var


def get_missing_params(param_sets, eps, existing_df=None):
    """Return parameter sets not already in the file, checking eps too."""
    if existing_df is None:
        return list(param_sets)

    missing = []
    for param_set in param_sets:
        distribution, mean, var = parse_param_set(param_set)

        if distribution in ("gamma", "lognormal"):
            mask = (
                (existing_df["distribution"] == distribution)
                & (existing_df["mean"] == mean)
                & (existing_df["variance"] == var)
                & (existing_df["eps"] == eps)
            )
        else:
            mask = (
                (existing_df["distribution"] == distribution)
                & (existing_df["mean"] == mean)
                & (existing_df["eps"] == eps)
            )

        if not mask.any():
            missing.append(param_set)

    return missing


# ==============================================================================
# PLOT FUNCTIONS
# ==============================================================================
def load_and_normalize(df_all, distribution, mean, var=None, eps=None):
    """
    Filter and normalize data for a specific parameter set.

    Returns
    -------
    z_norm, ztilde_norm : np.ndarray
        Normalized local field values (divided by std).
    computed_var : float
    """
    if distribution in ("gamma", "lognormal") and var is not None:
        df = df_all[
            (df_all["distribution"] == distribution)
            & (df_all["mean"] == mean)
            & (df_all["variance"] == var)
        ]
    else:
        df = df_all[(df_all["distribution"] == distribution) & (df_all["mean"] == mean)]

    if eps is not None:
        df = df[df["eps"] == eps]

    z = df["z"].values
    ztilde = df["ztilde"].values
    computed_var = df["variance"].iloc[0]

    print("Correlation:", np.corrcoef(z, ztilde)[0, 1])

    z_norm = z / np.std(z)
    ztilde_norm = ztilde / np.std(ztilde)

    return z_norm, ztilde_norm, computed_var


def plot_local_field_hist(z, zhat, color, fname, last=False, xlim=300.0, ylim=600.0):
    """
    Create a 2D heatmap histogram of (z, zhat) values.

    Parameters
    ----------
    z : np.ndarray
        Raw (unnormalized) local field values.
    zhat : np.ndarray
        Raw perturbation values (ztilde - z).
    color : np.ndarray
        RGB color array for the colormap.
    fname : str
        Filename suffix for saving.
    last : bool
        If True, this is the highest-variance panel: show x ticks (bottom spine).
        No y ticks are shown on any panel.
    xlim, ylim : float
        Symmetric axis limits; shared across all panels.
    """
    fig, ax_scatter = plt.subplots(figsize=(width_sm, height_sm))

    xbins = np.linspace(-xlim, xlim, n_bins)
    ybins = np.linspace(-ylim, ylim, n_bins)

    _, _, _, hist_img = ax_scatter.hist2d(
        z,
        zhat,
        bins=[xbins, ybins],
        density=True,
        cmap=fade_to_color_cmap(color, alpha_min=0.0),
    )

    for loc in ["left", "right", "top", "bottom"]:
        ax_scatter.spines[loc].set_visible(True)
    ax_scatter.set_frame_on(True)

    # Error shading: region between y-axis and y=-x (equivalent to Q2/Q4 in z-ztilde space)
    x_r = np.array([0.0, xlim])
    ax_scatter.fill_between(x_r, -ylim, -x_r, color=con_colors[1], alpha=0.3)
    x_l = np.array([-xlim, 0.0])
    ax_scatter.fill_between(x_l, -x_l, ylim, color=con_colors[1], alpha=0.3)

    ax_scatter.plot([0.0, 0.0], [-ylim, ylim], c="k", alpha=0.5, lw=1)
    ax_scatter.plot([-xlim, xlim], [0.0, 0.0], c="k", alpha=0.5, lw=1)
    _diag = max(xlim, ylim)
    ax_scatter.plot([-_diag, _diag], [_diag, -_diag], c="k", alpha=0.5, lw=1)

    ax_scatter.set_xlim(-xlim, xlim)
    ax_scatter.set_ylim(-ylim, ylim)
    ax_scatter.set_yticks([])
    _xt = _outer_tick(xlim)
    ax_scatter.set_xticks([-_xt, 0, _xt] if last else [])

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"hist_{fname}.svg", dpi=600)

    fig.colorbar(hist_img, ax=ax_scatter).set_label("Probability density")
    ax_scatter.set_xlabel(r"$z$")
    ax_scatter.set_ylabel(r"$\hat{z}$")

    plt.show()


def plot_gaussian_heatmap(sigma_x, sigma_y, color, fname, last=False,
                          xlim=300.0, ylim=600.0, n_grid=200):
    """
    Plot an analytical 2D Gaussian heatmap for (z, zhat) with zero covariance.

    The density is N(0, diag(sigma_x^2, sigma_y^2)), where:
        sigma_x^2 = sum_i w_i^2  (analytically: n_inputs * E[w^2])
        sigma_y^2 = eps^2 * sum_i w_i  (analytically: eps^2 * n_inputs * E[w])

    Parameters
    ----------
    sigma_x, sigma_y : float
        Standard deviations along the z and zhat axes respectively.
    color : np.ndarray
        RGB color array for the colormap.
    fname : str
        Filename suffix for saving.
    last : bool
        If True, this is the highest-variance panel: show x ticks (bottom spine).
        All panels always show y ticks (left spine).
    xlim, ylim : float
        Symmetric axis limits; shared across all panels.
    n_grid : int
        Number of grid points in each dimension.
    """
    x = np.linspace(-xlim, xlim, n_grid)
    y = np.linspace(-ylim, ylim, n_grid)
    X, Y = np.meshgrid(x, y)

    Z = (
        np.exp(-X**2 / (2 * sigma_x**2) - Y**2 / (2 * sigma_y**2))
        / (2 * np.pi * sigma_x * sigma_y)
    )

    fig, ax = plt.subplots(figsize=(width_sm, height_sm))

    mesh = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=fade_to_color_cmap(color, alpha_min=0.0),
        shading="auto",
        rasterized=True,
    )

    for loc in ["left", "right", "top", "bottom"]:
        ax.spines[loc].set_visible(True)
    ax.set_frame_on(True)

    # Error shading: region between y-axis and y=-x (equivalent to Q2/Q4 in z-ztilde space)
    x_r = np.array([0.0, xlim])
    ax.fill_between(x_r, -ylim, -x_r, color=con_colors[1], alpha=0.3)
    x_l = np.array([-xlim, 0.0])
    ax.fill_between(x_l, -x_l, ylim, color=con_colors[1], alpha=0.3)

    ax.plot([0.0, 0.0], [-ylim, ylim], c="k", alpha=0.5, lw=1)
    ax.plot([-xlim, xlim], [0.0, 0.0], c="k", alpha=0.5, lw=1)
    _diag = max(xlim, ylim)
    ax.plot([-_diag, _diag], [_diag, -_diag], c="k", alpha=0.5, lw=1)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    _yt = _outer_tick(ylim)
    ax.set_yticks([-_yt, 0, _yt])
    _xt = _outer_tick(xlim)
    ax.set_xticks([-_xt, 0, _xt] if last else [])

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"gaussian_{fname}.svg", dpi=600)

    fig.colorbar(mesh, ax=ax).set_label("Probability density")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\hat{z}$")

    plt.show()


def plot_parametric_loss(distribution, n_inputs):
    """
    Plot simulated loss vs variance for a given parametric distribution.

    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'lognormal', 'lomax', 'gamma').
    n_inputs : int
        Number of inputs used in the simulation.
    """
    df = pd.read_parquet(sim_dir + f"{distribution}_sim_{n_inputs}.parquet")

    mean_vals = np.sort(df["mean"].unique())
    var_vals = df["var"].unique()

    var_min, var_max = np.min(var_vals), np.max(var_vals)
    var_pred = np.logspace(np.log10(var_min) - 0.3, np.log10(var_max) + 0.3, n_pred)

    # Color range anchored to bin-median range from FlyWire data (matches simulated_loss.py)
    cmap = plt.get_cmap("dark_cool")
    norm = mean_bin_median_norm()

    fig, ax = plt.subplots(figsize=(width_lg, height_lg))

    for mean in mean_vals:
        mask = df["mean"] == mean
        color = cmap(norm(mean))

        ax.plot(
            var_pred,
            general_loss(mean, var_pred),
            c=color,
            lw=2,
            label=f"Mean$={mean}$",
            zorder=0,
        )
        ax.scatter(
            df[mask]["var"],
            df[mask]["sim_loss"],
            c="white",
            edgecolors=color,
            s=20,
            rasterized=True,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([1e-2, 0.2])
    ax.set_xlim([5e-1, 1e4])

    ax.spines[["right", "top"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_lg)
    plt.savefig(fig_dir + f"{distribution}_simulation_var.svg", dpi=600)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Mean")

    ax.set_xlabel("Variance")
    ax.set_ylabel("Simulated error probability")

    plt.show()


def plot_distribution_pdf(distribution, mean, var, color, fname, x_range, y_range=None):
    """
    Plot the PDF of the weight distribution used in simulations.

    For Dirac distributions, draws a vertical line with a scatter point (atom).
    For continuous distributions, draws the PDF curve with a filled region beneath.

    Parameters
    ----------
    distribution : str
        Distribution type: 'dirac', 'poisson', 'pareto', 'gamma', or 'lognormal'.
    mean : float
        Mean of the distribution.
    var : float or None
        Variance (required for 'gamma' and 'lognormal').
    color : tuple
        RGB color for the plot.
    fname : str
        Filename suffix for saving.
    x_range : tuple
        Shared (x_min, x_max) range for the x-axis.
    y_range : tuple or None
        (y_min, y_max) for continuous distributions. If None, uses matplotlib default.
    """
    fig, ax = plt.subplots(figsize=(0.4 * width_sm, 0.4 * height_sm))

    x_min, x_max = x_range

    if distribution == "dirac":
        ax.plot([mean, mean], [0, 1], color=color, lw=2)
        ax.scatter(
            [mean], [1], color="white", edgecolors=color, s=30, clip_on=False, zorder=5
        )
        ax.set_ylim(bottom=0)

    elif distribution == "gamma":
        theta = var / mean
        shape = mean**2 / var
        x = np.linspace(x_min, x_max, 500)
        pdf = gamma_dist.pdf(x, a=shape, scale=theta)
        ax.plot(x, pdf, color=color, lw=2)
        ax.fill_between(x, pdf, alpha=0.2, color=color)
        if y_range is not None:
            ax.set_ylim(y_range)

    elif distribution == "poisson":
        from scipy.stats import poisson as poisson_dist

        x_int = np.arange(max(0, int(x_min)), int(x_max) + 1)
        pmf = poisson_dist.pmf(x_int, mean)
        ax.bar(x_int, pmf, color=color, alpha=0.7, width=0.8)

    elif distribution == "lognormal":
        if var is None:
            raise ValueError("Lognormal distribution requires variance parameter")
        sigma2 = np.log(1 + var / mean**2)
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - sigma2 / 2
        x = np.linspace(max(x_min, 1e-6), x_max, 500)
        pdf = lognorm_dist.pdf(x, s=sigma, scale=np.exp(mu))
        ax.plot(x, pdf, color=color, lw=2)
        ax.fill_between(x, pdf, alpha=0.2, color=color)
        if y_range is not None:
            ax.set_ylim(y_range)

    elif distribution == "pareto":
        if mean > 1:
            alpha_param = mean / (mean - 1)
            x = np.linspace(max(1, x_min), x_max, 500)
            pdf = alpha_param / (x ** (alpha_param + 1))
            ax.plot(x, pdf, color=color, lw=2)
            ax.fill_between(x, pdf, alpha=0.2, color=color)

    ax.set_xlim(x_range)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines[["top", "right"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"pdf_{fname}.svg", dpi=600)
    plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: DECISION BOUNDARY CARTOON
# ------------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: DECISION BOUNDARY CARTOON")
print("=" * 60)

lim = 1.0

fig, ax = plt.subplots(figsize=(width_md, height_md))

y1, y2 = 0.7, 0.2

ax.plot([-1.0, 1.0], [0.0, 0.0], c="k", lw=1)
ax.plot([0.0, 0.0], [-1.0, 1.0], c="k", lw=1)

ax.plot([-1.0, 1.0], [y1, -y1], c=con_colors[0], lw=2, label="Unperturbed")
ax.plot([-1.0, 1.0], [y2, -y2], c=con_colors[1], lw=2, label="Perturbed")

ax.annotate(
    "",
    xytext=(0, 0),
    xy=(y1 * 0.6 / (1.0 + y1**2) ** 0.5, 0.6 / (1.0 + y1**2) ** 0.5),
    arrowprops=dict(arrowstyle="->", color=con_colors[0]),
)
ax.annotate(
    "",
    xytext=(0, 0),
    xy=(y2 * 0.6 / (1.0 + y2**2) ** 0.5, 0.6 / (1.0 + y2**2) ** 0.5),
    arrowprops=dict(arrowstyle="->", color=con_colors[1]),
)

ax.fill_between([-1.0, 1.0], [y1, -y1], [y2, -y2], color=con_colors[1], alpha=0.3)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + "classification_plane.svg", dpi=600)

ax.text(0.25, 0.75, "Fire", ha="left", va="bottom", transform=ax.transAxes)
ax.text(0.35, 0.35, "Not fire", ha="right", va="top", transform=ax.transAxes)

ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")

plt.show()
print("  Generated decision boundary cartoon")

# ------------------------------------------------------------------------------
# SECTION 2: ELLIPSE CARTOON WITH PRINCIPAL AXES (z, zhat plane)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: ELLIPSE CARTOON WITH PRINCIPAL AXES")
print("=" * 60)

# Analytical sigma values for the intermediate param set (index 1), normalized by sqrt(n_inputs)
_, mean_mid, var_mid = parse_param_set(param_sets[1])
sigma_x_mid = np.sqrt(var_mid + mean_mid**2)
sigma_y_mid = eps * np.sqrt(mean_mid)

R = 2.0        # level-set radius
lim = 1.2 * R * max(sigma_x_mid, sigma_y_mid)

fig, ax = plt.subplots(figsize=(width_md, height_md))

# Error shading: wedge between the y-axis and y = -x
x_r = np.array([0.0, lim])
ax.fill_between(x_r, -lim, -x_r, color=con_colors[1], alpha=0.3)
x_l = np.array([-lim, 0.0])
ax.fill_between(x_l, -x_l, lim, color=con_colors[1], alpha=0.3)

# Coordinate axes
ax.plot([-lim, lim], [0.0, 0.0], c="k", lw=1)
ax.plot([0.0, 0.0], [-lim, lim], c="k", lw=1)

# y = -x boundary line
ax.plot([-lim, lim], [lim, -lim], c="k", alpha=0.5, lw=1)

# Ellipse: x = R*sigma_x*cos(t), y = R*sigma_y*sin(t)
t = np.linspace(0, 2 * np.pi, 500)
x_ell = R * sigma_x_mid * np.cos(t)
y_ell = R * sigma_y_mid * np.sin(t)
ax.fill(x_ell, y_ell, color=sim_colors[1], alpha=0.3)
ax.plot(x_ell, y_ell, color=sim_colors[1], lw=2)

# Principal axis along z: arrow from origin to (R*sigma_x, 0)
ax.annotate("", xy=(R * sigma_x_mid, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color='k', lw=2, mutation_scale=12,
                            shrinkA=0, shrinkB=0))

# Principal axis along z_hat: arrow from origin to (0, R*sigma_y)
ax.annotate("", xy=(0, R * sigma_y_mid), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color='k', lw=2, mutation_scale=12,
                            shrinkA=0, shrinkB=0))

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
_t = _outer_tick(lim)
ax.set_xticks([-_t, 0, _t])
ax.set_yticks([-_t, 0, _t])

plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + "2d_local_field_distribution.svg", dpi=600)

ax.text(0.77, 0.23, "Error", ha="center", va="center", transform=ax.transAxes)
ax.text(0.23, 0.77, "Error", ha="center", va="center", transform=ax.transAxes)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$\hat{z}$")

plt.show()
print("  Generated ellipse cartoon with principal axes")

# ------------------------------------------------------------------------------
# SECTION 3: LOAD / RUN z/ztilde SIMULATIONS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: SIMULATION")
print("=" * 60)

run_zztilde_simulation()
df_all = pd.read_parquet(sim_file)
print(f"Loaded {len(df_all)} rows from {sim_file}")

# ------------------------------------------------------------------------------
# SHARED AXIS LIMITS FOR SECTIONS 4 & 5
# Normalized sigma values (divided by sqrt(n_inputs)):
#   sigma_x^2 = var + mean^2 = E[w^2],  sigma_y^2 = eps^2 * mean = eps^2 * E[w]
# Use ±2 sigma (taking the max across all param sets) so all six panels share the same range.
# ------------------------------------------------------------------------------
sigma_xs, sigma_ys = [], []
for param_set in param_sets:
    distribution, mean, var = parse_param_set(param_set)
    if distribution in ("gamma", "lognormal") and var is not None:
        df_tmp = df_all[
            (df_all["distribution"] == distribution)
            & (df_all["mean"] == mean)
            & (df_all["variance"] == var)
            & (df_all["eps"] == eps)
        ]
    else:
        df_tmp = df_all[
            (df_all["distribution"] == distribution)
            & (df_all["mean"] == mean)
            & (df_all["eps"] == eps)
        ]
    computed_var = df_tmp["variance"].iloc[0]
    sigma_xs.append(np.sqrt(computed_var + mean**2))
    sigma_ys.append(eps * np.sqrt(mean))

hist_xlim = 2.0 * max(sigma_xs)
hist_ylim = 2.0 * max(sigma_ys)

# ------------------------------------------------------------------------------
# SECTION 4: SIMULATED 2D HISTOGRAMS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: SIMULATED HISTOGRAMS")
print("=" * 60)

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    print(f"Processing histogram for distribution={distribution}, mean={mean}...")

    if distribution in ("gamma", "lognormal") and var is not None:
        df_subset = df_all[
            (df_all["distribution"] == distribution)
            & (df_all["mean"] == mean)
            & (df_all["variance"] == var)
            & (df_all["eps"] == eps)
        ]
    else:
        df_subset = df_all[
            (df_all["distribution"] == distribution)
            & (df_all["mean"] == mean)
            & (df_all["eps"] == eps)
        ]

    z_vals = df_subset["z"].values / np.sqrt(n_inputs)
    zhat_vals = df_subset["zhat"].values / np.sqrt(n_inputs)

    plot_local_field_hist(
        z_vals, zhat_vals, color=sim_colors[i], fname=f"{distribution}_{i}",
        last=(i == len(param_sets) - 1),
        xlim=hist_xlim, ylim=hist_ylim,
    )

    print("  Generated histogram")
    print(f"  Simulated std:   std(z)={np.std(z_vals):.4f},  std(zhat)={np.std(zhat_vals):.4f}")

# ------------------------------------------------------------------------------
# SECTION 5: ANALYTICAL GAUSSIAN HEATMAPS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: ANALYTICAL GAUSSIAN HEATMAPS")
print("=" * 60)

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    print(
        f"Generating Gaussian heatmap for distribution={distribution}, mean={mean}, "
        f"sigma_x={sigma_xs[i]:.1f}, sigma_y={sigma_ys[i]:.1f}..."
    )

    plot_gaussian_heatmap(
        sigma_xs[i], sigma_ys[i], sim_colors[i], fname=f"{distribution}_{i}",
        last=(i == len(param_sets) - 1),
        xlim=hist_xlim, ylim=hist_ylim,
    )

    print("  Generated heatmap")
    print(f"  Analytical std:  std(z)={sigma_xs[i]:.4f},  std(zhat)={sigma_ys[i]:.4f}")

# ------------------------------------------------------------------------------
# SECTION 6: LOSS VS VARIANCE
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: LOSS VS VARIANCE")
print("=" * 60)

plot_parametric_loss("gamma", 1000)
print("  Generated loss vs variance plot")

# ------------------------------------------------------------------------------
# SECTION 7: DISTRIBUTION PDFs
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: DISTRIBUTION PDFs")
print("=" * 60)

x_max_vals = []
for param_set in param_sets:
    distribution, mean, var = parse_param_set(param_set)
    if distribution == "dirac":
        x_max_vals.append(mean * 2)
    elif distribution == "gamma":
        theta = var / mean
        shape = mean**2 / var
        x_max_vals.append(gamma_dist.ppf(0.7, a=shape, scale=theta))
    elif distribution == "pareto":
        from scipy.stats import pareto as pareto_dist

        alpha_param = mean / (mean - 1) if mean > 1 else 2
        x_max_vals.append(pareto_dist.ppf(0.7, alpha_param, scale=1))
    elif distribution == "lognormal":
        sigma2 = np.log(1 + var / mean**2)
        sigma = np.sqrt(sigma2)
        mu = np.log(mean) - sigma2 / 2
        x_max_vals.append(lognorm_dist.ppf(0.7, s=sigma, scale=np.exp(mu)))
    elif distribution == "poisson":
        from scipy.stats import poisson as poisson_dist

        x_max_vals.append(poisson_dist.ppf(0.7, mean))

x_range_pdf = (0, max(x_max_vals))
continuous_y_range = (0, pdf_y_max) if pdf_y_max is not None else None

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    print(f"Plotting PDF for distribution={distribution}, mean={mean}...")
    y_range = continuous_y_range if distribution in ("gamma", "lognormal") else None
    plot_distribution_pdf(
        distribution,
        mean,
        var,
        sim_colors[i],
        fname=f"{distribution}_{i}",
        x_range=x_range_pdf,
        y_range=y_range,
    )
    print("  Generated PDF plot")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
