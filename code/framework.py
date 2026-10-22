"""
    Generates all framework figures for the manuscript.
-------------------------------------------------------------------------------
created on:
    Sun 13 Apr 2026
-------------------------------------------------------------------------------
last change:
    Thu 22 Oct 2026
-------------------------------------------------------------------------------
notes:
    Generates the main framework figures:
      1. Decision boundary cartoon (classification_plane.svg)
      2. Ellipse cartoon with principal axes (2d_local_field_distribution.svg)
      3. Simulated 2D histograms (hist_*.svg)
      4. Analytical Gaussian heatmaps (gaussian_*.svg)
      5. Error vs variance plot (*_simulation_var.svg) and error vs robustness
         curve-collapse plot (*_simulation_robustness.svg)
      6. Sparse-firing error rate vs robustness, parameterized by firing
         probability p_f, with simulated FAFB-inspired synthetic-neuron
         overlay points (sparse_error_vs_robustness.svg)
      7. Weight distribution PDFs (pdf_*.svg)

    Outputs to figures/framework/raw/.
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
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gamma as gamma_dist
from scipy.stats import multivariate_normal
from scipy.special import erfinv, erf
from params import (
    rng_seed,
    zztilde_param_sets,
    zztilde_n_inputs,
    zztilde_sigma,
    zztilde_n_draws,
    zztilde_n_perturb,
    shuffle_k_min,
    sparse_p_fire_vals,
)
from simulations import run_z_zhat_simulation
from figure_formatting import apply_style, outer_tick, log_format

# ------------------------------------------------------------------------------
# MATPLOTLIB CONFIGURATION
# ------------------------------------------------------------------------------
apply_style()

# ------------------------------------------------------------------------------
# SHARED CONFIGURATION
# ------------------------------------------------------------------------------
palette = np.array([[0, 77, 128], [181, 23, 0]]) / 255  # blue, red

# Directory paths
sim_dir = "../simulation_results/"
processed_dir = "../processed_data/"
fig_dir = "../figures/framework/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Single consolidated simulation file for z/zhat
sim_file = sim_dir + "z_zhat_simulations.parquet"

# Reference size (page width in mm) for scaling figure dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel dimensions in inches (scaled from mm reference)
width_xs = height_xs = 0.12 * pg_width / mm_to_in
width_sm = height_sm = 0.18 * pg_width / mm_to_in
width_md = height_md = 0.32 * pg_width / mm_to_in
width_lg = height_lg = 0.35 * pg_width / mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins_xs = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
fig_margins_sm = dict(left=0.2, right=0.98, bottom=0.2, top=0.98)
fig_margins_md = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_lg = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)

# Dark-cool colormap for parametric error plots
dark_cool = mcolors.LinearSegmentedColormap.from_list(
    "dark_cool", ["#0b3c49", "#1f5c8b", "#4a4aa8", "#7a3fa0", "#9b2f8a"]
)
if "dark_cool" not in plt.colormaps:
    plt.colormaps.register(dark_cool)

# z/zhat simulation parameters (defined in params.py)
param_sets = zztilde_param_sets
n_inputs = zztilde_n_inputs
sigma = zztilde_sigma
n_draws = zztilde_n_draws
n_perturb = zztilde_n_perturb

# Histogram parameters
n_bins = 20

# Colorbar maximum floor for the third z/zhat panel (index 2), applied only if
# the automatically-determined maximum density is lower than this value
third_panel_min_vmax = 0.1

# Distribution PDF plot parameters — set to a float to fix the y-axis for continuous PDFs
pdf_y_max = 1.0

# Parametric error parameters
n_pred = 200

# Colors for each parameter set (equally spaced from viridis, capped at 80%)
n_colors = len(param_sets)
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


def darken_cmap(name, max_val=0.85):
    """Truncate a colormap before its brightest (yellow) end, so it reads
    darker/higher-contrast on a white background."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.0, max_val, 256))
    return LinearSegmentedColormap.from_list(f"{name}_dark", colors)


plasma_dark_r = darken_cmap("plasma").reversed()


def mean_bin_median_norm(n_mean_bins=4):
    """
    Compute LogNorm anchored to log-midpoint mean values from FAFB neuron data.
    Matches the colorbar range used in simulated_error.py.
    """
    neuron_df = pd.read_parquet(processed_dir + "fafb_neurons.parquet")
    neuron_df = neuron_df[neuron_df["in_deg"] >= shuffle_k_min]
    neuron_df["mean"] = neuron_df["in_strength"] / neuron_df["in_deg"]
    neuron_df["var"] = (neuron_df["sum_w2"] / neuron_df["in_deg"]) - neuron_df[
        "mean"
    ] ** 2
    df_nonneg = neuron_df[neuron_df["var"] > 1e-5].copy()

    mean_bins = np.logspace(
        np.log10(df_nonneg["mean"].min()),
        np.log10(df_nonneg["mean"].max()),
        n_mean_bins + 1,
    )
    mean_mids = np.sqrt(mean_bins[:-1] * mean_bins[1:])
    print("Mean bin log-midpoints used for colors:")
    for i, m in enumerate(mean_mids):
        print(f"  Bin {i}: {m:.4f}")
    return mcolors.LogNorm(vmin=1, vmax=mean_mids.max())


def robustness_from_moments(mean, var):
    """Robustness r = sqrt(mean + var/mean), matching simulations.compute_robustness."""
    return np.sqrt(mean + var / mean)


def error_from_robustness(rob):
    """Predicted error as a function of robustness alone (curve-collapse form)."""
    return (1 / np.pi) * np.arccos((1.0 + 1.0 / rob**2) ** (-1 / 2))


def general_error(mean, var):
    """Predicted error from mean and variance."""
    return error_from_robustness(robustness_from_moments(mean, var))


def _phi2(h, k, rho):
    """Bivariate standard normal CDF with correlation rho, evaluated at (h, k)."""
    mean = (0.0, 0.0)
    cov = ((1.0, rho), (rho, 1.0))
    return multivariate_normal(mean, cov).cdf([h, k])


def _norm_loc(p0):
    """Standard normal quantile corresponding to firing probability p0."""
    return -np.sqrt(2) * erfinv(1.0 - 2 * p0)


def sparse_error_rate(p0, rho):
    """
    Error rate for a sparsely-firing neuron (firing probability p0) as a
    function of the z/ztilde correlation rho, matching
    supplement_code/sparse_inputs_outputs.py.
    """
    alpha = _norm_loc(p0)
    l = 0.5 * (1.0 + erf(alpha / np.sqrt(2)))
    l += 0.5 * (1.0 + erf(rho * alpha / np.sqrt(2)))
    l -= 2 * _phi2(alpha, rho * alpha, rho)
    return l


# ==============================================================================
# SIMULATION PARAMETER HELPERS (z/zhat)
# ==============================================================================
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


def select_simulation(df_all, param_set):
    """Rows of the z/zhat simulation matching a parameter set (and global sigma)."""
    distribution, mean, var = parse_param_set(param_set)
    mask = (
        (df_all["distribution"] == distribution)
        & (df_all["mean"] == mean)
        & (df_all["sigma"] == sigma)
    )
    if distribution == "gamma" and var is not None:
        mask &= df_all["variance"] == var
    return df_all[mask]


# ==============================================================================
# PLOT FUNCTIONS
# ==============================================================================
def plot_local_field_hist(
    z, zhat, color, fname, xlim=300.0, ylim=600.0, min_vmax=None, show_yticks=False
):
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
    xlim, ylim : float
        Symmetric axis limits; shared across all panels.
    min_vmax : float, optional
        Floor for the colorbar maximum. If the automatically-determined
        maximum density is lower than this, the colorbar max is raised to it.
    show_yticks : bool
        If True, show y ticks (-outer, 0, outer); otherwise hide them.
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

    if min_vmax is not None:
        hist_img.set_clim(vmax=max(hist_img.get_array().max(), min_vmax))

    for loc in ["left", "right", "top", "bottom"]:
        ax_scatter.spines[loc].set_visible(True)
    ax_scatter.set_frame_on(True)

    # Error shading: region between y-axis and y=-x (equivalent to Q2/Q4 in z-ztilde space)
    x_r = np.array([0.0, xlim])
    ax_scatter.fill_between(x_r, -ylim, -x_r, color=palette[1], alpha=0.3)
    x_l = np.array([-xlim, 0.0])
    ax_scatter.fill_between(x_l, -x_l, ylim, color=palette[1], alpha=0.3)

    ax_scatter.plot([0.0, 0.0], [-ylim, ylim], c="k", alpha=0.5, lw=1)
    ax_scatter.plot([-xlim, xlim], [0.0, 0.0], c="k", alpha=0.5, lw=1)
    _diag = max(xlim, ylim)
    ax_scatter.plot([-_diag, _diag], [_diag, -_diag], c="k", alpha=0.5, lw=1)

    ax_scatter.set_xlim(-xlim, xlim)
    ax_scatter.set_ylim(-ylim, ylim)
    if show_yticks:
        _yt = outer_tick(ylim)
        ax_scatter.set_yticks([-_yt, 0, _yt])
    else:
        ax_scatter.set_yticks([])
    _xt = outer_tick(xlim)
    ax_scatter.set_xticks([-_xt, 0, _xt])

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"hist_{fname}.svg", dpi=600)

    fig.colorbar(hist_img, ax=ax_scatter).set_label("Probability density")
    ax_scatter.set_xlabel(r"Local field $z/\sqrt{n}$")
    ax_scatter.set_ylabel(r"Local field perturbation $\hat{z}/\sqrt{n}$")

    plt.show()


def plot_gaussian_heatmap(
    sigma_x, sigma_y, color, fname, xlim=300.0, ylim=600.0, n_grid=200, min_vmax=None
):
    """
    Plot an analytical 2D Gaussian heatmap for (z, zhat) with zero covariance.

    The density is N(0, diag(sigma_x^2, sigma_y^2)), where:
        sigma_x^2 = sum_i w_i^2  (analytically: n_inputs * E[w^2])
        sigma_y^2 = sigma^2 * sum_i w_i  (analytically: sigma^2 * n_inputs * E[w])

    Parameters
    ----------
    sigma_x, sigma_y : float
        Standard deviations along the z and zhat axes respectively.
    color : np.ndarray
        RGB color array for the colormap.
    fname : str
        Filename suffix for saving.
    xlim, ylim : float
        Symmetric axis limits; shared across all panels.
    n_grid : int
        Number of grid points in each dimension.
    min_vmax : float, optional
        Floor for the colorbar maximum. If the automatically-determined
        maximum density is lower than this, the colorbar max is raised to it.
    """
    x = np.linspace(-xlim, xlim, n_grid)
    y = np.linspace(-ylim, ylim, n_grid)
    X, Y = np.meshgrid(x, y)

    Z = np.exp(-(X**2) / (2 * sigma_x**2) - Y**2 / (2 * sigma_y**2)) / (
        2 * np.pi * sigma_x * sigma_y
    )

    fig, ax = plt.subplots(figsize=(width_sm, height_sm))

    vmax = max(Z.max(), min_vmax) if min_vmax is not None else None
    mesh = ax.pcolormesh(
        X,
        Y,
        Z,
        cmap=fade_to_color_cmap(color, alpha_min=0.0),
        shading="auto",
        rasterized=True,
        vmax=vmax,
    )

    for loc in ["left", "right", "top", "bottom"]:
        ax.spines[loc].set_visible(True)
    ax.set_frame_on(True)

    # Error shading: region between y-axis and y=-x (equivalent to Q2/Q4 in z-ztilde space)
    x_r = np.array([0.0, xlim])
    ax.fill_between(x_r, -ylim, -x_r, color=palette[1], alpha=0.3)
    x_l = np.array([-xlim, 0.0])
    ax.fill_between(x_l, -x_l, ylim, color=palette[1], alpha=0.3)

    ax.plot([0.0, 0.0], [-ylim, ylim], c="k", alpha=0.5, lw=1)
    ax.plot([-xlim, xlim], [0.0, 0.0], c="k", alpha=0.5, lw=1)
    _diag = max(xlim, ylim)
    ax.plot([-_diag, _diag], [_diag, -_diag], c="k", alpha=0.5, lw=1)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    ax.set_yticks([])
    _xt = outer_tick(xlim)
    ax.set_xticks([-_xt, 0, _xt])

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"gaussian_{fname}.svg", dpi=600)

    fig.colorbar(mesh, ax=ax).set_label("Probability density")
    ax.set_xlabel(r"Local field $z/\sqrt{n}$")
    ax.set_ylabel(r"Local field perturbation $\hat{z}/\sqrt{n}$")

    plt.show()


def plot_parametric_error(distribution):
    """
    Plot simulated error vs variance for a given parametric distribution.

    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'gamma').
    """
    df = pd.read_parquet(sim_dir + f"{distribution}_sim.parquet")

    mean_vals = np.sort(df["mean"].unique())
    var_vals = df["var"].unique()

    var_min, var_max = np.min(var_vals), np.max(var_vals)
    var_pred = np.logspace(np.log10(var_min) - 0.3, np.log10(var_max) + 0.3, n_pred)

    # Color range anchored to bin-median range from FAFB data (matches simulated_error.py)
    cmap = plt.get_cmap("dark_cool")
    norm = mean_bin_median_norm()

    fig, ax = plt.subplots(figsize=(width_md, height_md))

    for mean in mean_vals:
        mask = df["mean"] == mean
        color = cmap(norm(mean))

        ax.plot(
            var_pred,
            general_error(mean, var_pred),
            c=color,
            lw=2,
            label=f"Mean$={mean}$",
            zorder=0,
        )
        ax.scatter(
            df[mask]["var"],
            df[mask]["sim_error"],
            c="white",
            edgecolors=color,
            s=20,
            rasterized=True,
        )

    log_format(ax)
    ax.set_ylim([1e-2, 0.2])
    ax.set_xlim([5e-1, 1e4])

    ax.spines[["right", "top"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_md)
    plt.savefig(fig_dir + f"{distribution}_simulation_var.svg", dpi=600)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Average weight")

    ax.set_xlabel("Variance in weights")
    ax.set_ylabel("Error rate")

    plt.show()


def plot_parametric_error_vs_robustness(distribution):
    """
    Plot simulated error vs robustness for a given parametric distribution.

    Same scatter points as plot_parametric_error (colored by mean bin), but
    with mean and variance collapsed onto a single robustness axis — the
    curve-collapse view of the error vs variance panel.

    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'gamma').
    """
    df = pd.read_parquet(sim_dir + f"{distribution}_sim.parquet")

    mean_vals = np.sort(df["mean"].unique())
    robustness = robustness_from_moments(df["mean"], df["var"])

    rob_min, rob_max = robustness.min(), robustness.max()
    rob_pad = 0.05 * (rob_max - rob_min)
    rob_pred = np.linspace(max(rob_min - rob_pad, 0.0), rob_max + rob_pad, n_pred)

    # Color range anchored to bin-median range from FAFB data (matches simulated_error.py)
    cmap = plt.get_cmap("dark_cool")
    norm = mean_bin_median_norm()

    fig, ax = plt.subplots(figsize=(width_md, height_md))

    # Single collapse curve: mean/variance dependence collapses onto robustness alone
    ax.plot(rob_pred, error_from_robustness(rob_pred), c="k", lw=2, zorder=0)

    for mean in mean_vals:
        mask = df["mean"] == mean
        color = cmap(norm(mean))

        ax.scatter(
            robustness[mask],
            df[mask]["sim_error"],
            c="white",
            edgecolors=color,
            s=20,
            zorder=2,
            rasterized=True,
        )

    ax.set_yscale("log")
    # Explicit, data-tight limits so the panel fills the frame the same way
    # plot_parametric_error's fixed [5e-1, 1e4] xlim does (rather than relying
    # on matplotlib's auto-padding), keeping the two panels' proportions matched.
    ax.set_xlim([max(rob_min - rob_pad, 0.0), 30.0])
    ax.set_ylim([1e-2, 0.2])

    ax.spines[["right", "top"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_md)
    plt.savefig(fig_dir + f"{distribution}_simulation_robustness.svg", dpi=600)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("Average weight")

    ax.set_xlabel("Robustness")
    ax.set_ylabel("Error rate")

    plt.show()


def plot_sparse_error_vs_robustness(
    r_min=0.3,
    r_max=30.0,
    p_vals=tuple(sparse_p_fire_vals),
    n_r=500,
    distribution="gamma",
):
    """
    Plot error rate vs robustness for sparsely-firing neurons, parameterized
    by firing probability p_f (curve-collapse form assuming sigma=1).

    Recreates the SI figure from supplement_code/sparse_inputs_outputs.py
    (SECTION 3: error rate vs robustness) using the framework's medium
    panel sizing/styling. Curve colors are evenly spaced along the colormap
    regardless of the (log-spaced) firing probability values themselves.

    If simulation_results/{distribution}_sim_sparse.parquet exists
    (see simulations.run_parametric_simulations_sparse), overlays FAFB-inspired
    synthetic-neuron Monte Carlo points for each p_fire, color-matched to its
    analytical curve.
    """
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    rho_of_r = (1.0 + (1.0 / r_vals) ** 2) ** (-0.5)

    p_vals = np.asarray(p_vals)
    cmap = plasma_dark_r
    colors = cmap(np.linspace(0.0, 1.0, len(p_vals)))

    fig, ax = plt.subplots(figsize=(width_md, height_md))

    for p0, color in zip(p_vals, colors):
        error_r = np.array([sparse_error_rate(p0, rho) for rho in rho_of_r])
        ax.plot(r_vals, error_r, color=color, lw=2, label=f"{p0:g}")

    sparse_sim_file = sim_dir + f"{distribution}_sim_sparse.parquet"
    if os.path.exists(sparse_sim_file):
        df_sparse = pd.read_parquet(sparse_sim_file)
        robustness = robustness_from_moments(df_sparse["mean"], df_sparse["var"])

        for p0, color in zip(p_vals, colors):
            mask = np.isclose(df_sparse["p_fire"], p0)
            ax.scatter(
                robustness[mask],
                df_sparse.loc[mask, "sim_error"],
                c="white",
                edgecolors=color,
                s=20,
                zorder=2,
                rasterized=True,
            )

    ax.set_yscale("log")
    ax.set_xlim([0.0, r_max])
    ax.set_ylim(bottom=1e-3)
    ax.spines[["right", "top"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_md)
    plt.savefig(fig_dir + "sparse_error_vs_robustness.svg", dpi=600)

    ax.set_xlabel("Robustness")
    ax.set_ylabel("Error rate")
    ax.legend(title="Firing probability", frameon=False)

    plt.show()


def plot_distribution_pdf(distribution, mean, var, color, fname, x_range, y_range=None):
    """
    Plot the PDF of the weight distribution used in simulations.

    For Dirac distributions, draws a vertical line with a scatter point (atom).
    For continuous distributions, draws the PDF curve with a filled region beneath.

    Parameters
    ----------
    distribution : str
        Distribution type: 'dirac' or 'gamma'.
    mean : float
        Mean of the distribution.
    var : float or None
        Variance (required for 'gamma').
    color : tuple
        RGB color for the plot.
    fname : str
        Filename suffix for saving.
    x_range : tuple
        Shared (x_min, x_max) range for the x-axis.
    y_range : tuple or None
        (y_min, y_max) for continuous distributions. If None, uses matplotlib default.
    """
    fig, ax = plt.subplots(figsize=(width_xs, height_xs))

    x_min, x_max = x_range

    if distribution == "dirac":
        ax.plot([mean, mean], [0, 1], color=color, lw=2)
        ax.scatter(
            [mean], [1], color=color, edgecolors=color, s=30, clip_on=False, zorder=5
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

    ax.set_xlim(x_range)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines[["top", "right"]].set_visible(False)

    plt.subplots_adjust(**fig_margins_xs)
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

fig, ax = plt.subplots(figsize=(width_lg, height_lg))

y1, y2 = 0.7, 0.2

ax.plot([-1.0, 1.0], [0.0, 0.0], c="k", lw=1)
ax.plot([0.0, 0.0], [-1.0, 1.0], c="k", lw=1)

ax.plot([-1.0, 1.0], [y1, -y1], c=palette[0], lw=2, label="Unperturbed")
ax.plot([-1.0, 1.0], [y2, -y2], c=palette[1], lw=2, label="Perturbed")

ax.annotate(
    "",
    xytext=(0, 0),
    xy=(y1 * 0.6 / (1.0 + y1**2) ** 0.5, 0.6 / (1.0 + y1**2) ** 0.5),
    arrowprops=dict(arrowstyle="->", color=palette[0]),
)
ax.annotate(
    "",
    xytext=(0, 0),
    xy=(y2 * 0.6 / (1.0 + y2**2) ** 0.5, 0.6 / (1.0 + y2**2) ** 0.5),
    arrowprops=dict(arrowstyle="->", color=palette[1]),
)

ax.fill_between([-1.0, 1.0], [y1, -y1], [y2, -y2], color=palette[1], alpha=0.3)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

for loc in ["left", "right", "top", "bottom"]:
    ax.spines[loc].set_visible(True)
ax.set_frame_on(True)

plt.subplots_adjust(**fig_margins_lg)
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
sigma_y_mid = sigma * np.sqrt(mean_mid)

R = 2.0  # level-set radius
lim = 1.1 * R * max(sigma_x_mid, sigma_y_mid)

fig, ax = plt.subplots(figsize=(width_lg, height_lg))

# Error shading: wedge between the y-axis and y = -x
x_r = np.array([0.0, lim])
ax.fill_between(x_r, -lim, -x_r, color=palette[1], alpha=0.3)
x_l = np.array([-lim, 0.0])
ax.fill_between(x_l, -x_l, lim, color=palette[1], alpha=0.3)

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
ax.annotate(
    "",
    xy=(R * sigma_x_mid, 0),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>", color="k", lw=2, mutation_scale=12, shrinkA=0, shrinkB=0
    ),
)

# Principal axis along z_hat: arrow from origin to (0, R*sigma_y)
ax.annotate(
    "",
    xy=(0, R * sigma_y_mid),
    xytext=(0, 0),
    arrowprops=dict(
        arrowstyle="-|>", color="k", lw=2, mutation_scale=12, shrinkA=0, shrinkB=0
    ),
)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
_t = outer_tick(lim)
ax.set_xticks([-_t, 0, _t])
ax.set_yticks([-_t, 0, _t])

for loc in ["left", "right", "top", "bottom"]:
    ax.spines[loc].set_visible(True)
ax.set_frame_on(True)

plt.subplots_adjust(**fig_margins_lg)
plt.savefig(fig_dir + "2d_local_field_distribution.svg", dpi=600)

ax.text(0.77, 0.23, "Error\n(silent)", ha="center", va="center", transform=ax.transAxes)
ax.text(0.23, 0.77, "Error\n(fire)", ha="center", va="center", transform=ax.transAxes)

ax.set_xlabel(r"Local field $z/\sqrt{n}$")
ax.set_ylabel(r"Local field perturbation $\hat{z}/\sqrt{n}$")

plt.show()
print("  Generated ellipse cartoon with principal axes")

# ------------------------------------------------------------------------------
# SECTION 3: LOAD / RUN z/zhat SIMULATIONS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: SIMULATION")
print("=" * 60)

run_z_zhat_simulation()
df_all = pd.read_parquet(sim_file)
print(f"Loaded {len(df_all)} rows from {sim_file}")

# ------------------------------------------------------------------------------
# SHARED AXIS LIMITS FOR SECTIONS 4 & 5
# Normalized sigma values (divided by sqrt(n_inputs)):
#   sigma_x^2 = var + mean^2 = E[w^2],  sigma_y^2 = sigma^2 * mean = sigma^2 * E[w]
# Use ±2 sigma (taking the max across all param sets) so all six panels share the same range.
# ------------------------------------------------------------------------------
sigma_xs, sigma_ys = [], []
for param_set in param_sets:
    _, mean, _ = parse_param_set(param_set)
    df_tmp = select_simulation(df_all, param_set)
    computed_var = df_tmp["variance"].iloc[0]
    sigma_xs.append(np.sqrt(computed_var + mean**2))
    sigma_ys.append(sigma * np.sqrt(mean))

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

    df_subset = select_simulation(df_all, param_set)

    z_vals = df_subset["z"].values / np.sqrt(n_inputs)
    zhat_vals = df_subset["zhat"].values / np.sqrt(n_inputs)

    plot_local_field_hist(
        z_vals,
        zhat_vals,
        color=sim_colors[i],
        fname=f"{distribution}_{i}",
        xlim=hist_xlim,
        ylim=hist_ylim,
        min_vmax=third_panel_min_vmax if i == 2 else None,
        show_yticks=(distribution == "dirac"),  # zero-variance panel only
    )

    print("  Generated histogram")
    print(
        f"  Simulated std:   std(z)={np.std(z_vals):.4f},  std(zhat)={np.std(zhat_vals):.4f}"
    )
    error_rate = np.mean(np.sign(z_vals) != np.sign(z_vals + zhat_vals))
    print(f"  Error rate:      {error_rate:.4f}")

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
        sigma_xs[i],
        sigma_ys[i],
        sim_colors[i],
        fname=f"{distribution}_{i}",
        xlim=hist_xlim,
        ylim=hist_ylim,
        min_vmax=third_panel_min_vmax if i == 2 else None,
    )

    print("  Generated heatmap")
    print(f"  Analytical std:  std(z)={sigma_xs[i]:.4f},  std(zhat)={sigma_ys[i]:.4f}")

# ------------------------------------------------------------------------------
# SECTION 6: ERROR VS VARIANCE / ERROR VS ROBUSTNESS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: ERROR VS VARIANCE / ERROR VS ROBUSTNESS")
print("=" * 60)

plot_parametric_error("gamma")
print("  Generated error vs variance plot")

plot_parametric_error_vs_robustness("gamma")
print("  Generated error vs robustness plot (curve collapse)")

# ------------------------------------------------------------------------------
# SECTION 6B: SPARSE FIRING ERROR VS ROBUSTNESS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6B: SPARSE FIRING ERROR VS ROBUSTNESS")
print("=" * 60)

plot_sparse_error_vs_robustness()
print("  Generated sparse-firing error vs robustness plot")

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

x_range_pdf = (0, max(x_max_vals))
continuous_y_range = (0, pdf_y_max) if pdf_y_max is not None else None

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    print(f"Plotting PDF for distribution={distribution}, mean={mean}...")
    y_range = continuous_y_range if distribution == "gamma" else None
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
