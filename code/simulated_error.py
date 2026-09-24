"""
    This script analyzes the loss simulated from FlyWire data .
-------------------------------------------------------------------------------
created on:
    Fri 21 Nov 2024
-------------------------------------------------------------------------------
last change:
    Wed 23 Sep 2026
-------------------------------------------------------------------------------
notes:
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
import logging
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import multivariate_normal
from scipy.special import erfinv, erf
from figure_formatting import apply_style, log_format
from params import error_p_fire_vals, error_sigma_vals, error_sigma

apply_style()

# Reference size (page width in mm) for scaling figure dimensions
pg_width = 165  # mm
mm_to_in = 25.4

# Panel size presets (scaled from mm reference): xs, sm, md, lg.
# xs: per-mean robustness-vs-variance plots
# sm: neuron-statistic distribution plots
# md: error-vs-robustness plot (matches framework.py's "medium" panel exactly)
# lg: analytic-vs-simulated error rate plot
width_xs = height_xs = 0.14 * pg_width / mm_to_in
width_sm = height_sm = 0.17 * pg_width / mm_to_in
width_md = height_md = 0.32 * pg_width / mm_to_in
width_lg = height_lg = 0.39 * pg_width / mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins_xs = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)
fig_margins_sm = dict(left=0.25, right=0.95, bottom=0.25, top=0.95)
fig_margins_md = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_lg = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)

alpha_min = 0.0

# Output directory
fig_dir = "../../figures/simulated_error/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Connectome list
connectomes = [
    "drosophila_central_brain",
    "drosophila_optic_medulla",
    "c_elegans",
    "platynereis_sensory_motor",
    "mouse_retina",
    "drosophila_whole_brain",
]

data_idx = 5
thresholded = False
scheme = "remove"

eps = 1.0

suffix = "_thresholded" if thresholded else ""

# Plotting colors
con_colors = (
    np.array(
        [
            [0, 77, 128],
            [181, 23, 0],
            [1, 113, 0],
            [242, 112, 0],
            [120, 0, 150],
            [255, 204, 0],
            [203, 41, 123],
            [0, 0, 0],
        ]
    )
    / 255
)
dark_cool = mcolors.LinearSegmentedColormap.from_list(
    "dark_cool", ["#0b3c49", "#1f5c8b", "#4a4aa8", "#7a3fa0", "#9b2f8a"]
)
if "dark_cool" not in plt.colormaps:
    plt.colormaps.register(dark_cool)


def darken_cmap(name, max_val=0.85):
    """Truncate a colormap before its brightest (yellow) end, so it reads
    darker/higher-contrast on a white background."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.0, max_val, 256))
    return LinearSegmentedColormap.from_list(f"{name}_dark", colors)


plasma_dark_r = darken_cmap("plasma").reversed()
viridis_dark = darken_cmap("viridis")


# ------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ------------------------------------------------------------------------------
def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)


def general_loss(mean, var):
    """Compute predicted loss from mean and variance."""
    rob = np.sqrt(mean + var / mean)
    return (1 / np.pi) * np.arccos((1.0 + (eps / rob) ** 2) ** (-1 / 2))


def _phi2(h, k, rho):
    """Bivariate standard normal CDF with correlation rho, evaluated at (h, k)."""
    mean = (0.0, 0.0)
    cov = ((1.0, rho), (rho, 1.0))
    return multivariate_normal(mean, cov).cdf([h, k])


def _norm_loc(p0):
    """Standard normal quantile corresponding to firing probability p0."""
    return -np.sqrt(2) * erfinv(1.0 - 2 * p0)


def sparse_error_rate(p0, rho):
    """Error rate for a sparsely-firing neuron (firing probability p0) as a
    function of the z/ztilde correlation rho (matches framework.py)."""
    alpha = _norm_loc(p0)
    l = 0.5 * (1.0 + erf(alpha / np.sqrt(2)))
    l += 0.5 * (1.0 + erf(rho * alpha / np.sqrt(2)))
    l -= 2 * _phi2(alpha, rho * alpha, rho)
    return l


data_dir = "../../data/"
processed_dir = "../processed_data/"
sim_dir = "../simulation_results/"

# Import neuron data
neuron_df = pd.read_parquet(processed_dir + "neuron_data.parquet")
error_df_full = pd.read_parquet(sim_dir + "error_data.parquet")
error_df = error_df_full[(error_df_full["p_fire"] == 0.5) & (error_df_full["sigma"] == 1.0)][
    ["root_id", "sim_error"]
]

# Append data
neuron_df = neuron_df.merge(error_df, on="root_id", how="outer")

# Compute relevant moments (uses the full neuron population, before the
# in-degree filter below)
neuron_df["mean"] = neuron_df["in_strength"] / neuron_df["in_deg"]
neuron_df["var"] = (neuron_df["sum_w2"] / neuron_df["in_deg"]) - neuron_df["mean"] ** 2

# ------------------------------------------------------------------------------
# DISTRIBUTIONS OF NEURON INPUT STATISTICS (LOG-BINNED HISTOGRAMS)
# ------------------------------------------------------------------------------
stat_color = con_colors[0]

n_hist_bins = 40


def plot_stat_hist(values, xlabel, fname, n_bins=n_hist_bins):
    """Plot a log-binned probability histogram of a positive-valued neuron statistic."""
    values = values.to_numpy()
    values = values[np.isfinite(values) & (values > 0)]

    bin_edges = np.logspace(np.log10(values.min()), np.log10(values.max()), n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    prob = counts / counts.sum()
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    nonzero = prob > 0

    fig, ax = plt.subplots(figsize=(width_sm, height_sm))
    ax.scatter(
        bin_centers[nonzero],
        prob[nonzero],
        c=stat_color,
        s=20,
        rasterized=True,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([bin_edges[0], bin_edges[-1]])

    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + fname, dpi=600)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")

    plt.show()


plot_stat_hist(neuron_df["in_deg"], "Number of inputs", "hist_in_degree.svg")
plot_stat_hist(neuron_df["mean"], "Average input weight", "hist_mean_weight.svg")
plot_stat_hist(neuron_df["var"], "Variance in input weight", "hist_variance.svg")

# Keep only high in-degree neurons
k_min = 10
neuron_df = neuron_df[neuron_df["in_deg"] >= k_min]

# ------------------------------------------------------------------------------
# SIMULATED LOSS BY NEURON STATISTICS
# ------------------------------------------------------------------------------
# Get range of statistics
nonneg = neuron_df["var"] > 1e-5
l_min, l_max = np.min(neuron_df["sim_error"]), np.max(neuron_df["sim_error"])
mu_min, mu_max = np.min(neuron_df["mean"]), np.max(neuron_df["mean"])
var_min, var_max = np.min(neuron_df[nonneg]["var"]), np.max(neuron_df[nonneg]["var"])
rob_min, rob_max = np.min(neuron_df["robustness"]), np.max(neuron_df["robustness"])

# Bins for histogram
n_bins = 100

# FIGURE: LOSS VS VARIANCE (BINNED SCATTER, COLORED BY MEAN)-----------------
# Filter to neurons with positive variance
df_nonneg = neuron_df[nonneg].copy()

# Binning parameters
n_mean_bins = 4
n_var_bins = 10
n_pred = 200

# Compute log-uniform mean bin boundaries
mean_bins = np.logspace(
    np.log10(df_nonneg["mean"].min()),
    np.log10(df_nonneg["mean"].max()),
    n_mean_bins + 1,
)

# Assign mean bin indices
df_nonneg["mean_bin"] = pd.cut(
    df_nonneg["mean"], bins=mean_bins, labels=False, include_lowest=True
)

# Assign variance bin indices using quantiles within each mean bin
df_nonneg["var_bin"] = -1
for i in range(n_mean_bins):
    if i < n_mean_bins - 1:
        mean_mask = (df_nonneg["mean"] >= mean_bins[i]) & (
            df_nonneg["mean"] < mean_bins[i + 1]
        )
    else:
        mean_mask = (df_nonneg["mean"] >= mean_bins[i]) & (
            df_nonneg["mean"] <= mean_bins[i + 1]
        )

    subset_var = df_nonneg.loc[mean_mask, "var"]
    if len(subset_var) == 0:
        continue

    var_q = np.percentile(subset_var, np.linspace(0, 100, n_var_bins + 1))
    var_bin_idx = pd.cut(
        df_nonneg.loc[mean_mask, "var"], bins=var_q, labels=False, include_lowest=True
    )
    df_nonneg.loc[mean_mask, "var_bin"] = var_bin_idx

# Get prediction values spanning the full variance range
var_pred = np.logspace(np.log10(var_min), np.log10(var_max), n_pred)

# Log midpoint (geometric mean of bin edges) per mean bin
mean_mids = np.sqrt(mean_bins[:-1] * mean_bins[1:])
print("Mean bin log-midpoints used for colors:")
for i, m in enumerate(mean_mids):
    print(f"  Bin {i}: {m:.4f}")

# Log-normalize with fixed vmin=1
mean_norm = mcolors.LogNorm(vmin=1, vmax=mean_mids.max())

# Set up figure
cmap = plt.get_cmap("dark_cool")
fig, ax = plt.subplots(figsize=(width_md, height_md))

for i in range(n_mean_bins):
    mean_mask = df_nonneg["mean_bin"] == i
    if mean_mask.sum() == 0:
        continue

    mean_mid = mean_mids[i]
    color = cmap(mean_norm(mean_mid))

    # Compute median loss and median variance for each variance bin
    grouped_loss = df_nonneg[mean_mask].groupby("var_bin")["sim_error"].median()
    grouped_var = df_nonneg[mean_mask].groupby("var_bin")["var"].median()

    # Plot prediction line
    ax.plot(var_pred, general_loss(mean_mid, var_pred), c=color, lw=2, zorder=0)

    # Plot scatter for bins with data (using median variance as x-position)
    valid_bins = grouped_loss.index.dropna().astype(int)
    ax.scatter(
        grouped_var[valid_bins],
        grouped_loss[valid_bins],
        c="white",
        edgecolors=color,
        s=20,
        rasterized=True,
    )

ax.set_ylim([1e-2, 0.2])
ax.set_xlim([5e-1, 1e4])

plt.subplots_adjust(**fig_margins_md)
log_format(ax)
plt.savefig(fig_dir + "loss_vs_variance_binned.svg", dpi=600)

# Print colorbar range to console
print(
    f"Colorbar range (avg incoming weight) — min: {mean_mids.min():.4f}, max: {mean_mids.max():.4f}"
)

# Add colorbar for interactive display (not in saved file)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mean_norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)

ax.set_xlabel("Variance")
ax.set_ylabel("Simulated error probability")

plt.show()

# ------------------------------------------------------------------------------
# SEPARATE LOSS PLOTS
# ------------------------------------------------------------------------------
for i in range(n_mean_bins):
    fig, ax = plt.subplots(figsize=(width_xs, height_xs))

    mean_mask = df_nonneg["mean_bin"] == i
    if mean_mask.sum() == 0:
        continue

    mean_mid = mean_mids[i]
    color = cmap(mean_norm(mean_mid))

    # Compute median, Q1, Q3 for loss and variance per variance bin
    grouped_loss = df_nonneg[mean_mask].groupby("var_bin")["sim_error"].median()
    grouped_loss_q1 = (
        df_nonneg[mean_mask].groupby("var_bin")["sim_error"].quantile(0.25)
    )
    grouped_loss_q3 = (
        df_nonneg[mean_mask].groupby("var_bin")["sim_error"].quantile(0.75)
    )
    grouped_var = df_nonneg[mean_mask].groupby("var_bin")["var"].median()
    grouped_var_q1 = df_nonneg[mean_mask].groupby("var_bin")["var"].quantile(0.25)
    grouped_var_q3 = df_nonneg[mean_mask].groupby("var_bin")["var"].quantile(0.75)

    # Plot prediction line
    ax.plot(var_pred, general_loss(mean_mid, var_pred), c=color, lw=2, zorder=0)

    # Plot scatter with IQR error bars for bins with data
    valid_bins = grouped_loss.index.dropna().astype(int)
    x_med = grouped_var[valid_bins].values
    y_med = grouped_loss[valid_bins].values

    ax.scatter(x_med, y_med, c="white", edgecolors=color, s=20, rasterized=True)

    ax.set_ylim([1e-2, 0.2])
    ax.set_xlim([5e-1, 1e4])

    plt.subplots_adjust(**fig_margins_xs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e1, 1e3])
    if i != 0:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    plt.savefig(fig_dir + f"loss_vs_variance_separate_{i}.svg", dpi=600)

    # Add colorbar for interactive display (not in saved file)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mean_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax)

    ax.set_xlabel("Variance")
    ax.set_ylabel("Simulated error probability")

    plt.show()


# ------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
# ------------------------------------------------------------------------------
# Figure parameters
log_axes = False
log_prob = False
n_bins = 30

# Compute predicted loss
neuron_df["pred_loss"] = (1 / np.pi) * np.arccos(
    (1.0 + (eps / neuron_df["robustness"]) ** 2) ** (-1 / 2)
)

# Average relative error
err_diff = np.abs(neuron_df["sim_error"] - neuron_df["pred_loss"])
avg_sim_err = neuron_df["sim_error"].mean()
print(f"Average simulated error: {avg_sim_err:.4f}")
print(f"Average error difference: {err_diff.mean():.4f}")

# Set up figure
fig, ax = plt.subplots(figsize=(width_lg, height_lg))

# Create histogram
if log_axes:
    # Get loss range
    min_loss, max_loss = 1e-2, 0.2

    xbins = np.logspace(np.log10(min_loss), np.log10(max_loss), n_bins)
    ybins = np.copy(xbins)
    ax.set_xscale("log")
    ax.set_yscale("log")
else:
    # Get loss range
    min_loss, max_loss = 0.0, 0.2

    xbins = np.linspace(min_loss, max_loss, n_bins)
    ybins = np.linspace(min_loss, max_loss, n_bins)

counts, xedges, yedges = np.histogram2d(
    neuron_df["sim_error"], neuron_df["pred_loss"], bins=[xbins, ybins]
)
prob = counts / counts.sum()
prob_masked = np.ma.masked_where(prob == 0, prob)
im = ax.pcolormesh(
    xedges,
    yedges,
    prob_masked.T,
    norm=LogNorm() if log_prob else None,
    cmap=fade_to_color_cmap(con_colors[0], alpha_min=alpha_min, name="fade_to_color"),
)

ax.set_xlim(min_loss, max_loss)
ax.set_ylim(min_loss, max_loss)

# Use the same locator for x and y so ticks coincide
locator = ax.yaxis.get_major_locator()
ax.xaxis.set_major_locator(locator)

# Plot y=x line
ax.plot([0.0, max_loss], [0.0, max_loss], c="k", ls="--", lw=1, zorder=0)

plt.subplots_adjust(**fig_margins_lg)
plt.savefig(fig_dir + "loss_vs_prediction.svg", dpi=600)

# Add labels
ax.set_xlabel("Simulated error probability")
ax.set_ylabel("Predicted error probability")

# Add colorbar after saving (only shows in plt.show(), not in PDF)
cb = fig.colorbar(im, ax=ax)
cb.set_label("Probability")

plt.show()

# ------------------------------------------------------------------------------
# LOSS VS ROBUSTNESS (BINNED SCATTER)
# ------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width_md, height_md))

# Plot scatter points for each mean bin (same bins as first plot)
for i in range(n_mean_bins):
    # Filter data for this mean bin
    mean_mask = df_nonneg["mean_bin"] == i
    if mean_mask.sum() == 0:
        continue

    mean_mid = mean_mids[i]
    color = cmap(mean_norm(mean_mid))

    # Compute median loss and median robustness per variance bin
    grouped_loss = df_nonneg[mean_mask].groupby("var_bin")["sim_error"].median()
    grouped_rob = df_nonneg[mean_mask].groupby("var_bin")["robustness"].median()

    # Plot scatter for bins with data
    valid_bins = grouped_loss.index.dropna().astype(int)
    ax.scatter(
        grouped_rob[valid_bins],
        grouped_loss[valid_bins],
        c="white",
        edgecolors=color,
        s=20,
        zorder=2,
        rasterized=True,
    )

# Plot prediction line
r_vals = np.linspace(rob_min, rob_max, 100)
ax.plot(
    r_vals,
    (1.0 / np.pi) * np.arccos((1.0 + (eps / r_vals) ** 2) ** (-0.5)),
    c="k",
    lw=2,
    zorder=0,
)

# ax.set_xscale('log')
ax.set_yscale("log")
ax.set_xlim([0.0, None])
ax.set_ylim([1e-2, 0.2])

plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + "loss_vs_robustness_binned.svg", dpi=600)

# Print colorbar range to console
print(
    f"Colorbar range (avg incoming weight) — min: {mean_mids.min():.4f}, max: {mean_mids.max():.4f}"
)

# Add colorbar for interactive display (not in saved file)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mean_norm)
sm.set_array([])
plt.colorbar(sm, ax=ax)

# Add labels
ax.set_xlabel("Robustness")
ax.set_ylabel("Simulated error probability")

plt.show()


# ------------------------------------------------------------------------------
# ERROR VS ROBUSTNESS — PARAMETER SWEEPS (FIRING PROBABILITY / NOISE STRENGTH)
# ------------------------------------------------------------------------------
def plot_error_vs_robustness_sweep(
    sweep_col,
    sweep_vals,
    fixed_col,
    fixed_val,
    cmap_name,
    legend_title,
    label_fmt,
    analytical_fn,
    fname,
):
    """
    Binned scatter of simulated error rate vs robustness across a parameter
    sweep (p_fire or sigma), reusing the same mean/var-bin neuron populations
    as the baseline loss-vs-robustness plot. One color per sweep value,
    uniformly spaced along cmap_name regardless of the value itself.
    """
    sweep_error_df = error_df_full[error_df_full[fixed_col] == fixed_val][
        ["root_id", sweep_col, "sim_error"]
    ]

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.0, 1.0, len(sweep_vals)))

    fig, ax = plt.subplots(figsize=(width_md, height_md))
    r_vals = np.linspace(rob_min, rob_max, 100)

    for sweep_val, color in zip(sweep_vals, colors):
        merged = bin_assignments.merge(
            sweep_error_df[sweep_error_df[sweep_col] == sweep_val],
            on="root_id",
            how="inner",
        )

        for i in range(n_mean_bins):
            mean_mask = merged["mean_bin"] == i
            if mean_mask.sum() == 0:
                continue
            grouped_loss = merged[mean_mask].groupby("var_bin")["sim_error"].median()
            grouped_rob = merged[mean_mask].groupby("var_bin")["robustness"].median()
            valid_bins = grouped_loss.index.dropna().astype(int)
            ax.scatter(
                grouped_rob[valid_bins],
                grouped_loss[valid_bins],
                c="white",
                edgecolors=color,
                s=20,
                zorder=2,
                rasterized=True,
            )

        ax.plot(
            r_vals,
            analytical_fn(sweep_val, r_vals),
            color=color,
            lw=2,
            zorder=0,
            label=label_fmt(sweep_val),
        )

    ax.set_yscale("log")
    ax.set_xlim([0.0, None])

    plt.subplots_adjust(**fig_margins_md)
    plt.savefig(fig_dir + fname, dpi=600)

    ax.set_xlabel("Robustness")
    ax.set_ylabel("Simulated error probability")
    ax.legend(title=legend_title, frameon=False)

    plt.show()


# Neuron -> (mean_bin, var_bin) population assignment, independent of the sweep
bin_assignments = df_nonneg[["root_id", "mean_bin", "var_bin", "robustness"]]

# FIGURE: FIRING PROBABILITY SWEEP (sigma fixed at baseline) -------------------
# Includes p_fire=0.5 (the baseline), whose data is already cached in
# error_data.parquet and whose analytical curve (sparse_error_rate at p0=0.5)
# reduces algebraically to the same general_loss formula used elsewhere.
plot_error_vs_robustness_sweep(
    sweep_col="p_fire",
    sweep_vals=sorted(error_p_fire_vals),
    fixed_col="sigma",
    fixed_val=error_sigma,
    cmap_name=plasma_dark_r,
    legend_title="Firing probability",
    label_fmt=lambda p0: f"$p_f={p0:g}$",
    analytical_fn=lambda p0, r: np.array(
        [sparse_error_rate(p0, (1.0 + (1.0 / rr) ** 2) ** (-0.5)) for rr in r]
    ),
    fname="error_vs_robustness_pfire_sweep.svg",
)

# FIGURE: NOISE STRENGTH SWEEP (p_fire fixed at baseline) ----------------------
plot_error_vs_robustness_sweep(
    sweep_col="sigma",
    sweep_vals=sorted(error_sigma_vals),
    fixed_col="p_fire",
    fixed_val=0.5,
    cmap_name=viridis_dark,
    legend_title="Noise strength",
    label_fmt=lambda s: rf"$\sigma={s:g}$",
    analytical_fn=lambda s, r: (1.0 / np.pi) * np.arccos((1.0 + (s / r) ** 2) ** (-0.5)),
    fname="error_vs_robustness_sigma_sweep.svg",
)
