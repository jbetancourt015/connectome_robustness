"""
    This script plots the distribution of connection strengths for all connectomes.
-------------------------------------------------------------------------------
created on:
    Mon 15 Dec 2025
-------------------------------------------------------------------------------
last change:
    Thu 22 Oct 2026
-------------------------------------------------------------------------------
notes:
    FAFB is plotted with shuffled/Poisson overlays and a per-region grid; all other
    connectomes are plotted from processed_data/{name}.npz.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from scipy.sparse import load_npz
from scipy.stats import poisson as poisson_dist

from figure_formatting import apply_style, log_format
from params import connectome_colors

apply_style()


# Figure size (in mm)
mm_to_in = 25.4
width = 40.0 / mm_to_in
height = 40.0 / mm_to_in
width_large = 62.0 / mm_to_in
height_large = 62.0 / mm_to_in
width_grid = 102.0 / mm_to_in
height_grid = 62.0 / mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
fig_margins_large = dict(left=0.15, right=0.95, bottom=0.15, top=0.95)
fig_margins_grid = dict(left=0.1, right=0.97, bottom=0.15, top=0.95)

processed_dir = "../processed_data/"
sim_dir = "../simulation_results/"
output_dir = "../figures/connection_strength_distributions/raw/"
os.makedirs(output_dir, exist_ok=True)

# Import connections data
conn_df = pd.read_parquet(processed_dir + "fafb_connections.parquet")


# ------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ------------------------------------------------------------------------------
def empirical_hist_pd(s):
    # keep finite values only
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return np.array([]), np.array([])

    # Bins are (i, i+1]  <=>  ceil(x)-1
    # Example: x=0.2 -> bin 0, x=1.0 -> bin 0, x=1.7 -> bin 1
    bin_idx = np.ceil(x).astype("int64") - 1
    bin_idx = pd.Series(bin_idx, index=x.index)

    g = x.groupby(bin_idx, sort=True)

    s_unique = g.mean().to_numpy()  # mean value in each occupied bin
    Ps = (g.size() / len(x)).to_numpy()  # mass in each occupied bin

    return s_unique, Ps


def empirical_hist_np(data):
    """Version of empirical_hist for numpy arrays (sparse matrix data)."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.array([]), np.array([])

    rounded_data = np.round(data).astype(int)
    s_unique, counts = np.unique(rounded_data, return_counts=True)

    # Get non-zero strengths
    mask = s_unique > 0
    s_unique, counts = s_unique[mask], counts[mask]

    Ps = counts / np.sum(counts)

    return s_unique, Ps


# ------------------------------------------------------------------------------
# FAFB DISTRIBUTIONS
# ------------------------------------------------------------------------------
# Get distribution of weights
s_fafb, P_fafb = empirical_hist_pd(conn_df["syn_count"])

# Get distribution of multinomial shuffled weights
fafb_shuffled_weights = np.load(sim_dir + "fafb_shuffled_weights.npz")["weights"]
s_shuffled, P_shuffled = empirical_hist_np(fafb_shuffled_weights)

# Get distribution of global shuffled weights
fafb_global_shuffled_weights = np.load(sim_dir + "fafb_global_shuffled_weights.npz")[
    "weights"
]
s_global, P_global = empirical_hist_np(fafb_global_shuffled_weights)

# Poisson PMF with lambda = total synapses / total connections
lam = conn_df["syn_count"].sum() / len(conn_df)
k_max = int(lam + 10 * np.sqrt(lam))  # a few std devs out is sufficient
k_vals = np.arange(1, k_max + 1)
pmf_vals = poisson_dist.pmf(k_vals, lam)

# FIGURE: Distribution of all connections in the FAFB dataset------------------
# Set up figure
fig, ax = plt.subplots(figsize=(width_large, height_large))

ax.plot(k_vals, pmf_vals, color="black", lw=1, zorder=1)
ax.scatter(
    s_global,
    P_global,
    color="cornflowerblue",
    s=10,
    rasterized=True,
    clip_on=False,
    zorder=3,
)
ax.scatter(
    s_shuffled, P_shuffled, color="grey", s=10, rasterized=True, clip_on=False, zorder=4
)
ax.scatter(
    s_fafb,
    P_fafb,
    color=connectome_colors["fafb"],
    s=10,
    rasterized=True,
    clip_on=False,
    zorder=5,
)
log_format(ax)

ax.set_xlim(1.0, None)
ax.set_ylim(3e-8, 1.0)

plt.subplots_adjust(**fig_margins_large)
plt.savefig(output_dir + "fafb_distribution.svg", dpi=600)
ax.set_xlabel("Synapses per connection")
ax.set_ylabel("Probability")
plt.show()
plt.close()

# Distribution of connections by brain region (using plurality-assigned regions)
region_df = conn_df[["syn_count", "conn_brain_region"]].copy()
region_df = region_df.rename(columns={"conn_brain_region": "brain_region"})
region_df = region_df[region_df["brain_region"] != "Other Regions"]

# Get region order
with open(processed_dir + "region_order.pkl", "rb") as f:
    region_order = pickle.load(f)

cmap = plt.get_cmap("viridis", len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: Distribution of connections by brain region (Grid)--------------------
n_cols = 5
n_rows = 3

excluded = [(0, 4), (1, 4)]
excluded_indices = [exc[0] * n_cols + exc[1] for exc in excluded]

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(width_grid, height_grid), sharex=True, sharey=True
)

region_idx = 0

for idx in tqdm(range(n_cols * n_rows)):
    ax = axes.flat[idx]

    if np.isin(idx, excluded_indices):
        ax.set_visible(False)
    else:
        r = region_order[region_idx]
        mask = region_df["brain_region"] == r
        s_region, P_region = empirical_hist_pd(region_df[mask]["syn_count"])
        ax.scatter(
            s_region,
            P_region,
            color=region_colors[r],
            s=10,
            rasterized=True,
            clip_on=False,
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Split two-word region names into two lines
        if r == "Periesophageal NP":
            region_label = "Periesoph-\nageal\nNP"
        else:
            region_label = r.replace(" ", "\n")
        ax.text(
            0.95,
            0.95,
            region_label,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
        )
        # Advance region index
        region_idx += 1

plt.setp(axes, xlim=(1.0, None), ylim=(None, 1.0))
plt.subplots_adjust(**fig_margins_grid, wspace=0.08, hspace=0.08)
plt.savefig(output_dir + "brain_region_grid.svg", dpi=600)
plt.show()
plt.close()

# ------------------------------------------------------------------------------
# OTHER CONNECTOME DISTRIBUTIONS (from processed connectome data)
# ------------------------------------------------------------------------------
# BANC, MANC, C. elegans and mouse retina. FAFB is handled above because it needs
# the neuron-level data (regions) and the shuffled/Poisson overlays.
for name in ["banc", "manc", "c_elegans", "mouse_retina"]:
    # Load connectome and extract synapse counts
    A = load_npz(processed_dir + f"{name}.npz")
    conn_strength = A.data

    # Get distribution of weights
    s_species, P_species = empirical_hist_np(conn_strength)

    # FIGURE: Distribution of all connections----------------------------------
    fig, ax = plt.subplots(figsize=(width, height))

    ax.scatter(
        s_species,
        P_species,
        color=connectome_colors[name],
        s=10,
        rasterized=True,
        clip_on=False,
        zorder=5,
    )
    log_format(ax)
    ax.set_xlim(1.0, None)
    ax.set_ylim(None, 1.0)

    plt.subplots_adjust(**fig_margins)
    plt.savefig(output_dir + f"{name}_distribution.svg", dpi=600)
    if name == "mouse_retina":
        ax.set_xlabel("Contact area per connection (µm$^2$)")
    else:
        ax.set_xlabel("Synapses per connection")
    ax.set_ylabel("Probability")
    plt.show()
    plt.close()
