"""
    This script generates eight robustness plots for the FAFB dataset:
    1. CDF of normalized robustness
    2. Violin plots of normalized robustness by brain region
    3. CDF of normalized robustness by reciprocity decile
    4. Running median of robustness vs visual peripherality (sliding window)
    5. Running median of robustness vs olfactory peripherality (sliding window)
    6. Violin plots by visual pathway: ON (L1, Mi1, T4) and OFF (L2, Tm1, T5)
       motion pathways from R1-6 to LPTCs
    7. Violin plots by olfactory pathway (ORN, PN, KC, MBON)
    8. CDF of normalized robustness by neurotransmitter type
-------------------------------------------------------------------------------
created on:
    Tue 3 Feb 2026
-------------------------------------------------------------------------------
last change:
    Thu 22 Oct 2026
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import os
from figure_formatting import apply_style, log10_formatter, log10_minor_locator
from params import connectome_colors

apply_style()

mm_to_in = 25.4

# Panel dimensions in inches (scaled from mm reference)
width_sm = height_sm = 46 / mm_to_in
width_md = height_md = 55 / mm_to_in  # standard single panel (55 mm)
width_wd = 110 / mm_to_in  # wide panel (violins with region labels)
height_wd = 55 / mm_to_in
width_wd_sm = 70 / mm_to_in
height_wd_sm = 46 / mm_to_in

# Fixed margins for consistent axes size across all figures
fig_margins_sm = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_md = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_wd = dict(left=0.09, right=0.975, bottom=0.18, top=0.95)
fig_margins_wd_sm = dict(left=0.09, right=0.975, bottom=0.18, top=0.95)

# Output directory
fig_dir = "../figures/drosophila_robustness/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Plotting colors
palette = np.array([[242, 112, 0], [0, 168, 157], [203, 41, 123]]) / 255  # orange, teal, pink


# ------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# ------------------------------------------------------------------------------
def compute_cdf(series):
    """Compute CDF from a pandas series."""
    counts = series.round(9).value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values


def blend_with_white(color, alpha):
    """Pre-blend an RGB color with a white background at the given alpha.

    Used instead of matplotlib's alpha kwarg for long, many-vertex paths:
    Inkscape wraps alpha<1 strokes in a PDF transparency group, which
    triggers a macOS Quartz/Preview rendering bug that darkens part of
    the stroke. Baking the blend into an opaque color avoids the group.
    """
    color = np.asarray(color)
    return alpha * color + (1 - alpha) * np.ones(3)


# ------------------------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------------------------
processed_dir = "../processed_data/"
sim_dir = "../simulation_results/"

# Import neuron and periphery data
neuron_df = pd.read_parquet(processed_dir + "fafb_neurons.parquet")
peri_df = pd.read_parquet(sim_dir + "fafb_periphery_data.parquet")

# Add peripherality data to neuron dataframe
neuron_df = neuron_df.merge(peri_df, on="root_id")

# ------------------------------------------------------------------------------
# LOAD PRE-SAVED SHUFFLED NORMALIZED ROBUSTNESS
# ------------------------------------------------------------------------------
shuffled_rob_df = pd.read_parquet(
    sim_dir + "fafb_shuffled.parquet",
    columns=["root_id", "norm_robustness_shuffled"],
)
neuron_df = neuron_df.merge(shuffled_rob_df, on="root_id", how="left")

# ------------------------------------------------------------------------------
# PLOT 1: CDF OF NORMALIZED ROBUSTNESS
# ------------------------------------------------------------------------------
rob_vals, rob_cdf = compute_cdf(neuron_df["norm_robustness"])
median_rob = neuron_df["norm_robustness"].median()

# Compute CDF for shuffled network (filter out NaN values)
shuffled_valid = neuron_df["norm_robustness_shuffled"].dropna()
rob_vals_shuf, rob_cdf_shuf = compute_cdf(shuffled_valid)
median_rob_shuffled = neuron_df["norm_robustness_shuffled"].median()

# ------------------------------------------------------------------------------
# PLOTTING HELPERS (use the median_rob / median_rob_shuffled values above)
# ------------------------------------------------------------------------------
def add_median_vlines(ax):
    """Vertical reference lines at the connectome (solid) and shuffled (dashed) medians."""
    ax.plot([median_rob, median_rob], [0.0, 1.0], lw=1, c="k")
    ax.plot(
        [median_rob_shuffled, median_rob_shuffled],
        [0.0, 1.0],
        lw=1,
        c="k",
        alpha=0.5,
        ls="--",
    )


def add_median_hlines(ax):
    """Horizontal log10 reference lines at the connectome and shuffled medians."""
    ax.axhline(np.log10(median_rob), lw=1, c="k", zorder=0)
    ax.axhline(np.log10(median_rob_shuffled), lw=1, c="k", alpha=0.5, ls="--", zorder=0)


def style_violins(parts, colors):
    """Color each violin body and style the median line."""
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_linewidth(2.5)
        parts["cmedians"].set_color("black")


def format_log10_yaxis(ax):
    """Format the y-axis (log10 of normalized robustness) with log10 tick labels."""
    ax.set_ylim(-1, 2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log10_formatter))
    ax.yaxis.set_minor_locator(log10_minor_locator())
    ax.tick_params(axis="y", which="minor", length=2)


def draw_broken_axis_lines(fig, ax_left, ax_right, seed_median, running_x, running_median, color):
    """Draw the median reference lines and the seed -> running-median curve as
    figure-level artists, so they span both panels of the broken x-axis."""
    # Median reference lines
    x0_fig = ax_left.get_position().x0
    x1_fig = ax_right.get_position().x1
    to_fig = fig.transFigure.inverted()
    y_med_fig = to_fig.transform(ax_left.transData.transform([[0, median_rob]]))[0, 1]
    y_shuf_fig = to_fig.transform(
        ax_left.transData.transform([[0, median_rob_shuffled]])
    )[0, 1]
    fig.add_artist(
        mpl.lines.Line2D(
            [x0_fig, x1_fig],
            [y_med_fig, y_med_fig],
            transform=fig.transFigure,
            lw=1,
            color="k",
            clip_on=False,
            zorder=1,
        )
    )
    fig.add_artist(
        mpl.lines.Line2D(
            [x0_fig, x1_fig],
            [y_shuf_fig, y_shuf_fig],
            transform=fig.transFigure,
            lw=1,
            color="k",
            alpha=0.5,
            ls="--",
            clip_on=False,
            zorder=1,
        )
    )

    # Seed -> running median as one Line2D in figure space (no junction artifact)
    seed_disp = ax_left.transData.transform([[0, seed_median]])
    rm_disp = ax_right.transData.transform(np.column_stack([running_x, running_median]))
    all_fig = to_fig.transform(np.vstack([seed_disp, rm_disp]))
    fig.add_artist(
        mpl.lines.Line2D(
            all_fig[:, 0],
            all_fig[:, 1],
            transform=fig.transFigure,
            lw=2,
            color=color,
            zorder=4,
            clip_on=False,
        )
    )
    # Seed point added as figure-level artist so it renders above the line
    fig.add_artist(
        mpl.lines.Line2D(
            [all_fig[0, 0]],
            [all_fig[0, 1]],
            transform=fig.transFigure,
            marker="o",
            markersize=np.sqrt(40),
            lw=0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=5,
            clip_on=False,
        )
    )


# Set up figure
fig, ax = plt.subplots(figsize=(width_md, height_md))

add_median_vlines(ax)
ax.step(rob_vals, rob_cdf, where="post", lw=2, color=connectome_colors["fafb"], label="Connectome")
ax.step(
    rob_vals_shuf,
    rob_cdf_shuf,
    where="post",
    lw=2,
    color=blend_with_white(connectome_colors["fafb"], 0.5),
    label="Shuffled",
)


ax.set_xscale("log")
ax.set_xlim(0.3, 40)

plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + "cdf_normalized_robustness.svg", dpi=600)

ax.set_xlabel("Normalized robustness")
ax.set_ylabel("Cumulative probability")
ax.legend()

plt.show()

# Print median robustness to console
print(f"Median normalized robustness: {median_rob:.4f}")

# ------------------------------------------------------------------------------
# PLOT 2: VIOLIN PLOTS BY BRAIN REGION (LOG SCALE)
# ------------------------------------------------------------------------------
# Drop "Other Regions"
neuron_df_regions = neuron_df[neuron_df["brain_region"] != "Other Regions"].copy()

# Sort regions by median
region_order = (
    neuron_df_regions.groupby("brain_region")["norm_robustness"]
    .median()
    .sort_values()
    .index
)

cmap = plt.get_cmap("viridis", len(region_order))
region_colors = {region: cmap(i) for i, region in enumerate(region_order)}

# Transform to log scale (filter out non-positive values)
mask0 = neuron_df_regions["norm_robustness"] > 0
neuron_df_regions["log_robustness"] = np.log10(neuron_df_regions["norm_robustness"])

# Get data for violin plots (using log-transformed data)
violin_data = [
    neuron_df_regions.loc[
        mask0 & (neuron_df_regions["brain_region"] == region), "log_robustness"
    ].to_numpy(dtype=float)
    for region in region_order
]

# Set up figure (wider for region labels)
fig, ax = plt.subplots(figsize=(width_wd, height_wd))

# Add population median line and shuffled robustness line (behind violins)
add_median_hlines(ax)
ax.axhline(0.0, lw=1, c="k", ls="--", zorder=0, alpha=0.5)

parts = ax.violinplot(
    violin_data, showmeans=False, showmedians=True, showextrema=False, widths=0.9
)

style_violins(parts, [region_colors[region] for region in region_order])

# Labels/ticks
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right", rotation_mode="anchor")

format_log10_yaxis(ax)

plt.subplots_adjust(**fig_margins_wd)
plt.savefig(fig_dir + "violin_by_brain_region.svg", dpi=600)

ax.set_ylabel("Normalized robustness")
plt.show()

# ------------------------------------------------------------------------------
# PLOT 3: CDF BY RECIPROCITY DECILE
# ------------------------------------------------------------------------------
n_quantiles = 10

# Add deciles to dataset
neuron_df["reciprocity_q"] = pd.qcut(
    neuron_df["reciprocity"], q=n_quantiles, labels=False
)

# Pre-compute median reciprocity per decile
reciprocities = [
    neuron_df[neuron_df["reciprocity_q"] == i]["reciprocity"].median()
    for i in range(n_quantiles)
]
rec_min, rec_max = min(reciprocities), max(reciprocities)

# Normalize median reciprocities to [0, 1] for colormap lookup
rec_norm = [(rec - rec_min) / (rec_max - rec_min) for rec in reciprocities]

# Set up figure
fig, ax = plt.subplots(figsize=(width_sm, height_sm))

# Add population median line (behind data)
add_median_vlines(ax)

cmap = plt.get_cmap("coolwarm")

for i in range(n_quantiles):
    mask = neuron_df["reciprocity_q"] == i
    rob, cdf = compute_cdf(neuron_df[mask]["norm_robustness"])
    ax.step(rob, cdf, where="post", lw=2, c=cmap(rec_norm[i]))

ax.set_xscale("log")
ax.set_xlim(0.3, 40)

plt.subplots_adjust(**fig_margins_sm)
plt.savefig(fig_dir + "cdf_by_reciprocity_decile.svg", dpi=600)

# Print colorbar range to console
print(
    f"Colorbar range — min median reciprocity: {rec_min:.4f}, max median reciprocity: {rec_max:.4f}"
)

# Add colorbar for interactive display (not in saved file)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=rec_min, vmax=rec_max))
sm.set_array([])
plt.colorbar(sm, ax=ax)

ax.set_xlabel("Normalized robustness")
ax.set_ylabel("Cumulative probability")

plt.show()

# ------------------------------------------------------------------------------
# PLOT 4 & 5: RUNNING MEDIAN OF ROBUSTNESS VS PERIPHERALITY
# ------------------------------------------------------------------------------
# Sliding window parameters for running median
log_band_width = 0.8  # Width of band in log10(percentile) space
log_step_size = 0.1  # Step size for sliding window
log_max = np.log10(1)


def _compute_running_median(neuron_df, seed):
    mask_seed = neuron_df[f"distance_{seed}"] == 0
    mask_pos = neuron_df[f"distance_{seed}"] > 0
    seed_median = neuron_df[mask_seed]["norm_robustness"].median()
    pos_df = neuron_df[mask_pos].copy()
    pos_df["quantile"] = pos_df[f"distance_{seed}"].rank(pct=True)
    min_pct = pos_df["quantile"].min()
    log_min = np.log10(max(min_pct, 0.001))  # Avoid log(0)
    window_centers_log = np.arange(
        log_min + log_band_width / 2,
        log_max - log_band_width / 2 + log_step_size,
        log_step_size,
    )
    running_x, running_median = [], []
    for center_log in window_centers_log:
        low_pct = 10 ** (center_log - log_band_width / 2)
        high_pct = 10 ** (center_log + log_band_width / 2)
        band_mask = (pos_df["quantile"] >= low_pct) & (pos_df["quantile"] < high_pct)
        if band_mask.sum() > 0:
            running_x.append(10**center_log)
            running_median.append(pos_df[band_mask]["norm_robustness"].median())
    return seed_median, np.array(running_x), np.array(running_median)


seed_median_visual, running_x_visual, running_median_visual = _compute_running_median(
    neuron_df, "visual"
)
seed_median_olfactory, running_x_olfactory, running_median_olfactory = (
    _compute_running_median(neuron_df, "olfactory")
)

# Shared x-axis left limit: lower of the two current break points
shared_x_left = min(running_x_visual[0], running_x_olfactory[0])

# ------------------------------------------------------------------------------
# PLOT 4: RUNNING MEDIAN OF ROBUSTNESS VS VISUAL PERIPHERALITY
# ------------------------------------------------------------------------------
# Set up figure with broken x-axis. The wide figure keeps the same absolute break
# (left-panel width, gap, mark size) as the olfactory figure (width_sm, ratios 1:6,
# wspace 0.05); only the right panel absorbs the extra width.
ref_cell = (width_sm * (fig_margins_sm["right"] - fig_margins_sm["left"])) / (2 + 0.05)
ref_left_w = ref_cell * 2 / 7  # left panel width (in)
ref_right_w = ref_cell * 2 * 6 / 7
ref_gap = 0.05 * ref_cell  # gap between panels (in)
wd_axes_w = width_wd_sm * (fig_margins_wd_sm["right"] - fig_margins_wd_sm["left"])
wd_right_w = wd_axes_w - ref_left_w - ref_gap
fig, (ax_left, ax_right) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(width_wd_sm, height_wd_sm),
    gridspec_kw={
        "width_ratios": [ref_left_w, wd_right_w],
        "wspace": ref_gap / ((ref_left_w + wd_right_w) / 2),
    },
)
ax_right.set_facecolor("none")

# Shared y-axis range for both panels
ax_left.set_ylim(-0.1, 8.0)

ax_left.set_xlim(-0.5, 0.5)
ax_left.set_xticks([0])
ax_left.set_xticklabels(["0"])

# Right panel: log-scale x-axis (line drawn below after layout is set)
ax_right.set_xscale("log")
ax_right.set_xlim(left=shared_x_left, right=1)

# Hide the spines between the two axes to show the "break"
ax_left.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.tick_params(left=False)  # hide left ticks on right panel

# Add diagonal break marks (scale d for left panel to match visual angle)
d = 0.015  # size of diagonal lines in axes coords (vertical)
mark = d * ref_right_w  # horizontal mark half-size (in), same as olfactory figure
d_left = mark / ref_left_w  # horizontal extent in axes coords for left panel
d_right = mark / wd_right_w  # horizontal extent in axes coords for right panel

kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=0.8)
ax_left.plot((1 - d_left, 1 + d_left), (-d, +d), **kwargs)  # bottom-right

kwargs.update(transform=ax_right.transAxes)
ax_right.plot((-d_right, +d_right), (-d, +d), **kwargs)  # bottom-left

plt.subplots_adjust(**fig_margins_wd_sm)

draw_broken_axis_lines(
    fig, ax_left, ax_right, seed_median_visual, running_x_visual, running_median_visual, palette[1]
)
plt.savefig(fig_dir + "robustness_vs_visual_peripherality.svg", dpi=600)

# Format figure
ax_left.set_ylabel("Normalized robustness")
fig.supxlabel("Distance to periphery (quantile)", fontsize=9)

plt.show()

# ------------------------------------------------------------------------------
# PLOT 5: RUNNING MEDIAN OF ROBUSTNESS VS OLFACTORY PERIPHERALITY
# ------------------------------------------------------------------------------
# Set up figure with broken x-axis
fig, (ax_left, ax_right) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(width_sm, height_sm),
    gridspec_kw={"width_ratios": [1, 6], "wspace": 0.05},
)
ax_right.set_facecolor("none")

# Shared y-axis range for both panels
ax_left.set_ylim(-0.1, 8.0)

ax_left.set_xlim(-0.5, 0.5)
ax_left.set_xticks([0])
ax_left.set_xticklabels(["0"])

# Right panel: log-scale x-axis (line drawn below after layout is set)
ax_right.set_xscale("log")
ax_right.set_xlim(left=shared_x_left, right=1)

# Hide the spines between the two axes to show the "break"
ax_left.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.tick_params(left=False)  # hide left ticks on right panel

# Add diagonal break marks (scale d for left panel to match visual angle)
d = 0.015  # size of diagonal lines in axes coords (for right panel)
d_left = d * 6  # scale by width ratio (6:1) for left panel

kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=0.8)
ax_left.plot((1 - d_left, 1 + d_left), (-d, +d), **kwargs)  # bottom-right

kwargs.update(transform=ax_right.transAxes)
ax_right.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left

plt.subplots_adjust(**fig_margins_sm)

draw_broken_axis_lines(
    fig, ax_left, ax_right, seed_median_olfactory, running_x_olfactory, running_median_olfactory, palette[2]
)
plt.savefig(fig_dir + "robustness_vs_olfactory_peripherality.svg", dpi=600)

# Format figure
ax_left.set_ylabel("Normalized robustness")
fig.supxlabel("Distance to periphery (quantile)", fontsize=9)

plt.show()

# ------------------------------------------------------------------------------
# PLOT 6: VIOLIN PLOTS – VISUAL PATHWAY (ON/OFF MOTION PATHWAYS + LPTCs)
# ------------------------------------------------------------------------------
# Define visual pathway steps in biological order (periphery → deeper processing)
visual_pathway_order = ["R1-6", "L1", "L2", "Mi1", "Tm1", "T4", "T5", "LPTCs"]

# Lobula plate tangential cells (HS, VS, H1/H2, DCH/VCH), which T4 and T5 converge onto
lptc_types = ["HSN", "HSE", "HSS", "H1", "H2", "DCH", "VCH"] + [
    f"VS{i}" for i in range(1, 7)
]

# Assign each neuron to its visual pathway step
visual_masks = {
    "R1-6": neuron_df["primary_type"] == "R1-6",
    "L1": neuron_df["primary_type"] == "L1",
    "L2": neuron_df["primary_type"] == "L2",
    "Mi1": neuron_df["primary_type"] == "Mi1",
    "Tm1": neuron_df["primary_type"] == "Tm1",
    "T4": neuron_df["primary_type"].str.startswith("T4", na=False),
    "T5": neuron_df["primary_type"].str.startswith("T5", na=False),
    "LPTCs": neuron_df["primary_type"].isin(lptc_types),
}

# Violin colors: shared stages teal, ON motion pathway grey, OFF orange
on_color = "#999999"
off_color = palette[0]
visual_colors = {
    "R1-6": palette[1],
    "L1": on_color,
    "L2": off_color,
    "Mi1": on_color,
    "Tm1": off_color,
    "T4": on_color,
    "T5": off_color,
    "LPTCs": palette[1],
}

# Prepare log-transformed robustness data for each step
visual_violin_data = []
for step in visual_pathway_order:
    mask = visual_masks[step] & (neuron_df["norm_robustness"] > 0)
    visual_violin_data.append(
        np.log10(neuron_df.loc[mask, "norm_robustness"]).to_numpy(dtype=float)
    )

# Set up figure
fig, ax = plt.subplots(figsize=(width_wd_sm, height_wd_sm))

# Population median line (behind violins)
add_median_hlines(ax)

parts = ax.violinplot(
    visual_violin_data, showmeans=False, showmedians=True, showextrema=False, widths=0.9
)

style_violins(parts, [visual_colors[step] for step in visual_pathway_order])

# Labels / ticks
ax.set_xticks(range(1, len(visual_pathway_order) + 1))
ax.set_xticklabels(visual_pathway_order)

format_log10_yaxis(ax)

plt.subplots_adjust(**fig_margins_wd_sm)
plt.savefig(fig_dir + "violin_visual_pathway.svg", dpi=600)

ax.set_ylabel("Normalized robustness")
plt.show()

# ------------------------------------------------------------------------------
# PLOT 7: VIOLIN PLOTS – OLFACTORY PATHWAY (ORN, PN, KC, MBON)
# ------------------------------------------------------------------------------
# Define olfactory pathway steps in biological order (periphery → deeper processing)
olfactory_pathway_order = ["ORN", "PN", "KC", "MBON"]

# Assign each neuron to its olfactory pathway step
olfactory_masks = {
    "ORN": neuron_df["primary_type"].str.startswith("ORN", na=False),
    "PN": neuron_df["class"] == "ALPN",
    "KC": neuron_df["primary_type"].str.startswith("KC", na=False),
    "MBON": neuron_df["primary_type"].str.startswith("MBON", na=False),
}

# Prepare log-transformed robustness data for each step
olfactory_violin_data = []
for step in olfactory_pathway_order:
    mask = olfactory_masks[step] & (neuron_df["norm_robustness"] > 0)
    olfactory_violin_data.append(
        np.log10(neuron_df.loc[mask, "norm_robustness"]).to_numpy(dtype=float)
    )

# Set up figure
fig, ax = plt.subplots(figsize=(width_sm, height_sm))

# Population median line (behind violins)
add_median_hlines(ax)

parts = ax.violinplot(
    olfactory_violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9,
)

style_violins(parts, [palette[2]] * len(olfactory_pathway_order))

# Labels / ticks
ax.set_xticks(range(1, len(olfactory_pathway_order) + 1))
ax.set_xticklabels(olfactory_pathway_order)

format_log10_yaxis(ax)

plt.subplots_adjust(**fig_margins_sm)
plt.savefig(fig_dir + "violin_olfactory_pathway.svg", dpi=600)

ax.set_ylabel("Normalized robustness")
plt.show()

# ------------------------------------------------------------------------------
# PLOT 8: CDF BY NEUROTRANSMITTER TYPE
# ------------------------------------------------------------------------------
# Restrict to neurons with an assigned nt_type
nt_df = neuron_df.dropna(subset=["nt_type"])

# Order NT types by median normalized robustness
nt_order = (
    nt_df.groupby("nt_type")["norm_robustness"].median().sort_values().index.tolist()
)

n_nt = len(nt_order)
cmap_nt = plt.get_cmap("managua")
nt_colors = [cmap_nt(i / (n_nt - 1)) for i in range(n_nt)]

fig, ax = plt.subplots(figsize=(width_sm, height_sm))

add_median_vlines(ax)

for i, nt in enumerate(nt_order):
    mask = nt_df["nt_type"] == nt
    rob, cdf = compute_cdf(nt_df[mask]["norm_robustness"])
    ax.step(rob, cdf, where="post", lw=2, c=nt_colors[i])

ax.set_xscale("log")
ax.set_xlim(0.3, 40)

plt.subplots_adjust(**fig_margins_sm)
plt.savefig(fig_dir + "nt_robustness_distribution.svg", dpi=600)

ax.set_xlabel("Normalized robustness")
ax.set_ylabel("Cumulative probability")
ax.legend(
    [plt.Line2D([0], [0], color=nt_colors[i], lw=2) for i in range(n_nt)],
    nt_order,
    loc="upper left",
)

plt.show()
