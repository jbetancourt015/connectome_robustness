"""
    This script generates figures comparing per-neuron robustness against
    shuffled null networks across all connectomes (scatter plots, histograms,
    and region-level breakdowns). Outputs to figures/robustness_comparison/raw/.
-------------------------------------------------------------------------------
created on:
    Tue 18 Feb 2025
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
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.sparse import load_npz
import logging
import pickle
import os
from params import connectome_colors

plt.rcParams.update({
    'font.family': 'Helvetica',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,   # thin axis lines for publication
    'axes.spines.top': False,    # remove top spine
    'axes.spines.right': False,  # remove right spine
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Figure size (in mm)
mm_to_in = 25.4
width = 40./mm_to_in
height = 40./mm_to_in
width_small = 30./mm_to_in
height_small = 30./mm_to_in
width_large = 80./mm_to_in
height_large = 80./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.24, right=1., bottom=0.24, top=1.)
fig_margins_small = dict(left=0.2, right=1., bottom=0.2, top=1.)
fig_margins_large = dict(left=0.12, right=0.95, bottom=0.12, top=0.95)

alpha_min = .2

# Output directory
fig_dir = "../figures/robustness_comparison/raw/"
os.makedirs(fig_dir, exist_ok=True)

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

# Number of bins for histogram
n_bins = 100

# Percentile for histogram axis range
p = 99.9

# Directories
processed_dir = '../processed_data/'
sim_dir = '../simulation_results/'

def load_robustness_dataset(connectome):
    df = pd.read_parquet(sim_dir + f"{connectome}_shuffled.parquet")
    required_cols = {"robustness", "shuffled_robustness"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {sorted(missing)} in {connectome}_shuffled.parquet"
        )
    return df

#------------------------------------------------------------------------------
# ROBUSTNESS PLOTTING FUNCTIONS
#------------------------------------------------------------------------------
def format_frac_pct(frac):
    """Format a fraction as a percentage string, using just enough decimal
    places that a value below 1.0 never displays as '100%'."""
    if frac >= 1.0:
        return "100%"
    for d in range(6):
        pct = round(100 * frac, d)
        if pct < 100:
            return f"{pct:.{d}f}%"
    return f"{100 * frac:.6f}%"

def plot_robustness(name, r_conn, r_shuf):
    r_min, r_max = 0., max(max(r_conn), max(r_shuf))
    
    # Set up figure - use large size for FAFB, small for others
    if name == "fafb":
        fig, ax_scatter = plt.subplots(figsize=(width_large, height_large))
    else:
        fig, ax_scatter = plt.subplots(figsize=(width, height))

    # Plot connectome robustness against shuffled-weight robustness
    ax_scatter.scatter(r_conn, r_shuf, color=connectome_colors[name], s=10, alpha=.2, rasterized=True)
    ax_scatter.plot([r_min,r_max], [r_min,r_max], c='k', ls='--', lw=1)
    ax_scatter.set_aspect('equal')
    
    ax_scatter.set_xlim(0.,None)
    ax_scatter.set_ylim(0.,None)
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Apply consistent margins
    if name == "fafb":
        plt.subplots_adjust(**fig_margins_large)
    else:
        plt.subplots_adjust(**fig_margins)

    # Save plot without labels
    plt.savefig(fig_dir + f"{name}_scatter.svg", dpi=600)

    # Add labels
    ax_scatter.set_xlabel('Robustness')
    ax_scatter.set_ylabel('Robustness (shuffled)')

    plt.show()


def plot_robustness_hist(name, r_conn, r_shuf, frac_decreased):
    r_min = 0.

    # Compute percentile-based upper limit from connectome robustness
    r_p = np.percentile(r_conn, p if name != "mouse_retina" else 99)

    # Set up figure - use large size for FAFB, small for others
    if name == "fafb":
        fig, ax_scatter = plt.subplots(figsize=(width_large, height_large))
    else:
        fig, ax_scatter = plt.subplots(figsize=(width, height))

    xbins = np.linspace(r_min, r_p, n_bins if name != "mouse_retina" else 20)
    ybins = xbins.copy()
    
    # Compute histogram counts and normalize to probability
    counts, xedges, yedges = np.histogram2d(r_conn, r_shuf, bins=[xbins, ybins])
    print('Total counts',counts.sum())
    prob = counts / counts.sum()  # Normalize so all bins sum to 1
    
    # Plot with pcolormesh (need to mask zeros for LogNorm)
    prob_masked = np.ma.masked_where(prob == 0, prob)
    im = ax_scatter.pcolormesh(
        xedges, yedges, prob_masked.T,
        norm=LogNorm(),
        cmap=fade_to_color_cmap(connectome_colors[name], alpha_min=alpha_min, name="fade_to_color"),
        rasterized = True
    )
    
    # Set axis limits to percentile-based range
    ax_scatter.set_xlim(0, r_p)
    ax_scatter.set_ylim(0, r_p)
    
    # diagonal & labels & text
    ax_scatter.plot([r_min, r_p], [r_min, r_p], "--", c="k", lw=1)
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Apply consistent margins
    if name == "fafb":
        plt.subplots_adjust(**fig_margins_large)
    else:
        plt.subplots_adjust(**fig_margins)

    # Save plot without labels
    plt.savefig(fig_dir + f"{name}_hist.svg", dpi=600)

    ax_scatter.text(0.05, 0.95, format_frac_pct(frac_decreased), transform=ax_scatter.transAxes,
                    fontsize=6, va='top', ha='left')

    # Add labels
    ax_scatter.set_xlabel('Robustness')
    ax_scatter.set_ylabel('Robustness (shuffled)')
    
    # Add colorbar after saving (only shows in plt.show(), not in PDF)
    cb = fig.colorbar(im, ax=ax_scatter)
    cb.set_label("Probability")

    plt.show()

#------------------------------------------------------------------------------
# COMPARE ROBUSTNESS
#------------------------------------------------------------------------------
# Connectomes shown in the figure: FAFB, BANC, MANC, C. elegans, mouse retina.
plot_order = ["fafb", "banc", "manc", "c_elegans", "mouse_retina"]

# Collect fraction of neurons with decreased robustness after shuffling
frac_decreased = {}

for name in plot_order:
    robustness_df = load_robustness_dataset(name).dropna()
    r_conn = robustness_df["robustness"].to_numpy()
    r_shuf = robustness_df["shuffled_robustness"].to_numpy()
    frac_decreased[name] = np.mean(r_conn > r_shuf)

    # Plot robustness comparisons
    plot_robustness(name, r_conn, r_shuf)
    plot_robustness_hist(name, r_conn, r_shuf, frac_decreased[name])

#------------------------------------------------------------------------------
# BAR PLOT: FRACTION OF NEURONS WITH DECREASED ROBUSTNESS
#------------------------------------------------------------------------------
# Colors and fractions in plot_order
bar_colors = [connectome_colors[n] for n in plot_order]
bar_fractions = [frac_decreased[n] for n in plot_order]

# Set up figure
fig_bar_margins = dict(left=0.17, right=0.95, bottom=0.35, top=0.94)
fig, ax = plt.subplots(figsize=(0.7*width_large, 1.25*height))

# Create bar plot
x_pos = np.arange(len(plot_order))
ax.bar(x_pos, bar_fractions, color=bar_colors, edgecolor='none', width=0.6)

# Configure axes
ax.set_xticks(x_pos)
# Manually specify labels with LaTeX italics for species/genus names
formatted_labels = ['FAFB', 'BANC', 'MANC', r'\textit{C. elegans}', 'Mouse retina']
ax.set_xticklabels(formatted_labels, rotation=70, ha='right', rotation_mode='anchor')
# Enable LaTeX rendering only for the italic label
for label in ax.get_xticklabels():
    if 'textit' in label.get_text():
        label.set_usetex(True)
ax.set_ylim(0., 1.)
ax.set_xlim(-0.5, len(plot_order) - 0.5)

# Apply margins
plt.subplots_adjust(**fig_bar_margins)

# Save plot without labels
plt.savefig(fig_dir + "robustness_decrease_fraction.svg", dpi=600)

# Add labels
ax.set_ylabel('Fraction of robust neurons')

plt.show()

#------------------------------------------------------------------------------
# FAFB BRAIN REGION ANALYSIS
#------------------------------------------------------------------------------
# Load FAFB per-neuron robustness with brain-region annotations
fafb_df = load_robustness_dataset('fafb').dropna()
if 'brain_region' not in fafb_df.columns:
    raise ValueError("FAFB robustness dataset must include a 'brain_region' column.")

with open(processed_dir + 'region_order.pkl', 'rb') as f:
    fafb_region_order = pickle.load(f)

# Create viridis colormap for regions
fafb_cmap = plt.get_cmap('viridis', len(fafb_region_order))
fafb_region_colors = {region: fafb_cmap(i) for i, region in enumerate(fafb_region_order)}

# Compute fraction with decreased robustness for each FAFB brain region
fafb_df_filtered = fafb_df[fafb_df['brain_region'] != 'Other Regions']
fafb_frac_series = (
    fafb_df_filtered.assign(decreased=fafb_df_filtered['robustness'] > fafb_df_filtered['shuffled_robustness'])
    .groupby('brain_region', sort=False)['decreased']
    .mean()
)
fafb_frac_decreased = {region: fafb_frac_series.get(region, np.nan) for region in fafb_region_order}

#------------------------------------------------------------------------------
# BAR PLOT: FRACTION OF NEURONS WITH DECREASED ROBUSTNESS BY FAFB BRAIN REGION
#------------------------------------------------------------------------------
# Prepare data for bar plot
fafb_bar_labels = list(fafb_region_order)
fafb_bar_fractions = [fafb_frac_decreased[region] for region in fafb_region_order]
fafb_bar_colors = [fafb_region_colors[region] for region in fafb_region_order]

# Set up figure
fig_bar_margins_region = dict(left=0.12, right=0.98, bottom=0.35, top=0.95)
fig, ax = plt.subplots(figsize=(1.35*width_large, 1.5*height))

# Create bar plot
x_pos = np.arange(len(fafb_bar_labels))
ax.axhline(frac_decreased['fafb'], color='k', ls='--', lw=0.75, zorder=0)
ax.bar(x_pos, fafb_bar_fractions, color=fafb_bar_colors, edgecolor='none', width=0.6)

# Configure axes
ax.set_xticks(x_pos)
ax.set_xticklabels(fafb_bar_labels, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylim(0., 1.)
ax.set_xlim(-0.5, len(fafb_bar_labels) - 0.5)

# Apply margins
plt.subplots_adjust(**fig_bar_margins_region)

# Save plot without labels
plt.savefig(fig_dir + "fafb_region_robustness_decrease.svg", dpi=600)

# Add labels
ax.set_ylabel('Fraction of robust neurons')

plt.show()

#------------------------------------------------------------------------------
# FAFB MEAN VS VARIANCE HISTOGRAM
#------------------------------------------------------------------------------
k_min_stat = 2

fafb_A = load_npz(processed_dir + 'fafb.npz')
fafb_k = fafb_A.getnnz(axis=0)
fafb_mask = fafb_k >= k_min_stat
fafb_A2 = fafb_A.copy()
fafb_A2.data = fafb_A2.data ** 2
fafb_mu  = np.asarray(fafb_A.sum(axis=0)).ravel()[fafb_mask] / fafb_k[fafb_mask]
fafb_mu2 = np.asarray(fafb_A2.sum(axis=0)).ravel()[fafb_mask] / fafb_k[fafb_mask]
fafb_var = fafb_mu2 - fafb_mu ** 2
fafb_pos = fafb_var > 1e-10
fafb_mu, fafb_var = fafb_mu[fafb_pos], fafb_var[fafb_pos]

xbins_mv = np.logspace(np.log10(fafb_mu.min()),  np.log10(fafb_mu.max()),  n_bins)
ybins_mv = np.logspace(np.log10(fafb_var.min()), np.log10(fafb_var.max()), n_bins)
counts_mv, xedges_mv, yedges_mv = np.histogram2d(fafb_mu, fafb_var, bins=[xbins_mv, ybins_mv])
prob_mv = counts_mv / counts_mv.sum()
prob_mv_masked = np.ma.masked_where(prob_mv == 0, prob_mv)

fig, ax = plt.subplots(figsize=(width_small, height_small))
im_mv = ax.pcolormesh(
    xedges_mv, yedges_mv, prob_mv_masked.T,
    norm=LogNorm(),
    cmap=fade_to_color_cmap(connectome_colors["fafb"], alpha_min=alpha_min, name="fade_to_color"),
    rasterized=True,
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1., xedges_mv[-1])
ax.set_ylim(yedges_mv[0], yedges_mv[-1])
lim_mv = [max(xedges_mv[0], yedges_mv[0]), min(xedges_mv[-1], yedges_mv[-1])]
ax.plot(lim_mv, lim_mv, ls='--', lw=1, c='k', alpha=0.5, zorder=3)
ax.tick_params(axis='both', labelsize=7)
plt.subplots_adjust(**fig_margins_small)

plt.savefig(fig_dir + 'fafb_mean_variance_hist.svg', dpi=600)

ax.set_xlabel('Average weight')
ax.set_ylabel('Variance in weights')
cb = fig.colorbar(im_mv, ax=ax)
cb.set_label('Probability')

plt.show()
