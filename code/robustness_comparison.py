"""
    This script generates figures comparing per-neuron robustness against
    shuffled null networks across all connectomes (scatter plots, histograms,
    and region-level breakdowns). Outputs to figures/robustness_comparison/.
-------------------------------------------------------------------------------
created on:
    Tue 18 Feb 2025
-------------------------------------------------------------------------------
last change:
    Mon 29 Jun 2026
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
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy.sparse import load_npz
from matplotlib.ticker import ScalarFormatter
import logging
import pickle
import os

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
fig_dir = "../../figures/robustness_comparison/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc', 'drosophila_manc']

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]])/255;

# Map data_idx → con_colors index (matches connection_strength_distributions.py)
data_to_color = {
    0: 6,  # central brain
    1: 7,  # optic medulla
    2: 3,  # c_elegans
    3: 5,  # platynereis
    4: 4,  # mouse_retina
    5: 0,  # FAFB whole brain
    6: 1,  # BANC
    7: 2,  # MANC
}

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

def load_robustness_dataset(connectome, suffix=''):
    df = pd.read_parquet(sim_dir + f"{connectome}_shuffled{suffix}.parquet")
    required_cols = {"robustness", "shuffled_robustness"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {sorted(missing)} in {connectome}_shuffled{suffix}.parquet"
        )
    return df

#------------------------------------------------------------------------------
# ROBUSTNESS PLOTTING FUNCTIONS
#------------------------------------------------------------------------------
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
formatter.set_powerlimits((0, 0))

def log_kde_to_original(q_vals, n_grid=512, bw=None):
    u = np.log10(q_vals[q_vals > 0])
    kde = gaussian_kde(u, bw_method=bw)  # bw=None => Scott’s rule
    umin, umax = u.min(), u.max()
    u_grid = np.linspace(umin, umax, n_grid)
    x_grid = np.power(10.0, u_grid)                  # back to original Q
    f_u    = kde(u_grid)                             # density in log10-space
    f_x    = f_u / (x_grid * np.log(10.0))           # change-of-variables
    return x_grid, f_x

def plot_robustness(data_idx, Q1, Q2, null_net, log_axes=False):    
    Q_min, Q_max = min(min(Q1),min(Q2)), max(max(Q1),max(Q2))
    
    Q_min = 0.
    
    # Set up figure - use large size for FAFB (data_idx=5), small for others
    if data_idx == 5:
        fig, ax_scatter = plt.subplots(figsize=(width_large, height_large))
    else:
        fig, ax_scatter = plt.subplots(figsize=(width, height))
    
    # Plot sensitivities for random weight
    ax_scatter.scatter(Q1, Q2, color=con_colors[data_to_color[data_idx]], s=10, alpha=.2, rasterized=True)
    ax_scatter.plot([Q_min,Q_max], [Q_min,Q_max], c='k', ls='--', lw=1)
    ax_scatter.set_aspect('equal')
    
    ax_scatter.set_xlim(0.,None)
    ax_scatter.set_ylim(0.,None)
    
    if log_axes:
        # Set axis scales
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Apply consistent margins
    if data_idx == 5:
        plt.subplots_adjust(**fig_margins_large)
    else:
        plt.subplots_adjust(**fig_margins)
    
    # Save plot without labels
    plt.savefig(fig_dir + f"{connectomes[data_idx]}_scatter.svg", dpi=600)

    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Shuffled weight robustness' if null_net=='rand_weight' else 'Poisson robustness')

    plt.show()


def plot_robustness_hist(data_idx, Q1, Q2, log_axes=False, labels=True):
    Q_min = min(min(Q1),min(Q2))

    # Compute percentile-based upper limit from connectome robustness
    r_p = np.percentile(Q1, p if data_idx!= 4 else 99)

    # Set up figure - use large size for FAFB (data_idx=5), small for others
    if data_idx == 5:
        fig, ax_scatter = plt.subplots(figsize=(width_large, height_large))
    else:
        fig, ax_scatter = plt.subplots(figsize=(width, height))
    
    if log_axes:
        # Create histogram bins
        xbins = np.logspace(np.log10(Q_min), np.log10(r_p), n_bins if data_idx!=4 else 20)
        ybins = xbins.copy()
        
        # Set axis scales
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
        
    else:
        Q_min = 0.
        xbins = np.linspace(Q_min, r_p, n_bins if data_idx!=4 else 20)
        ybins = xbins.copy()
    
    # Compute histogram counts and normalize to probability
    counts, xedges, yedges = np.histogram2d(Q1, Q2, bins=[xbins, ybins])
    print('Total counts',counts.sum())
    prob = counts / counts.sum()  # Normalize so all bins sum to 1
    
    # Plot with pcolormesh (need to mask zeros for LogNorm)
    prob_masked = np.ma.masked_where(prob == 0, prob)
    im = ax_scatter.pcolormesh(
        xedges, yedges, prob_masked.T,
        norm=LogNorm(),
        cmap=fade_to_color_cmap(con_colors[data_to_color[data_idx]], alpha_min=alpha_min, name="fade_to_color"),
        rasterized = True
    )
    
    # Set axis limits to percentile-based range
    ax_scatter.set_xlim(0, r_p)
    ax_scatter.set_ylim(0, r_p)
    
    # diagonal & labels & text
    ax_scatter.plot([Q_min, r_p], [Q_min, r_p], "--", c="k", lw=1)
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Apply consistent margins
    if data_idx == 5:
        plt.subplots_adjust(**fig_margins_large)
    else:
        plt.subplots_adjust(**fig_margins)
    
    # Save plot without labels
    plt.savefig(fig_dir + f"{connectomes[data_idx]}_hist.svg", dpi=600)

    frac_below = np.mean(Q1 > Q2)
    ax_scatter.text(0.05, 0.95, f'{100*frac_below:.1f}%', transform=ax_scatter.transAxes,
                    fontsize=6, va='top', ha='left')

    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Shuffled weight robustness')
    
    # Add colorbar after saving (only shows in plt.show(), not in PDF)
    cb = fig.colorbar(im, ax=ax_scatter)
    cb.set_label("Probability")

    plt.show()

#------------------------------------------------------------------------------
# COMPARE ROBUSTNESS
#------------------------------------------------------------------------------
norm = False
log_axes = False

# Collect fraction of neurons with decreased robustness after shuffling
frac_decreased_list = []
processed_indices = []

for data_idx in range(len(connectomes)):
    # Use multinomial shuffle for all connectomes except mouse retina (data_idx=4)
    suffix = '' if data_idx == 4 else '_multinomial'
    robustness_df = load_robustness_dataset(connectomes[data_idx], suffix=suffix).dropna()
    Q1 = robustness_df["robustness"].to_numpy()
    Q2 = robustness_df["shuffled_robustness"].to_numpy()
    frac_decreased = np.mean(Q1 > Q2)
    frac_decreased_list.append(frac_decreased)
    processed_indices.append(data_idx)

    # Plot robustness comparisons
    plot_robustness(data_idx, Q1, Q2, 'rand_weight', log_axes=log_axes)
    plot_robustness_hist(data_idx, Q1, Q2, log_axes=log_axes)

#------------------------------------------------------------------------------
# BAR PLOT: FRACTION OF NEURONS WITH DECREASED ROBUSTNESS
#------------------------------------------------------------------------------
# Abbreviated labels for readability
short_names = ['Central Brain', 'Optic Medulla', 'C. elegans', 'Platynereis', 
               'Mouse retina', 'FAFB', 'BANC', 'MANC']
# Indices of labels that should be italicized (species/genus names)
italic_indices = [2, 3]

# Get labels and colors for selected connectomes
plot_indices = [5, 6, 7, 2, 4]
bar_labels = [short_names[i] for i in plot_indices]

bar_colors = [con_colors[data_to_color[i]] for i in plot_indices]

# Map processed indices to fractions, then filter for plot_indices
frac_map = dict(zip(processed_indices, frac_decreased_list))
bar_fractions = [frac_map[i] for i in plot_indices]

# Set up figure
fig_bar_margins = dict(left=0.17, right=0.95, bottom=0.35, top=0.94)
fig, ax = plt.subplots(figsize=(0.7*width_large, 1.25*height))

# Create bar plot
x_pos = np.arange(len(plot_indices))
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
ax.set_xlim(-0.5, len(plot_indices) - 0.5)

# Apply margins
plt.subplots_adjust(**fig_bar_margins)

# Save plot without labels
plt.savefig(fig_dir + "robustness_decrease_fraction.svg", dpi=600)

# Add labels
ax.set_ylabel('Fraction with decreased robustness')

plt.show()

#------------------------------------------------------------------------------
# FAFB BRAIN REGION ANALYSIS
#------------------------------------------------------------------------------
# Load FAFB per-neuron robustness with brain-region annotations
fafb_df = load_robustness_dataset('drosophila_whole_brain', suffix='_multinomial').dropna()
if 'brain_region' not in fafb_df.columns:
    raise ValueError("FAFB robustness dataset must include a 'brain_region' column.")

with open(processed_dir + 'region_order.pkl', 'rb') as f:
    fafb_region_order = pickle.load(f)

# Create viridis colormap for regions
fafb_cmap = plt.get_cmap('viridis', len(fafb_region_order))
fafb_region_colors = {r: fafb_cmap(i) for i, r in enumerate(fafb_region_order)}

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
fafb_bar_fractions = [fafb_frac_decreased[r] for r in fafb_region_order]
fafb_bar_colors = [fafb_region_colors[r] for r in fafb_region_order]

# Set up figure
fig_bar_margins_region = dict(left=0.12, right=0.98, bottom=0.35, top=0.95)
fig, ax = plt.subplots(figsize=(1.35*width_large, 1.5*height))

# Create bar plot
x_pos = np.arange(len(fafb_bar_labels))
ax.axhline(frac_map[5], color='k', ls='--', lw=0.75, zorder=0)
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
ax.set_ylabel('Fraction with decreased robustness')

plt.show()

#------------------------------------------------------------------------------
# FAFB MEAN VS VARIANCE HISTOGRAM
#------------------------------------------------------------------------------
k_min_stat = 2

fafb_A = load_npz(processed_dir + 'drosophila_whole_brain.npz')
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
    cmap=fade_to_color_cmap(con_colors[data_to_color[5]], alpha_min=alpha_min, name="fade_to_color"),
    rasterized=True,
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(xedges_mv[0], xedges_mv[-1])
ax.set_ylim(yedges_mv[0], yedges_mv[-1])
lim_mv = [max(xedges_mv[0], yedges_mv[0]), min(xedges_mv[-1], yedges_mv[-1])]
ax.plot(lim_mv, lim_mv, ls='--', lw=1, c='k', alpha=0.5, zorder=3)
ax.tick_params(axis='both', labelsize=7)
plt.subplots_adjust(**fig_margins_small)

plt.savefig(fig_dir + 'fafb_mean_variance_hist.svg', dpi=600)

ax.set_xlabel('Mean incoming weight')
ax.set_ylabel('Variance in weight')
cb = fig.colorbar(im_mv, ax=ax)
cb.set_label('Probability')

plt.show()
