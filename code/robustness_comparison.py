"""
    This script calculates statistics of single-neuron sensitivities
-------------------------------------------------------------------------------
created on:
    Tue 18 Feb 2025
-------------------------------------------------------------------------------
last change:
    Thu 12 Feb 2026
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import network_processing
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from time import time
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter
from scipy.sparse import coo_matrix
import pickle
import logging

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
width_large = 80./mm_to_in
height_large = 80./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.24, right=1., bottom=0.24, top=1.)
fig_margins_large = dict(left=0.12, right=0.95, bottom=0.12, top=0.95)

alpha_min = .2

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

#------------------------------------------------------------------------------
# SENSITIVITY PLOTTING FUNCTIONS
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

def plot_robustness(data_idx, A, A_rand, eta, null_net, normalized=False, log_axes=False):    
    # Get sensitivity of connectomes
    Q1 = network_processing.compute_robustness(A, eta, normalized)
    Q2 = network_processing.compute_robustness(A_rand, eta, normalized)
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
    plt.savefig(f"../../paper_figures/robustness_comparison/{connectomes[data_idx]}_scatter.svg", dpi=600)

    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Shuffled weight robustness' if null_net=='rand_weight' else 'Poisson sensitivity')

    plt.show()


def plot_robustness_hist(data_idx, A, A_rand, eta, null_net, normalized=False, log_axes=False, labels=True):
    # Get sensitivity of connectomes
    Q1 = network_processing.compute_robustness(A, eta, normalized)
    Q2 = network_processing.compute_robustness(A_rand, eta, normalized)
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
    plt.savefig(f"../../paper_figures/robustness_comparison/{connectomes[data_idx]}_hist.svg", dpi=600)
    
    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Shuffled weight robustness')
    
    # Add colorbar after saving (only shows in plt.show(), not in PDF)
    cb = fig.colorbar(im, ax=ax_scatter)
    cb.set_label("Probability")

    plt.show()

#------------------------------------------------------------------------------
# COMPARE SENSITIVITIES
#------------------------------------------------------------------------------
thresholded = False
scheme = 'rand_weight'
norm = False
log_axes = False

# Collect fraction of neurons with decreased robustness after shuffling
frac_decreased_list = []
processed_indices = []

for data_idx in range(len(connectomes)):
    # Get connectome
    A = network_processing.load_connectome(data_idx, thresholded=thresholded if data_idx==5 else False)
    N = A.shape[0]
    
    n_threshold = 1
    if data_idx == 5 and thresholded:
        n_threshold = 5
    elif data_idx == 6:
        n_threshold = 3

    if not((scheme!='rand_weight') and data_idx==4):
        # Get null network
        A_rand = network_processing.null_network(A, scheme=scheme, 
                                                conn_type='cont' if data_idx==4 else 'disc', 
                                                n_threshold=n_threshold)
    
        # Compute robustness and fraction with decreased robustness
        Q1 = network_processing.compute_robustness(A, 1., norm)
        Q2 = network_processing.compute_robustness(A_rand, 1., norm)
        frac_decreased = np.mean(Q1 > Q2)  # fraction where shuffling decreased robustness
        frac_decreased_list.append(frac_decreased)
        processed_indices.append(data_idx)
    
        # Plot sensitivities
        plot_robustness(data_idx, A, A_rand, 1., 'rand_weight', normalized=norm, log_axes=log_axes)
        plot_robustness_hist(data_idx, A, A_rand, 1., 'rand_weight', normalized=norm, log_axes=log_axes)

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
fig_bar_margins = dict(left=0.17, right=0.95, bottom=0.35, top=0.95)
fig, ax = plt.subplots(figsize=(0.7*width_large, 1.5*height))

# Create bar plot
x_pos = np.arange(len(plot_indices))
ax.bar(x_pos, bar_fractions, color=bar_colors, edgecolor='none', width=0.6)

# Configure axes
ax.set_xticks(x_pos)
# Manually specify labels with LaTeX italics for species/genus names
formatted_labels = ['FAFB', 'BANC', 'MANC', r'\textit{C. elegans}', 'Mouse retina']
ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
# Enable LaTeX rendering only for the italic label
for label in ax.get_xticklabels():
    if 'textit' in label.get_text():
        label.set_usetex(True)
ax.set_ylim(0., 1.)
ax.set_xlim(-0.5, len(plot_indices) - 0.5)

# Apply margins
plt.subplots_adjust(**fig_bar_margins)

# Save plot without labels
plt.savefig("../../paper_figures/robustness_comparison/robustness_decrease_fraction.svg", dpi=600)

# Add labels
ax.set_ylabel('Fraction with decreased robustness')

plt.show()

#------------------------------------------------------------------------------
# FAFB BRAIN REGION ANALYSIS
#------------------------------------------------------------------------------
# Data directories
data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Load parquet data files
conn_df = pd.read_parquet(processed_dir + 'connections_data.parquet')
neuron_df = pd.read_parquet(processed_dir + 'neuron_data.parquet')

# Build sparse matrix with index mapping
nodes_index = pd.Index(
    pd.unique(pd.concat([conn_df['pre_root_id'], conn_df['post_root_id']], ignore_index=True)),
    name="root_id"
)

id_to_idx = pd.Series(np.arange(nodes_index.size), index=nodes_index)
idx_to_id = nodes_index.to_numpy()

# Map edges to integer rows/cols
rows = id_to_idx.reindex(conn_df['pre_root_id']).to_numpy()
cols = id_to_idx.reindex(conn_df['post_root_id']).to_numpy()
data = conn_df['syn_count'].to_numpy()

# Create sparse CSC matrix
A_fafb = coo_matrix((data, (rows, cols)),
                    shape=(nodes_index.size, nodes_index.size)).tocsc()
N_fafb = A_fafb.shape[0]

# Define empirical heterogeneity function (same as in synapse_count_distributions.py)
def empirical_heterogeneity(data):
    """Compute empirical heterogeneity: <|x_i - x_j|> / <x>."""
    x = np.sort(data[np.isfinite(data)])
    n = len(x)
    if n < 2:
        return np.nan
    # Efficient mean absolute difference via sorted formula
    i = np.arange(1, n + 1)
    mad = (2.0 / (n * n)) * np.sum((2 * i - n - 1) * x)
    return mad / np.mean(x)

# Prepare region data for ordering (using connection-level brain region)
region_df = conn_df[['syn_count', 'conn_brain_region']].copy()
region_df = region_df.rename(columns={'conn_brain_region': 'brain_region'})
region_df = region_df[region_df['brain_region'] != 'Other Regions']

# Sort regions by empirical heterogeneity (same ordering as synapse_count_distributions.py)
fafb_region_order = (
    region_df
    .groupby('brain_region')['syn_count']
    .apply(lambda x: empirical_heterogeneity(x.values))
    .sort_values()
    .index
)

# Create viridis colormap for regions
fafb_cmap = plt.get_cmap('viridis', len(fafb_region_order))
fafb_region_colors = {r: fafb_cmap(i) for i, r in enumerate(fafb_region_order)}

# Shuffle weights for the full FAFB matrix
A_fafb_rand = network_processing.null_network(A_fafb, scheme='rand_weight', 
                                               conn_type='disc', n_threshold=1)

# Compute robustness for original and shuffled matrices
# Note: compute_robustness returns values only for neurons with in-degree > 1
k_fafb = A_fafb.getnnz(axis=0)
mask_k_gt1 = k_fafb > 1  # Boolean mask for neurons with k > 1
valid_indices = np.where(mask_k_gt1)[0]  # Matrix column indices with k > 1

Q1_fafb = network_processing.compute_robustness(A_fafb, 1., norm)
Q2_fafb = network_processing.compute_robustness(A_fafb_rand, 1., norm)

# Create mapping from matrix index to Q array index (only for valid neurons)
idx_to_q_idx = {col_idx: q_idx for q_idx, col_idx in enumerate(valid_indices)}

# Filter neuron_df to exclude "Other Regions"
neuron_df_filtered = neuron_df[neuron_df['brain_region'] != 'Other Regions']

# Compute fraction with decreased robustness for each brain region
fafb_frac_decreased = {}
for region in fafb_region_order:
    # Get root_ids for neurons in this region
    region_root_ids = neuron_df_filtered[neuron_df_filtered['brain_region'] == region]['root_id'].values
    
    # Map root_ids to matrix column indices
    region_col_indices = id_to_idx.reindex(region_root_ids).dropna().astype(int).values
    
    # Filter to only include neurons with k > 1 (those that have robustness computed)
    valid_region_indices = [idx for idx in region_col_indices if idx in idx_to_q_idx]
    
    if len(valid_region_indices) == 0:
        fafb_frac_decreased[region] = np.nan
        continue
    
    # Get Q array indices for these neurons
    q_indices = [idx_to_q_idx[idx] for idx in valid_region_indices]
    
    # Extract robustness values for this region
    Q1_region = Q1_fafb[q_indices]
    Q2_region = Q2_fafb[q_indices]
    
    # Compute fraction with decreased robustness
    fafb_frac_decreased[region] = np.mean(Q1_region > Q2_region)

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
ax.set_xticklabels(fafb_bar_labels, rotation=45, ha='right')
ax.set_ylim(0., 1.)
ax.set_xlim(-0.5, len(fafb_bar_labels) - 0.5)

# Apply margins
plt.subplots_adjust(**fig_bar_margins_region)

# Save plot without labels
plt.savefig("../../paper_figures/robustness_comparison/fafb_region_robustness_decrease.svg", dpi=600)

# Add labels
ax.set_ylabel('Fraction with decreased robustness')

plt.show()
