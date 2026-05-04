"""
    This script generates nine robustness plots for the FlyWire dataset:
    1. CDF of normalized robustness
    2. Violin plots of normalized robustness by brain region
    3. CDF of excitatory vs inhibitory normalized robustness
    4. CDF of normalized robustness by reciprocity decile
    5. Running median of robustness vs optic peripherality (sliding window)
    6. Running median of robustness vs olfactory peripherality (sliding window)
    7. Violin plots by visual pathway (R1-6, L1, Mi9, T4)
    8. Violin plots by olfactory pathway (ORN, PN, KC, MBON)
    9. CDF of normalized robustness by neurotransmitter type
-------------------------------------------------------------------------------
created on:
    Tue 3 Feb 2026
-------------------------------------------------------------------------------
last change:
    Fri 1 May 2026
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
import logging
import os
from scipy.sparse import coo_matrix
from matplotlib.patches import ConnectionPatch
import network_processing
from figure_formatting import apply_style, log10_formatter

apply_style()

# Nature figure size
mm_to_in = 25.4
width = 55./mm_to_in
height = 55./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
wide_fig_margins = dict(left=0.12, right=0.98, bottom=0.22, top=0.95)

# Output directory
fig_dir = "../../figures/drosophila_robustness/raw/"
os.makedirs(fig_dir, exist_ok=True)

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]])/255

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    """Compute CDF from a pandas series."""
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# LOAD DATASETS
#------------------------------------------------------------------------------
data_dir = '../../data/'
processed_dir = '../processed_data/'

# Import neuron and connections data
neuron_df = pd.read_parquet(processed_dir+'neuron_data.parquet')
conn_df = pd.read_parquet(processed_dir+'connections_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

# Add peripherality data to neuron dataframe
neuron_df = neuron_df.merge(peri_df, on="root_id")

#------------------------------------------------------------------------------
# LOAD PRE-SAVED SHUFFLED NORMALIZED ROBUSTNESS
#------------------------------------------------------------------------------
sim_dir = '../simulation_results/'
shuffled_rob_df = pd.read_parquet(
    sim_dir + 'drosophila_whole_brain_shuffled_multinomial.parquet',
    columns=['root_id', 'norm_robustness_shuffled']
)
neuron_df = neuron_df.merge(shuffled_rob_df, on='root_id', how='left')

#------------------------------------------------------------------------------
# PLOT 1: CDF OF NORMALIZED ROBUSTNESS
#------------------------------------------------------------------------------
rob_vals, rob_cdf = compute_cdf(neuron_df['norm_robustness'])
median_rob = neuron_df['norm_robustness'].median()

# Compute CDF for shuffled network (filter out NaN values)
shuffled_valid = neuron_df['norm_robustness_shuffled'].dropna()
rob_vals_shuf, rob_cdf_shuf = compute_cdf(shuffled_valid)
median_rob_shuffled = neuron_df['norm_robustness_shuffled'].median()

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

ax.plot([median_rob, median_rob], [0., 1.], lw=1, c='k', ls='--')
ax.plot([median_rob_shuffled, median_rob_shuffled], [0., 1.], lw=1, c='k', alpha=.5, ls='--')
ax.step(rob_vals, rob_cdf, where='post', lw=2, color=con_colors[0], label='Connectome')
ax.step(rob_vals_shuf, rob_cdf_shuf, where='post', lw=2, color=con_colors[0], alpha=0.5, label='Shuffled')


ax.set_xscale('log')
ax.set_xlim(1e-1, 1e2)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'cdf_normalized_robustness.svg', dpi=600)

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')
ax.legend()

plt.show()

#------------------------------------------------------------------------------
# PLOT 2: VIOLIN PLOTS BY BRAIN REGION (LOG SCALE)
#------------------------------------------------------------------------------
# Drop "Other Regions"
neuron_df_regions = neuron_df[neuron_df['brain_region'] != "Other Regions"].copy()

# Sort regions by median
region_order = (
    neuron_df_regions
    .groupby('brain_region')['norm_robustness']
    .median()
    .sort_values()
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# Transform to log scale (filter out non-positive values)
mask0 = neuron_df_regions['norm_robustness'] > 0
neuron_df_regions["log_robustness"] = np.log10(neuron_df_regions["norm_robustness"])

# Get data for violin plots (using log-transformed data)
violin_data = [neuron_df_regions.loc[mask0 & (neuron_df_regions["brain_region"] == r), "log_robustness"].to_numpy(dtype=float) for r in region_order]

# Set up figure (wider for region labels)
fig, ax = plt.subplots(figsize=(2*width, height))

# Add population median line (behind violins)
ax.axhline(np.log10(median_rob), lw=1, c='k', ls='--', zorder=0)

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body to match points
for i, body in enumerate(parts['bodies']):
    region = region_order[i]
    color = region_colors[region]
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')
    
# Labels/ticks
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")

# Format y-axis with log10 tick labels
ax.set_ylim(-1, 2)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log10_formatter))

plt.subplots_adjust(**wide_fig_margins)
plt.savefig(fig_dir + 'violin_by_brain_region.svg', dpi=600)

ax.set_ylabel("Normalized Robustness")
plt.show()

#------------------------------------------------------------------------------
# PLOT 3: CDF OF EXCITATORY VS INHIBITORY NEURONS
#------------------------------------------------------------------------------
# Get index of all neurons
all_neurons = pd.Index(
    pd.unique(
        conn_df[['pre_root_id', 'post_root_id']].values.ravel('K')
    ),
    name='root_id'
)

# Get composition of outgoing synapses
outgoing = (
    conn_df
    .groupby(['pre_root_id', 'nt_class'], as_index=False)['syn_count']
    .sum()
    .pivot(index='pre_root_id', columns='nt_class', values='syn_count')
)

outgoing = outgoing.reindex(columns=['exc', 'inh', 'mod'], fill_value=0)
outgoing_tot = outgoing.sum(axis=1)

frac_out = pd.DataFrame(index=outgoing.index)
frac_out['frac_out_exc'] = outgoing['exc'] / outgoing_tot
frac_out['frac_out_inh'] = outgoing['inh'] / outgoing_tot
frac_out = frac_out.reindex(all_neurons)

# Create synapse composition dataframe
syn_comp = (
    pd.DataFrame(index=all_neurons)
    .join(frac_out)
    .reset_index()
)

# Set threshold for classification
thresh = 0.6

# Classify neurons as excitatory/inhibitory
syn_comp['is_out_exc'] = syn_comp['frac_out_exc'] >= thresh
syn_comp['is_out_inh'] = syn_comp['frac_out_inh'] >= thresh

def classify_row(row):
    inh = row['is_out_inh']
    exc = row['is_out_exc']

    if not inh and not exc:
        return np.nan
    
    if inh and not exc:
        return 'inh'
    if exc and not inh:
        return 'exc'

syn_comp['out_class'] = syn_comp.apply(classify_row, axis=1)

# Merge with neuron-level data
neuron_df_class = neuron_df.merge(syn_comp, how='outer', on='root_id')

# Get CDFs of inhibitory and excitatory neurons
mask_exc = neuron_df_class['out_class'] == 'exc'
mask_inh = neuron_df_class['out_class'] == 'inh'

rob_exc, cdf_exc = compute_cdf(neuron_df_class[mask_exc]['norm_robustness'])
rob_inh, cdf_inh = compute_cdf(neuron_df_class[mask_inh]['norm_robustness'])

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Add population median line (behind data)
ax.plot([median_rob, median_rob], [0., 1.], lw=1, c='k', ls='--')

ax.step(rob_exc, cdf_exc, where='post', color=con_colors[0], lw=2, label='Excitatory')
ax.step(rob_inh, cdf_inh, where='post', color=con_colors[1], lw=2, label='Inhibitory')

ax.set_xscale('log')
ax.set_xlim(1e-1, 1e2)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'cdf_exc_vs_inh.svg', dpi=600)

ax.legend()

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# PLOT 4: CDF BY RECIPROCITY DECILE
#------------------------------------------------------------------------------
n_quantiles = 10

# Add deciles to dataset
neuron_df['reciprocity_q'] = pd.qcut(neuron_df['reciprocity'], q=n_quantiles, labels=False)

# Pre-compute median reciprocity per decile
reciprocities = [neuron_df[neuron_df['reciprocity_q'] == i]['reciprocity'].median()
                 for i in range(n_quantiles)]
rec_min, rec_max = min(reciprocities), max(reciprocities)

# Normalize median reciprocities to [0, 1] for colormap lookup
rec_norm = [(r - rec_min) / (rec_max - rec_min) for r in reciprocities]

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Add population median line (behind data)
ax.plot([median_rob, median_rob], [0., 1.], lw=1, c='k', ls='--')

cmap = plt.get_cmap('coolwarm')

for i in range(n_quantiles):
    mask = neuron_df['reciprocity_q'] == i
    rob, cdf = compute_cdf(neuron_df[mask]['norm_robustness'])
    ax.step(rob, cdf, where='post', lw=2, c=cmap(rec_norm[i]))

ax.set_xscale('log')
ax.set_xlim(1e-1, 1e2)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'cdf_by_reciprocity_decile.svg', dpi=600)

# Print colorbar range to console
print(f"Colorbar range — min median reciprocity: {rec_min:.4f}, max median reciprocity: {rec_max:.4f}")

# Add colorbar for interactive display (not in saved file)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=rec_min, vmax=rec_max))
sm.set_array([])
plt.colorbar(sm, ax=ax)

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# PLOT 5: RUNNING MEDIAN OF ROBUSTNESS VS OPTIC PERIPHERALITY
#------------------------------------------------------------------------------
# Sliding window parameters for running median
log_band_width = 0.8  # Width of band in log10(percentile) space
log_step_size = 0.1   # Step size for sliding window

# Set up seed and masks
seed = 'optic'
mask_seed = neuron_df[f"distance_{seed}"] == 0      # seed neurons (distance = 0)
mask_pos = neuron_df[f"distance_{seed}"] > 0        # all positive distance neurons

# Compute median robustness for seed
seed_median = neuron_df[mask_seed]['norm_robustness'].median()

# For positive-distance neurons: compute percentile ranks
pos_df = neuron_df[mask_pos].copy()
pos_df['quantile'] = pos_df[f"distance_{seed}"].rank(pct=True)

# Define sliding window centers in log10(percentile) space
# Start just after the minimum percentile, end at 100
min_pct = pos_df['quantile'].min()
log_min = np.log10(max(min_pct, 0.001))  # Avoid log(0)
log_max = np.log10(1)

# Generate window centers
window_centers_log = np.arange(log_min + log_band_width/2, 
                                log_max - log_band_width/2 + log_step_size, 
                                log_step_size)

# Compute running median for each window position
running_x = []
running_median = []

for center_log in window_centers_log:
    # Define band edges in percentile space
    low_pct = 10 ** (center_log - log_band_width / 2)
    high_pct = 10 ** (center_log + log_band_width / 2)
    
    # Find neurons within this band
    band_mask = (pos_df['quantile'] >= low_pct) & (pos_df['quantile'] < high_pct)
    
    if band_mask.sum() > 0:
        # x-coordinate is the geometric mean of band edges (= 10^center)
        running_x.append(10 ** center_log)
        running_median.append(pos_df[band_mask]['norm_robustness'].median())

running_x = np.array(running_x)
running_median = np.array(running_median)

# Set up figure with broken x-axis
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, 
    sharey=True, 
    figsize=(width, height),
    gridspec_kw={'width_ratios': [1, 6], 'wspace': 0.05}
)
ax_right.set_facecolor('none')

# Add population median line (behind data, spans both panels via shared y-axis)
ax_left.set_ylim(-.1,8.)
ax_left.axhline(median_rob, lw=1, c='k', ls='--', zorder=0)
ax_right.axhline(median_rob, lw=1, c='k', ls='--', zorder=0)

# Left panel: seed point at the panel boundary (right edge of ax_left)
ax_left.scatter([0], [seed_median], c='white', edgecolors=con_colors[5], s=40,
                zorder=5, clip_on=False)

# Connect seed point to first running median point
con = ConnectionPatch(
    xyA=(0, seed_median), coordsA=ax_left.transData,
    xyB=(running_x[0], running_median[0]), coordsB=ax_right.transData,
    color=con_colors[5], lw=2, zorder=4, clip_on=False)
ax_left.add_artist(con)
ax_left.set_xlim(-0.5, 0.5)
ax_left.set_xticks([0])
ax_left.set_xticklabels(['0'])

# Right panel: running median line in log-scale
ax_right.plot(running_x, running_median, lw=2, c=con_colors[5])
ax_right.set_xscale('log')
ax_right.set_xlim(right=1)

# Hide the spines between the two axes to show the "break"
ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.tick_params(left=False)  # hide left ticks on right panel

# Add diagonal break marks (scale d for left panel to match visual angle)
d = 0.015  # size of diagonal lines in axes coords (for right panel)
d_left = d * 6  # scale by width ratio (6:1) for left panel

kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, lw=0.8)
ax_left.plot((1 - d_left, 1 + d_left), (-d, +d), **kwargs)  # bottom-right

kwargs.update(transform=ax_right.transAxes)
ax_right.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left

# Break in data line at panel boundary: white mask cuts the line, then diagonal marks on top
ylim_min, ylim_max = -0.1, 8.0
y_axes = (running_median[0] - ylim_min) / (ylim_max - ylim_min)
dy_gap = 0.02
ax_right.plot((-d, +d), (y_axes, y_axes),
              transform=ax_right.transAxes, color='white', clip_on=False, lw=6,
              solid_capstyle='round', zorder=8)
kwargs_break = dict(transform=ax_right.transAxes, color=con_colors[5], clip_on=False, lw=2, zorder=9)
ax_right.plot((-d, +d), (y_axes + dy_gap/2 + d, y_axes + dy_gap/2 - d), **kwargs_break)
ax_right.plot((-d, +d), (y_axes - dy_gap/2 + d, y_axes - dy_gap/2 - d), **kwargs_break)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'robustness_vs_optic_peripherality.svg', dpi=600)

# Format figure
ax_left.set_ylabel('Median robustness')
fig.supxlabel('Peripherality percentile', fontsize=9)

plt.show()

#------------------------------------------------------------------------------
# PLOT 6: RUNNING MEDIAN OF ROBUSTNESS VS OLFACTORY PERIPHERALITY
#------------------------------------------------------------------------------
# Set up seed and masks
seed = 'olfactory'
mask_seed = neuron_df[f"distance_{seed}"] == 0      # seed neurons (distance = 0)
mask_pos = neuron_df[f"distance_{seed}"] > 0        # all positive distance neurons

# Compute median robustness for seed
seed_median = neuron_df[mask_seed]['norm_robustness'].median()

# For positive-distance neurons: compute quantile ranks
pos_df = neuron_df[mask_pos].copy()
pos_df['quantile'] = pos_df[f"distance_{seed}"].rank(pct=True)

# Define sliding window centers in log10(quantile) space
# Start just after the minimum quantile, end at 100
min_pct = pos_df['quantile'].min()
log_min = np.log10(max(min_pct, 0.001))  # Avoid log(0)
log_max = np.log10(1)

# Generate window centers
window_centers_log = np.arange(log_min + log_band_width/2, 
                                log_max - log_band_width/2 + log_step_size, 
                                log_step_size)

# Compute running median for each window position
running_x = []
running_median = []

for center_log in window_centers_log:
    # Define band edges in percentile space
    low_pct = 10 ** (center_log - log_band_width / 2)
    high_pct = 10 ** (center_log + log_band_width / 2)
    
    # Find neurons within this band
    band_mask = (pos_df['quantile'] >= low_pct) & (pos_df['quantile'] < high_pct)
    
    if band_mask.sum() > 0:
        # x-coordinate is the geometric mean of band edges (= 10^center)
        running_x.append(10 ** center_log)
        running_median.append(pos_df[band_mask]['norm_robustness'].median())

running_x = np.array(running_x)
running_median = np.array(running_median)

# Set up figure with broken x-axis
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, 
    sharey=True, 
    figsize=(width, height),
    gridspec_kw={'width_ratios': [1, 6], 'wspace': 0.05}
)
ax_right.set_facecolor('none')

# Add population median line (behind data, spans both panels via shared y-axis)
ax_left.set_ylim(-.1,8.)
ax_left.axhline(median_rob, lw=1, c='k', ls='--', zorder=0)
ax_right.axhline(median_rob, lw=1, c='k', ls='--', zorder=0)

# Left panel: seed point at the panel boundary (right edge of ax_left)
ax_left.scatter([0], [seed_median], c='white', edgecolors=con_colors[6], s=40,
                zorder=5, clip_on=False)

# Connect seed point to first running median point
con = ConnectionPatch(
    xyA=(0, seed_median), coordsA=ax_left.transData,
    xyB=(running_x[0], running_median[0]), coordsB=ax_right.transData,
    color=con_colors[6], lw=2, zorder=4, clip_on=False)
ax_left.add_artist(con)
ax_left.set_xlim(-0.5, 0.5)
ax_left.set_xticks([0])
ax_left.set_xticklabels(['0'])

# Right panel: running median line in log-scale
ax_right.plot(running_x, running_median, lw=2, c=con_colors[6])
ax_right.set_xscale('log')
ax_right.set_xlim(right=1)

# Hide the spines between the two axes to show the "break"
ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.tick_params(left=False)  # hide left ticks on right panel

# Add diagonal break marks (scale d for left panel to match visual angle)
d = 0.015  # size of diagonal lines in axes coords (for right panel)
d_left = d * 6  # scale by width ratio (6:1) for left panel

kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, lw=0.8)
ax_left.plot((1 - d_left, 1 + d_left), (-d, +d), **kwargs)  # bottom-right

kwargs.update(transform=ax_right.transAxes)
ax_right.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left

# Break in data line at panel boundary: white mask cuts the line, then diagonal marks on top
ylim_min, ylim_max = -0.1, 8.0
y_axes = (running_median[0] - ylim_min) / (ylim_max - ylim_min)
dy_gap = 0.02
ax_right.plot((-d, +d), (y_axes, y_axes),
              transform=ax_right.transAxes, color='white', clip_on=False, lw=6,
              solid_capstyle='round', zorder=8)
kwargs_break = dict(transform=ax_right.transAxes, color=con_colors[6], clip_on=False, lw=2, zorder=9)
ax_right.plot((-d, +d), (y_axes + dy_gap/2 + d, y_axes + dy_gap/2 - d), **kwargs_break)
ax_right.plot((-d, +d), (y_axes - dy_gap/2 + d, y_axes - dy_gap/2 - d), **kwargs_break)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'robustness_vs_olfactory_peripherality.svg', dpi=600)

# Format figure
ax_left.set_ylabel('Median robustness')
fig.supxlabel('Peripherality percentile', fontsize=9)

plt.show()

#------------------------------------------------------------------------------
# PLOT 7: VIOLIN PLOTS – VISUAL PATHWAY (R1-6, L1, Mi9, T4)
#------------------------------------------------------------------------------
# Define visual pathway steps in biological order (periphery → deeper processing)
visual_pathway_order = ['R1-6', 'L1', 'Mi9', 'T4']

# Assign each neuron to its visual pathway step
visual_masks = {
    'R1-6': neuron_df['primary_type'] == 'R1-6',
    'L1':   neuron_df['primary_type'] == 'L1',
    'Mi9':  neuron_df['primary_type'] == 'Mi9',
    'T4':   neuron_df['primary_type'].str.startswith('T4', na=False),
}

# Prepare log-transformed robustness data for each step
visual_violin_data = []
for step in visual_pathway_order:
    mask = visual_masks[step] & (neuron_df['norm_robustness'] > 0)
    visual_violin_data.append(
        np.log10(neuron_df.loc[mask, 'norm_robustness']).to_numpy(dtype=float)
    )

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Population median line (behind violins)
ax.axhline(np.log10(median_rob), lw=1, c='k', ls='--', zorder=0)

parts = ax.violinplot(
    visual_violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(con_colors[5])
    body.set_edgecolor('black')
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')

# Labels / ticks
ax.set_xticks(range(1, len(visual_pathway_order) + 1))
ax.set_xticklabels(visual_pathway_order)

# Format y-axis with log10 tick labels
ax.set_ylim(-1, 2)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log10_formatter))

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'violin_visual_pathway.svg', dpi=600)

ax.set_ylabel('Normalized Robustness')
plt.show()

#------------------------------------------------------------------------------
# PLOT 8: VIOLIN PLOTS – OLFACTORY PATHWAY (ORN, PN, KC, MBON)
#------------------------------------------------------------------------------
# Define olfactory pathway steps in biological order (periphery → deeper processing)
olfactory_pathway_order = ['ORN', 'PN', 'KC', 'MBON']

# Assign each neuron to its olfactory pathway step
olfactory_masks = {
    'ORN':  neuron_df['primary_type'].str.startswith('ORN', na=False),
    'PN':   neuron_df['class'] == 'ALPN',
    'KC':   neuron_df['primary_type'].str.startswith('KC', na=False),
    'MBON': neuron_df['primary_type'].str.startswith('MBON', na=False),
}

# Prepare log-transformed robustness data for each step
olfactory_violin_data = []
for step in olfactory_pathway_order:
    mask = olfactory_masks[step] & (neuron_df['norm_robustness'] > 0)
    olfactory_violin_data.append(
        np.log10(neuron_df.loc[mask, 'norm_robustness']).to_numpy(dtype=float)
    )

# Set up figure
fig, ax = plt.subplots(figsize=(width, height))

# Population median line (behind violins)
ax.axhline(np.log10(median_rob), lw=1, c='k', ls='--', zorder=0)

parts = ax.violinplot(
    olfactory_violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(con_colors[6])
    body.set_edgecolor('black')
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')

# Labels / ticks
ax.set_xticks(range(1, len(olfactory_pathway_order) + 1))
ax.set_xticklabels(olfactory_pathway_order)

# Format y-axis with log10 tick labels
ax.set_ylim(-1, 2)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log10_formatter))

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'violin_olfactory_pathway.svg', dpi=600)

ax.set_ylabel('Normalized Robustness')
plt.show()

#------------------------------------------------------------------------------
# PLOT 9: CDF BY NEUROTRANSMITTER TYPE
#------------------------------------------------------------------------------
# Restrict to neurons with an assigned nt_type
nt_df = neuron_df.dropna(subset=['nt_type'])

# Order NT types by median normalized robustness
nt_order = (
    nt_df.groupby('nt_type')['norm_robustness']
    .median()
    .sort_values()
    .index
    .tolist()
)

n_nt = len(nt_order)
cmap_nt = plt.get_cmap('managua')
nt_colors = [cmap_nt(i / (n_nt - 1)) for i in range(n_nt)]

fig, ax = plt.subplots(figsize=(width, height))

ax.plot([median_rob, median_rob], [0., 1.], lw=1, c='k', ls='--')

for i, nt in enumerate(nt_order):
    mask = nt_df['nt_type'] == nt
    rob, cdf = compute_cdf(nt_df[mask]['norm_robustness'])
    ax.step(rob, cdf, where='post', lw=2, c=nt_colors[i])

ax.set_xscale('log')
ax.set_xlim(1e-1, 1e2)

plt.subplots_adjust(**fig_margins)
plt.savefig(fig_dir + 'nt_robustness_distribution.svg', dpi=600)

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')
ax.legend(
    [plt.Line2D([0], [0], color=nt_colors[i], lw=2) for i in range(n_nt)],
    nt_order,
    loc='upper left'
)

plt.show()
