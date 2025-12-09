"""
    This script plots statistics of the robustness of neurons in the FlyWire
    connectome.
-------------------------------------------------------------------------------
created on:
    Tue 4 Nov 2025
-------------------------------------------------------------------------------
last change:
    Tue 4 Nov 2025
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
import logging
import seaborn as sns

plt.rcParams.update({
    'text.usetex': False,  # keep LaTeX off globally
    'mathtext.fontset': 'cm',   # or 'stixsans' for sans-serif
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,   # thin axis lines for publication
    'pdf.fonttype': 42,      # ensures text stays as text in Illustrator
    'ps.fonttype': 42,       # same for PostScript
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)


con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

# Nature figure size
width = 3.5
height = 3.2

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')

raw_df = pd.read_csv(data_dir+'flywire_consolidated_cell_types.csv.gz', compression='gzip')

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# OVERALL ROBUSTNESS DISTRIBUTION
#------------------------------------------------------------------------------
# Compute distribution of robustness
rob_cdf_index, rob_cdf_values = compute_cdf(neuron_df['norm_robustness'])

# FIGURE: distribution of normalized robustness across the whole brain---------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.step(rob_cdf_index, 1.-rob_cdf_values, where='post', lw=1, c=con_colors[0])

# ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('Survival function (1-CDF)')

plt.show()

#------------------------------------------------------------------------------
# ROBUSTNESS BY BRAIN REGION
#------------------------------------------------------------------------------
# Sort regions by median
region_order = (
    neuron_df
    .groupby('brain_region')['norm_robustness']
    .median()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

# FIGURE: distribution of normalized robustness by brain region----------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

# Get robustness distribution by region
for r in region_order:
    # Compute distribution of robustness
    mask = neuron_df['brain_region']==r
    rob_cdf_index, rob_cdf_values = compute_cdf(neuron_df[mask]['norm_robustness'])
    ax.step(rob_cdf_index[:-1], 1.-rob_cdf_values[:-1], where='pre', lw=1, c=region_colors[r], label=r)

ax.legend()

ax.set_yscale('log')

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('Survival function (1-CDF)')

plt.show()

# FIGURE: distribution of normalized robustness by brain region----------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

# Get robustness distribution by region
for r in region_order:
    # Compute distribution of robustness
    mask = neuron_df['brain_region']==r
    rob_cdf_index, rob_cdf_values = compute_cdf(neuron_df[mask]['norm_robustness'])
    ax.step(rob_cdf_index[:-1], rob_cdf_values[:-1], where='pre', lw=1, c=region_colors[r], label=r)

ax.legend()

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()

# FIGURE: pdf of normalized robustness by brain region----------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

# Get robustness distribution by region
for r in region_order:
    # Compute distribution of robustness
    mask = neuron_df['brain_region']==r
    sns.kdeplot(neuron_df[mask]['norm_robustness'], ax=ax, color=region_colors[r], label=r)

ax.legend()

ax.set_xlabel('Normalized robustness')
ax.set_ylabel('Density')

plt.show()

# FIGURE: violin plot of normalized robustness by brain region----------------
violin_data = [neuron_df.loc[neuron_df["brain_region"] == r, "norm_robustness"].to_numpy(dtype=float) for r in region_order]

# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

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
ax.set_ylabel("Normalized Robustness")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")

# Add labels
plt.show()

#------------------------------------------------------------------------------
# (LOG) ROBUSTNESS BY BRAIN REGION
#------------------------------------------------------------------------------
# FIGURE: violin plot with log robustness----------------
# Make a transformed column
mask0 = neuron_df['norm_robustness'] > 0
neuron_df["log_robustness"] = np.log10(neuron_df["norm_robustness"])

violin_data = [neuron_df.loc[mask0&(neuron_df["brain_region"] == r), "log_robustness"].to_numpy(dtype=float) for r in region_order]

# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

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
ax.set_ylabel("(log) Normalized Robustness")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")

# Add labels
plt.show()

# FIGURE: pdf of normalized robustness by brain region----------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

# Get robustness distribution by region
for r in region_order:
    # Compute distribution of robustness
    mask = neuron_df['brain_region']==r
    sns.kdeplot(neuron_df[mask0&mask]['log_robustness'], ax=ax, color=region_colors[r], label=r)

ax.legend()

ax.set_xlabel('(log) Normalized robustness')
ax.set_ylabel('Density')

plt.show()

#------------------------------------------------------------------------------
# ROBUSTNESS OF SPECIFIC NEURONS
#------------------------------------------------------------------------------
# k_min = 10

# neuron_df = neuron_df[neuron_df['in_deg'] >= k_min]

# # Map of type labels to cell classes
# type_map ={
#     'R1-6': 'R cells',
#     'EPG': 'EP-G cells',
#     'EPGt': 'EP-G cells',
#     'ER3d': 'Ring neurons',
#     'ER3a': 'Ring neurons',
#     'ER1': 'Ring neurons',
#     'ER4d': 'Ring neurons',
#     'ER5': 'Ring neurons',
#     'KCapbp-ap2': 'Kenyon cells',
#     'KCg-m': 'Kenyon cells',
#     'KCab': 'Kenyon cells',
#     'KCg-d': 'Kenyon cells',
#     'KCab-p': 'Kenyon cells',
#     }

# classes = ['R cells', 'Ring neurons', 'EP-G cells', 'Kenyon cells']

# # Append cell class to neuron data
# neuron_df['class'] = neuron_df['primary_type'].map(type_map)
# neuron_df = neuron_df.dropna()

# # Get data for violin plots
# violin_data = [neuron_df.loc[neuron_df["class"] == c, "norm_robustness"].to_numpy(dtype=float) for c in classes]

# # FIGURE: VIOLIN PLOT OF ROBUSTNESS FOR DIFFERENT CLASSES----------------------
# # Set up figure
# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# parts = ax.violinplot(
#     violin_data,
#     showmeans=False,
#     showmedians=True,
#     showextrema=False,
#     widths=0.9
# )

# # Color each violin body to match points
# for i, body in enumerate(parts['bodies']):
#     color = con_colors[i]
#     body.set_facecolor(color)
#     body.set_edgecolor("black")
#     body.set_alpha(0.6)

# # Style median line
# if 'cmedians' in parts and parts['cmedians'] is not None:
#     parts['cmedians'].set_linewidth(2.5)
#     parts['cmedians'].set_color('black')
    
# # Labels/ticks
# ax.set_ylabel("Normalized robustness")
# ax.set_xticks(range(1, len(classes) + 1))
# ax.set_xticklabels(classes, rotation=35, ha="right")

# plt.show()






