"""
    This script scores all neurons in terms of the distance to some peripheral
    set of neurons.
-------------------------------------------------------------------------------
created on:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
last change:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
    Varun:
        name:       Varun Varanasi
        email:      varunvaranasi@g.harvard.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging

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

colormap ='viridis'

# Nature figure size
width = 3.5
height = 3.2

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron and connections data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

#------------------------------------------------------------------------------
# STATISTICS - HOW TO DEFINE THE PERIPHERY?
#------------------------------------------------------------------------------
neuron_df['log_in_deg'] = np.log(neuron_df['in_deg'])

# Get data for violin plot
region_order = (
    neuron_df
    .groupby('brain_region')['log_in_deg']
    .median()               # average Q_mean over neuropils in that region
    .sort_values()        # sort regions by their overall Q
    .index
)

cmap = plt.get_cmap('viridis', len(region_order))
region_colors = {r: cmap(i) for i, r in enumerate(region_order)}

violin_data = [neuron_df.loc[neuron_df["brain_region"] == r, "log_in_deg"].to_numpy(dtype=float) for r in region_order]

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
ax.set_ylabel("(log) In-degree")
ax.set_xticks(range(1, len(region_order) + 1))
ax.set_xticklabels(region_order, rotation=35, ha="right")

plt.show()

#------------------------------------------------------------------------------
# DISTRIBUTION OF IN-DEGREES
#------------------------------------------------------------------------------
# Compute the CDF of in-degree
counts = neuron_df['in_deg'].value_counts().sort_index()
cum_counts = counts.cumsum()
cdf = cum_counts / cum_counts.iloc[-1]

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

mask = cdf.index < 100
ax.step(cdf.index[mask], cdf.values[mask], where='post', lw=2, color=con_colors[0])

# Plot properties
ax.set_xlabel('In-degree')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# DISTRIBUTION OF DISTANCE TO THE PERIPHERY
#------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

sns.kdeplot(peri_df['distance_optic'], ax=ax, color=con_colors[0], label='Optic')
sns.kdeplot(peri_df['distance_olfactory'], ax=ax, color=con_colors[1], label='Olfactory')
sns.kdeplot(peri_df['distance_joint'], ax=ax, color=con_colors[2], label='Joint')

ax.legend()

ax.set_xlabel('Distance to the periphery')
ax.set_ylabel('Density')

plt.show()

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

#------------------------------------------------------------------------------
# COMPARE ROBUSTNESS TO PERIPHERALITY
#------------------------------------------------------------------------------
def percentile_band_df(
    df,
    x_col="distance",
    y_col="robustness",
    bins=30,
    percentiles=(0.05, 0.5, 0.95),
    min_count=5
):
    """
    Return a tidy dataframe with equal-width bins of x_col and percentiles of y_col.
    Columns: bin, x_left, x_right, x_center, n, q05, q50, q95
    """
    d = df[[x_col, y_col]].dropna()

    # Build equal-width edges
    x_min, x_max = d[x_col].min(), d[x_col].max()
    edges = np.linspace(x_min, x_max, bins + 1)

    # Assign bins [0..bins-1]
    bin_idx = pd.cut(d[x_col], bins=edges, include_lowest=True, labels=False)
    d = d.assign(_bin=bin_idx).dropna(subset=["_bin"])
    d["_bin"] = d["_bin"].astype(int)

    # Quantiles per bin → unstack to columns
    q = (
        d.groupby("_bin", observed=True)[y_col]
        .quantile(percentiles)
        .unstack(level=-1)              # columns are the quantiles
    )

    # Rename columns to q05, q50, q95 (keeps arbitrary percentiles too)
    q.columns = [f"q{int(p*100):02d}" for p in q.columns]

    # Counts per bin
    n = d.groupby("_bin", observed=True)[y_col].size().rename("n")

    # Geometry
    centers = (edges[:-1] + edges[1:]) / 2
    out = (
        q.join(n)
         .reset_index()
         .rename(columns={"_bin": "bin"})
    )
    out["x_left"] = out["bin"].map(lambda i: edges[i])
    out["x_right"] = out["bin"].map(lambda i: edges[i+1])
    out["x_center"] = out["bin"].map(lambda i: centers[i])

    # Drop sparse bins (avoid flaky extreme quantiles)
    out = out[out["n"] >= min_count].sort_values("x_center").reset_index(drop=True)

    # Ensure expected columns exist even if some percentiles weren’t requested
    for p in (0.05, 0.5, 0.95):
        col = f"q{int(p*100):02d}"
        if col not in out.columns:
            out[col] = np.nan

    return out

# Append peripherality to neuron characteristics
neuron_df = neuron_df.merge(peri_df, on=("root_id"))

seeds =['optic', 'olfactory', 'joint']

# for seed in seeds:
#     # Get dataframe of summary statistics
#     summary = percentile_band_df(neuron_df, x_col=f"distance_{seed}", y_col='norm_robustness')
    
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(1.9*width, .9*height))
    
#     ax.plot(summary['x_center'], summary['q05'], lw=2, c=con_colors[0], label='90% range')
#     ax.plot(summary['x_center'], summary['q50'], lw=2, c='k', label='Median')
#     ax.plot(summary['x_center'], summary['q95'], lw=2, c=con_colors[0])
#     ax.legend()
    
#     ax.fill_between(summary['x_center'], 
#                     summary['q05'], 
#                     summary['q95'], 
#                     color=con_colors[0], alpha=0.25, linewidth=0)
    
#     ax.set_xlabel('Distance to the periphery')
#     ax.set_ylabel('Normalized robustness')
    
#     plt.show()

# # Repeat plots for big neurons only
# for seed in seeds:
#     # Get dataframe of summary statistics
#     summary = percentile_band_df(neuron_df[neuron_df['in_deg']>=10], x_col=f"distance_{seed}", y_col='norm_robustness')
    
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(1.9*width, .9*height))
    
#     ax.plot(summary['x_center'], summary['q05'], lw=2, c=con_colors[1], label='90% range')
#     ax.plot(summary['x_center'], summary['q50'], lw=2, c='k', label='Median')
#     ax.plot(summary['x_center'], summary['q95'], lw=2, c=con_colors[1])
#     ax.legend()
    
#     ax.fill_between(summary['x_center'], 
#                     summary['q05'], 
#                     summary['q95'], 
#                     color=con_colors[1], alpha=0.25, linewidth=0)
    
#     ax.set_xlabel('Distance to the periphery')
#     ax.set_ylabel('Normalized robustness')
    
#     plt.show()

# #------------------------------------------------------------------------------
# # PLOT CDF OF ROBUSTNESS BY PERIPHERALITY
# #------------------------------------------------------------------------------
# n_bands = 5
# bands = [[i,i+1] for i in range(n_bands)]

# cmap = plt.get_cmap('viridis', n_bands)

# labels = ['Optic', 'Olfactory', 'Joint']

# for seed in seeds:
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
#     # Compute CDF of robustness for each band
#     for i, b in enumerate(bands):
#         mask = (neuron_df[f"distance_{seed}"] >= b[0]) & (neuron_df[f"distance_{seed}"] < b[1])
#         counts = neuron_df[mask]['norm_robustness'].value_counts().sort_index()
#         cum_counts = counts.cumsum()
#         cdf = cum_counts / cum_counts.iloc[-1]
        
#         # Plot CDF
#         ax.step(cdf.index, cdf.values, where='post', lw=2, c=cmap(i), label=f"Dist.$\in$[{b[0]},{b[1]})", zorder=n_bands-i)
    
#     ax.legend()
    
#     ax.set_xscale('log')
#     # ax.set_xlim(0.,3.)
    
#     # Format figure
#     ax.set_xlabel('Normalized robustness')
#     ax.set_ylabel('CDF')

# # Look at big neurons only
# n_bands = 4
# bands = [[i+1,i+2] for i in range(n_bands)]

# for seed in seeds:
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
#     # Compute CDF of robustness for each band
#     for i, b in enumerate(bands):
#         mask = (neuron_df[f"distance_{seed}"] >= b[0]) & (neuron_df[f"distance_{seed}"] < b[1])
#         counts = neuron_df[mask]['norm_robustness'].value_counts().sort_index()
#         cum_counts = counts.cumsum()
#         cdf = cum_counts / cum_counts.iloc[-1]
        
#         # Plot CDF
#         ax.step(cdf.index, cdf.values, where='post', lw=2, c=cmap(i), label=f"Dist.$\in$[{b[0]},{b[1]})", zorder=n_bands-i)
    
#     ax.legend()
    
#     ax.set_xscale('log')
    
#     # Format figure
#     ax.set_xlabel('Normalized robustness')
#     ax.set_ylabel('CDF')
    
# #------------------------------------------------------------------------------
# # PLOT CDF OF ROBUSTNESS BY PERIPHERALITY DECILE
# #------------------------------------------------------------------------------
# n_quantiles = 10
# cmap = plt.get_cmap('viridis', n_quantiles)

# for seed in seeds:
#     neuron_df[f"decile_{seed}"] = pd.qcut(neuron_df[f"distance_{seed}"], q=n_quantiles, labels=False)
    
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
#     # Compute CDF of robustness for each band
#     for i in range(n_quantiles):
#         mask = neuron_df[f"decile_{seed}"] == i
#         counts = neuron_df[mask]['norm_robustness'].value_counts().sort_index()
#         cum_counts = counts.cumsum()
#         cdf = cum_counts / cum_counts.iloc[-1]
        
#         # Plot CDF
#         ax.step(cdf.index, cdf.values, where='post', lw=2, c=cmap(i), label=f"Decile ${i}$", zorder=n_quantiles-i)
    
#     ax.plot([0.55,0.55], [0.,1.], c='k', ls='--', lw=1)
    
#     ax.legend()
    
#     ax.set_xscale('log')
    
#     # Format figure
#     ax.set_xlabel('Normalized robustness')
#     ax.set_ylabel('CDF')
    
# #------------------------------------------------------------------------------
# # PLOT CDF OF ROBUSTNESS BY LOGARITHMIC PERCENTILES
# #------------------------------------------------------------------------------
# # Suppose df has column "x"
# percentiles = [0, 2, 4, 8, 100]
# n_bands = 4
# cmap = plt.get_cmap('PuRd_r', n_bands+1)

# # Convert percentile boundaries to actual x-values
# for seed in seeds:
#     mask0 = neuron_df[f"distance_{seed}"] > 0
#     cuts = np.percentile(neuron_df[mask0][f"distance_{seed}"].dropna(), percentiles)
    
#     # Create readable labels
#     labels = [f"{percentiles[i]}-{percentiles[i+1]}" for i in range(len(percentiles)-1)]
#     neuron_df[f"{seed}_band"] = pd.cut(neuron_df[f"distance_{seed}"], bins=cuts, labels=labels, include_lowest=True)
    
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
#     # Plot CDF of seeds
#     rob_vals, cdf_vals = compute_cdf(neuron_df[~mask0]['norm_robustness'])
#     ax.step(rob_vals, cdf_vals, where='post', lw=2, c='k', ls='--', label='Seed')
    
#     for i in range(n_bands):
#         mask = neuron_df[f"{seed}_band"] == labels[i]
#         rob_vals, cdf_vals = compute_cdf(neuron_df[mask&mask0]['norm_robustness'])
        
#         # Plot CDF
#         ax.step(rob_vals, cdf_vals, where='post', lw=2, c=cmap(i), label=f"Pctiles {labels[i]}", zorder=len(percentiles)-i)
        
#         ax.legend()
        
#         ax.set_xscale('log')
        
#         # Format figure
#         ax.set_xlabel('Normalized robustness')
#         ax.set_ylabel('CDF')
    
# #------------------------------------------------------------------------------
# # PLOT MEDIAN ROBUSTNESS OF EACH PERCENTILE
# #------------------------------------------------------------------------------
# n_quantiles = 30

# for seed in seeds:
#     mask0 = neuron_df[f"distance_{seed}"] > 0
#     masked_df = neuron_df[mask0].copy()
#     masked_df[f"quantile_{seed}"] = pd.qcut(masked_df[f"distance_{seed}"], q=n_quantiles, labels=False)
    
#     # Set up figure
#     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
#     # Compute median of robustness for each band
#     medians = []
#     for i in range(n_quantiles):
#         mask = masked_df[f"quantile_{seed}"] == i
#         medians.append(masked_df[mask]['norm_robustness'].median())
        
#     # Plot medians
#     ax.plot(np.arange(n_quantiles), medians, lw=2, c=con_colors[4])
    
#     # Format figure
#     ax.set_xlabel('Peripherality quantile')
#     ax.set_ylabel('Median robustness')
    
#------------------------------------------------------------------------------
# ANALYSIS - VISUAL SEED
#------------------------------------------------------------------------------
# Suppose df has column "x"
percentiles = np.array([0, 1, 2, 4, 8, 16, 32, 64, 100])
# percentiles = np.arange(0,101)
centers = (percentiles[1:]+percentiles[:-1])/2

# Convert percentile boundaries to actual x-values
seed = 'optic'
mask0 = neuron_df[f"distance_{seed}"] > 0
masked_df = neuron_df[mask0].copy()
cuts = np.percentile(neuron_df[mask0][f"distance_{seed}"].dropna(), percentiles)

# Create readable labels
labels = [f"{percentiles[i]}-{percentiles[i+1]}" for i in range(len(percentiles)-1)]
masked_df[f"{seed}_band"] = pd.cut(neuron_df[f"distance_{seed}"], bins=cuts, labels=labels, include_lowest=True)

# Compute median of robustness for each band
medians = []
for l in labels:
    mask = masked_df[f"{seed}_band"] == l
    medians.append(masked_df[mask]['norm_robustness'].median())

seed_median = neuron_df[~mask0]['norm_robustness'].median()

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))
    
# Plot medians
ax.plot(centers, medians, lw=2, c=con_colors[4])

ax.scatter([centers[0]/2], [seed_median], c='white', edgecolors=con_colors[4])

ax.set_xscale('log')

# Format figure
ax.set_xlabel('Peripherality quantile')
ax.set_ylabel('Median robustness')

plt.show()

#------------------------------------------------------------------------------
# COMPOISITION OF BUCKETS
#------------------------------------------------------------------------------
for l in labels[:5]:
    mask = masked_df[f"{seed}_band"] == l
    print('Bucket:',l)
    print(masked_df[mask]['primary_type'].value_counts()/np.sum(masked_df[mask]['primary_type'].value_counts()))
    