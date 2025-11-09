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
import logging

plt.rcParams.update({
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

# ax.set_yscale('log')
# ax.set_xscale('log')

plt.show()









