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

# Nature figure size
width = 3.5
height = 3.2

data_dir = '../../raw_data/'
processed_dir = '../processed_data/'

# Import neuron data
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')

#------------------------------------------------------------------------------
# ROBUSTNESS OF SPECIFIC NEURONS
#------------------------------------------------------------------------------
k_min = 10

neuron_df = neuron_df[neuron_df['in_deg'] >= k_min]

# Map of type labels to cell classes
type_map ={
    'R1-6': 'R cells',
    'EPG': 'EP-G cells',
    'EPGt': 'EP-G cells',
    'ER3d': 'Ring neurons',
    'ER3a': 'Ring neurons',
    'ER1': 'Ring neurons',
    'ER4d': 'Ring neurons',
    'ER5': 'Ring neurons',
    'KCapbp-ap2': 'Kenyon cells',
    'KCg-m': 'Kenyon cells',
    'KCab': 'Kenyon cells',
    'KCg-d': 'Kenyon cells',
    'KCab-p': 'Kenyon cells',
    }

classes = ['R cells', 'Ring neurons', 'EP-G cells', 'Kenyon cells']

# Append cell class to neuron data
neuron_df['class'] = neuron_df['primary_type'].map(type_map)
neuron_df = neuron_df.dropna()

# Get data for violin plots
violin_data = [neuron_df.loc[neuron_df["class"] == c, "norm_robustness"].to_numpy(dtype=float) for c in classes]

# FIGURE: VIOLIN PLOT OF ROBUSTNESS FOR DIFFERENT CLASSES----------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

parts = ax.violinplot(
    violin_data,
    showmeans=False,
    showmedians=True,
    showextrema=False,
    widths=0.9
)

# Color each violin body to match points
for i, body in enumerate(parts['bodies']):
    color = con_colors[i]
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.6)

# Style median line
if 'cmedians' in parts and parts['cmedians'] is not None:
    parts['cmedians'].set_linewidth(2.5)
    parts['cmedians'].set_color('black')
    
# Labels/ticks
ax.set_ylabel("Normalized robustness")
ax.set_xticks(range(1, len(classes) + 1))
ax.set_xticklabels(classes, rotation=35, ha="right")

plt.show()






