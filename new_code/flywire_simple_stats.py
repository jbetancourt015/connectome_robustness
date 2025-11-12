"""
    This script plots simple statistics of the FlyWire dataset.
-------------------------------------------------------------------------------
created on:
    Tue 11 Nov 2025
-------------------------------------------------------------------------------
last change:
    Tue 11 Nov 2025
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
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')

#------------------------------------------------------------------------------
# RECIPROCAL PAIRS
#------------------------------------------------------------------------------
# Create a reciprocal dataframe
reciprocal_df = conn_df.merge(
    conn_df,
    left_on=["pre_root_id", "post_root_id"],
    right_on=["post_root_id", "pre_root_id"],
    suffixes=("_ab", "_ba"),
    how="inner"
)

reciprocal_pairs = reciprocal_df.drop_duplicates(
    subset=["pre_root_id_ab", "post_root_id_ab"]
)
n_reciprocal = len(reciprocal_pairs)

# Ratio between max and min of reciprocal connections
reciprocal_df['w_max'] = reciprocal_df[['syn_count_ab', 'syn_count_ba']].max(axis=1)
reciprocal_df['w_min'] = reciprocal_df[['syn_count_ab', 'syn_count_ba']].min(axis=1)

reciprocal_df['ratio'] = reciprocal_df['w_max']/reciprocal_df['w_min']

# Statistics
print('Mean ratio:', reciprocal_df['ratio'].mean())
print('Std dev ratio:', reciprocal_df['ratio'].std())



