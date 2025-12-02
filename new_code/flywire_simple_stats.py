"""
    This script plots simple statistics of the FlyWire dataset.
-------------------------------------------------------------------------------
created on:
    Tue 11 Nov 2025
-------------------------------------------------------------------------------
last change:
    Fri 14 Nov 2025
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
import re

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
conn_df = pd.read_parquet(data_dir+'connections_data.parquet')
neuron_df = pd.read_parquet(data_dir+'neuron_data.parquet')
peri_df = pd.read_parquet(data_dir+'periphery_data.parquet')

# Add peripherality data
neuron_df = neuron_df.merge(peri_df, on=("root_id"))

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def compute_cdf(series):
    counts = series.value_counts().sort_index()
    cum_counts = counts.cumsum()
    cdf = cum_counts / cum_counts.iloc[-1]
    return cdf.index, cdf.values

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

#------------------------------------------------------------------------------
# OVERLAP STATISTICS (RECIPROCITY)
#------------------------------------------------------------------------------
# FIGURE: Distribution of reciprocity------------------------------------------
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))

ax.hist(neuron_df['reciprocity'], bins=20, density=True, histtype='step', color=con_colors[0])

ax.set_xlabel('Reciprocity')
ax.set_ylabel('Density')

plt.show()

# FIGURE: Robustness by reciprocity quartile-----------------------------------
n_quantiles = 10

# Add quartile to dataset
neuron_df['reciprocity_q'] = pd.qcut(neuron_df['reciprocity'], q=n_quantiles, labels=False)

# Plot robustness CDF by quartile
fig, ax = plt.subplots(figsize=(1.9*width, .9*height))
cmap = plt.get_cmap('viridis', n_quantiles)

for i in range(n_quantiles):
    mask = neuron_df['reciprocity_q'] == i
    rob, cdf = compute_cdf(neuron_df[mask]['norm_robustness'])
    
    # Plot CDF
    ax.step(rob, cdf, where='post', lw=2, c=cmap(i), label=f"Reciprocity D{i+1}")
    
ax.legend()

ax.set_xscale('log')
    
ax.set_xlabel('Normalized robustness')
ax.set_ylabel('CDF')

plt.show()

#------------------------------------------------------------------------------
# STATISTICS OF SPECIFIC NEURONS
#------------------------------------------------------------------------------
# --- 1. Define neuron group patterns ---

patterns = {
    "LCs"     : re.compile(r"^LC", re.IGNORECASE),
    "LPLCs"   : re.compile(r"^LPLC", re.IGNORECASE),
    "LLPCs"   : re.compile(r"^LLPC", re.IGNORECASE),
    "LPCs"    : re.compile(r"^LPC", re.IGNORECASE),
    "LT cells": re.compile(r"^LT", re.IGNORECASE),
    "T4"     : re.compile(r"^T4", re.IGNORECASE),
    "T5"   : re.compile(r"^T5", re.IGNORECASE),
    "Mi1"   : re.compile(r"^Mi1", re.IGNORECASE),
    "Mi4"    : re.compile(r"^Mi4", re.IGNORECASE),
    "CT1": re.compile(r"^CT1", re.IGNORECASE),
    "PNs": re.compile(r"^PN", re.IGNORECASE),
    "LNs": re.compile(r"^LN", re.IGNORECASE),
    "Kenyon cells": re.compile(r"^KC", re.IGNORECASE),
    "MBONs": re.compile(r"^MBON", re.IGNORECASE),
    "DANs": re.compile(r"^DAN", re.IGNORECASE),
    "R cells": re.compile(r"^ER", re.IGNORECASE),
    "EPG cells": re.compile(r"^EPG", re.IGNORECASE),
    "PFNp cells": re.compile(r"^PFNp", re.IGNORECASE),
    "FB cells": re.compile(r"^FB", re.IGNORECASE),
    "hdelta": re.compile(r"^hdelta", re.IGNORECASE),
    "vdelta": re.compile(r"^vdelta", re.IGNORECASE),
    "PEN1/PEN2": re.compile(r"^PEN", re.IGNORECASE),
    "DNs": re.compile(r"^DN", re.IGNORECASE),
}


# --- 2. Which variables to summarize ---

vars_to_compute = [
    "in_deg",
    "out_deg",
    "norm_robustness",
    "distance_joint",
    "reciprocity",
]

# --- 3. Function to extract rows by regex ---

def filter_by_regex(df, col, regex):
    return df[df[col].str.contains(regex)]


# --- 4. Function to compute "mean (std)" ---

def mean_std_str(series):
    if series.empty:
        return ""
    m = series.mean()
    s = series.std()
    return f"{m:.2f} ({s:.2f})"


# --- 5. Build the table row by row ---

rows = []

for label, regex in patterns.items():
    if label == 'DANs':
        subset = filter_by_regex(neuron_df, "class", regex)
    else:
        subset = filter_by_regex(neuron_df, "primary_type", regex)

    row = {
        "Neuron": label,
        "n_neurons": len(subset)
    }
    for var in vars_to_compute:
        row[var] = mean_std_str(subset[var])
    rows.append(row)

summary_df = pd.DataFrame(rows)

# Get names of neurons obtained by regex pattern
unique_rows = []

for label, regex in patterns.items():
    if label == 'DANs':
        subset = filter_by_regex(neuron_df, "class", regex)
        
        # Count occurrences of each neuron name
        counts = (
            subset["primary_type"]
            .value_counts()        # counts descending by default
            .rename_axis("Name")
            .reset_index(name="Count")
        )
        
    else:
        subset = filter_by_regex(neuron_df, "primary_type", regex)

        # Count occurrences of each neuron name
        counts = (
            subset["primary_type"]
            .value_counts()        # counts descending by default
            .rename_axis("Name")
            .reset_index(name="Count")
        )

    unique_rows.append({
        "Neuron": label,
        "n_neurons": len(counts),
        "Names": counts["Name"].tolist(),   # ordered by count
        "Counts": counts["Count"].tolist()  # matching order
    })

unique_names_df = pd.DataFrame(unique_rows)

#------------------------------------------------------------------------------
# STATISTICS OF SPECIFIC NEURONS
#------------------------------------------------------------------------------
class_rows = []

for neuron_class in neuron_df['class'].unique():
    mask = neuron_df['class'] == neuron_class
    
    row = {
        "Neuron": neuron_class,
        "n_neurons": np.sum(mask)
    }
    for var in vars_to_compute:
        row[var] = mean_std_str(neuron_df[mask][var])
    class_rows.append(row)

class_stats = pd.DataFrame(class_rows)






