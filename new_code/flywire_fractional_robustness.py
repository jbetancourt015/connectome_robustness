"""
    This script computes the fractional robustness for excitatory and 
    inhibitory synapses in the FlyWire connectome
-------------------------------------------------------------------------------
created on:
    Mon 10 Nov 2025
-------------------------------------------------------------------------------
last change:
    Mon 10 Nov 2025
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

#------------------------------------------------------------------------------
# COMPUTE FRACTIONAL ROBUSTNESS
#------------------------------------------------------------------------------
# Import neuron data
conn_df = pd.read_csv(data_dir+'flywire_connections.csv.gz', compression='gzip')

# Map of neurotransmitter types
nt_to_class = {
    'ACH': 'exc',
    'GLUT': 'inh',
    'GABA': 'inh',
    'DA':  'mod',
    'SER': 'mod',
    'OCT': 'mod'
}

conn_df['nt_class'] = conn_df['nt_type'].map(nt_to_class)

# Determine connection type by a majority rule
grouped = (
    conn_df.groupby(['pre_root_id', 'post_root_id'])
    .apply(lambda g: pd.Series({
        # total number of synapses for that connection
        'total_syn_count': g['syn_count'].sum(),
        # majority neurotransmitter class
        'nt_class_majority': g['nt_class'].mode().iloc[0]
    }))
    .reset_index()
)

# Compute excitatory and inhibitory robustness
def compute_robustness(syn_counts, nt_classes):
    """
    Example placeholder — replace with your real function.
    Should return a tuple: (exc_robust, inh_robust)
    """
    # Compute moments
    exc_sum = syn_counts[nt_classes == 'exc'].sum()
    exc_sq_sum = (syn_counts[nt_classes == 'exc']**2).sum()
    inh_sum = syn_counts[nt_classes == 'inh'].sum()
    inh_sq_sum = (syn_counts[nt_classes == 'inh']**2).sum()
    # Compute normalized and unnormalized robustness
    if exc_sum > 0:
        exc_rob = np.sqrt((inh_sq_sum/exc_sum) + (exc_sq_sum/exc_sum))
        exc_uni_rob = np.sqrt((inh_sq_sum/exc_sum) + exc_sum/len(syn_counts[nt_classes == 'exc']))
        exc_norm_rob = exc_rob/exc_uni_rob
    else:
        exc_norm_rob = 0
    if inh_sum > 0:
        inh_rob = np.sqrt((exc_sq_sum/inh_sum) + (inh_sq_sum/inh_sum))
        inh_uni_rob = np.sqrt((exc_sq_sum/inh_sum) + inh_sum/len(syn_counts[nt_classes == 'inh']))
        inh_norm_rob = inh_rob/inh_uni_rob
    else:
        inh_norm_rob = 0
    return exc_norm_rob, inh_norm_rob

#------------------------------------------------------------------------------
# PLOT ROBUSTNESS SIDE BY SIDE
#------------------------------------------------------------------------------
# # Reshape to long:
# df_long = (
#     df
#     .melt(id_vars=['class'],
#           value_vars=['robustness_exc', 'robustness_inh'],
#           var_name='nt_kind', value_name='robustness')
# )
# df_long['nt_kind'] = df_long['nt_kind'].map({
#     'robustness_exc': 'Excitatory',
#     'robustness_inh': 'Inhibitory'
# })

# plt.figure(figsize=(10,5))
# ax = sns.violinplot(
#     data=df_long, x='class', y='robustness',
#     hue='nt_kind', split=True, cut=0, inner='quartile', linewidth=1
# )
# ax.legend(title='')
# ax.set_xlabel('Neuron class')
# ax.set_ylabel('Robustness')
# plt.tight_layout()
# plt.show()