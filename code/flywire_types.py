"""
    This script computes the statistics of connections between neurons by type
-------------------------------------------------------------------------------
created on:
    Wed 6 Aug 2025
-------------------------------------------------------------------------------
last change:
    Mon 11 Aug 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
# Exploration of FlyWire dataset
import numpy as np
import pandas as pd
import network_functions
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "font.serif": ["Garamond"],
    "text.latex.preamble": r'\usepackage{amsfonts}'
})

processed_dir = '../processed_data/'

connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

data_idx = 5

# Load connectome
A = network_functions.load_connectome(data_idx)

#------------------------------------------------------------------------------
# DISTIRBUTION OF SAMPLED WEIGHTS
#------------------------------------------------------------------------------
def empirical_hist(data):
    # Initialize bins
    bins = np.arange(int(np.ceil(np.nanmax(data))))
    s_unique = np.zeros(len(bins)-1)
    Ps = np.zeros(len(bins)-1)
    # Fill arrays
    for i in range(len(bins)-1):
        inds_bin = (data > bins[i])*(data <= bins[i+1])
        if np.sum(inds_bin) > 0:
            s_unique[i] = np.mean(data[inds_bin])
            Ps[i] = np.sum(inds_bin)/len(data)
    # Remove absent events
    s_unique = s_unique[Ps > 0];
    Ps = Ps[Ps > 0];
    return s_unique, Ps

#------------------------------------------------------------------------------
# FILES AND DIRECTORIES
#------------------------------------------------------------------------------
data_dir = '../../raw_data/'
processed_dir = '../../processed_data/'

conn_file = 'flywire_connections.csv.gz'
types_file = 'flywire_consolidated_cell_types.csv.gz'

#------------------------------------------------------------------------------
# ADD NEURON TYPES
#------------------------------------------------------------------------------
# Load dataset into pandas DataFrame
conn_df = pd.read_csv(data_dir + conn_file, compression='gzip')
types_df = pd.read_csv(data_dir + types_file, compression='gzip')

# Aggregate connections over neuropils
conn_df = conn_df.groupby(['pre_root_id','post_root_id'])['syn_count'].sum().reset_index()
conn_df = conn_df[conn_df['syn_count'] > 5]

# Add types to connections DataFrame
type_map = types_df.set_index('root_id')['primary_type']
conn_df['pre_type'] = conn_df['pre_root_id'].map(type_map)
conn_df['post_type'] = conn_df['post_root_id'].map(type_map)

#------------------------------------------------------------------------------
# FILTER MOST COMMON NEURONS TO FIND CONNECTION TABLE
#------------------------------------------------------------------------------
# Get mean and std dev per connection type
stats_df = (
    conn_df
    .groupby(['pre_type','post_type'])
    .agg(
        strength_mean = ('syn_count','mean'),
        strength_var = ('syn_count', 'var'),
        n_conn  = ('syn_count','size')
        )
    .reset_index()
)

# Select observations with a minimum number of connections
stats_df = stats_df[stats_df['n_conn'] > 10]


plt.scatter(stats_df['strength_mean'], stats_df['strength_var'], color=con_colors[0], rasterized=True)
# Plot plausible eta value
eta = 2
w_min, w_max = 20, 100
v_min = 1
v_max = v_min*(w_max/w_min)**eta
plt.plot([w_min,w_max],[v_min,v_max],lw=1,c='k',ls='--')
plt.text((w_min*w_max)**.5, 0.7*(v_min*v_max)**.5, '$\eta = %s$'%(eta), horizontalalignment='left', verticalalignment='top')
# Labels
plt.xlabel('Mean connection strength')
plt.ylabel('Variance in connection strength')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.savefig('../figures/flywire/var_mean_strength_neuron_types.pdf', dpi=600, bbox_inches='tight')
plt.show()


#------------------------------------------------------------------------------
# COMPUTE MEAN AND STD OF LOG Y AXIS
#------------------------------------------------------------------------------
def bin_by_log_mean(df, mean_col, var_col, nbins=30, min_per_bin=5):
    """Bin by log10(mean_col) uniformly, then compute stats of log10(var_col)."""
    # Keep only positive, finite values
    m = df[mean_col].to_numpy()
    v = df[var_col].to_numpy()
    mask = np.isfinite(m) & np.isfinite(v) & (m > 0) & (v > 0)
    logm = np.log10(m[mask])
    logv = np.log10(v[mask])

    tmp = pd.DataFrame({'logm': logm, 'logv': logv})
    tmp['bin'] = pd.cut(tmp['logm'], bins=nbins)

    binned = (
        tmp.groupby('bin', observed=True)
           .agg(
               log_mean_center=('logm', 'mean'),
               log_var_mean=('logv', 'mean'),
               log_var_std =('logv', 'std'),
               n=('logv', 'size')
           )
           .dropna(subset=['log_var_std'])  # bins with <2 points will have NaN std
           .reset_index(drop=True)
    )

    # keep only bins with enough samples
    binned = binned[binned['n'] >= min_per_bin].copy()
    binned['mean_center'] = 10**binned['log_mean_center']  # back to linear x
    return binned

def plot_binned_logvar(binned, eta=2, ax=None):
    """Plot mean ±1 s.d. of log10(variance) vs mean (x log-scaled)."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(
        binned['mean_center'], binned['log_var_mean'],
        yerr=binned['log_var_std'], fmt='o', ms=4, capsize=2, lw=1, color = con_colors[0]
    )
    ax.set_xscale('log')
    ax.set_xlabel('Mean connection strength')
    ax.set_ylabel('Variance in connection strength (log)')

    # Optional η reference line (var ∝ mean^η => log10(var) = η log10(mean) + c)
    if eta is not None and len(binned) > 0:
        xm = np.log10(binned['mean_center'].to_numpy())
        ym = binned['log_var_mean'].to_numpy()
        c = np.median(ym - eta * xm)  # robust intercept so the line passes near the data
        xline = np.logspace(xm.min(), xm.max(), 200)
        yline = eta * np.log10(xline) + c
        ax.plot(xline, yline, c='k', ls='--', lw=1)
        ax.text(xline[len(xline)//3], yline[len(yline)//3]*1.5, rf'$\eta={eta}$')

    return ax

# Plot mean and std on y axis
binned = bin_by_log_mean(stats_df, mean_col='strength_mean', var_col='strength_var',
                         nbins=25, min_per_bin=8)
plot_binned_logvar(binned, eta=2)
plt.savefig('../figures/flywire/binned_var_mean_strength_neuron_types.pdf', dpi=600, bbox_inches='tight')
plt.show()





