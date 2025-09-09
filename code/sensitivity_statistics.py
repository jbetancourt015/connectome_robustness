"""
    This script calculates statistics of single-neuron sensitivities
-------------------------------------------------------------------------------
created on:
    Tue 18 Feb 2025
-------------------------------------------------------------------------------
last change:
    Tue 2 Sep 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import network_functions
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from time import time
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 3.5
height = 3.2

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

data_idx = 5
thresholded = True

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

# Get connectome
start = time()
A = network_functions.load_connectome(data_idx, thresholded=thresholded)
print('Loaded connectome, time:', time()-start)
N = A.shape[0]

# Get null networks
start = time()
A_rand = network_functions.null_network(A, scheme='rand_weight', conn_type='cont' if data_idx==4 else 'disc', thresholded=thresholded if data_idx==5 else False)
print('Random weight network generated, time:', time()-start)

start = time()
A_poisson = network_functions.null_network(A, scheme='poisson', conn_type='cont' if data_idx==4 else 'disc')
print('Poisson network generated, time:', time()-start)

#------------------------------------------------------------------------------
# SENSITIVITY PLOTS
#------------------------------------------------------------------------------
def plot_sensitivities(eta, null_net, normalized=False, labels=True):
    folder = 'figures' if labels else 'raw_figures'
    
    # Get sensitivity of connectomes
    Q1 = network_functions.compute_sensitivity(A, eta, normalized)
    Q2 = network_functions.compute_sensitivity(A_rand if null_net=='rand_weight' else A_poisson, eta, normalized)
    frac_lower = np.mean(Q1 > Q2)
    Q_min, Q_max = min(min(Q1),min(Q2)), max(max(Q1),max(Q2))
    
    # Set up subfigures
    fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
    
    # Plot sensitivities for random weight
    ax_scatter.scatter(Q1, Q2, color=con_colors[data_idx], alpha=.2, rasterized=True)
    ax_scatter.plot([Q_min,Q_max], [Q_min,Q_max], c='k', ls='--', lw=1)
    ax_scatter.set_aspect('equal')
    
    # Add labels if necessary
    if labels:
        ax_scatter.set_xlabel('Connectome sensitivity')
        ax_scatter.set_ylabel('Random weight sensitivity' if null_net=='rand_weight' else 'Poisson sensitivity')
        scheme_label = f"$\eta = {eta}$"
        ax_scatter.text(Q_max, Q_min+0.1*(Q_max-Q_min), scheme_label, ha='right', va='bottom')
        ax_scatter.text(Q_max, Q_min, 'Frac. lower: %s'%("{:.3f}".format(frac_lower)), ha='right', va='bottom')
    
    # Use divider to attach KDE axes to the scatter's box
    divider = make_axes_locatable(ax_scatter)
    ax_kde_x = divider.append_axes("top", size='10%', pad=0.1, sharex=ax_scatter)
    ax_kde_y = divider.append_axes("right", size='10%', pad=0.1, sharey=ax_scatter)
    
    # Plot marginal densities
    sns.kdeplot(x=Q1, ax=ax_kde_x, fill=True, color=con_colors[data_idx], 
                alpha=0.6, clip=(Q_min, Q_max))
    ax_kde_x.axis("off")
    
    sns.kdeplot(y=Q2, ax=ax_kde_y, fill=True, color=con_colors[data_idx], 
                alpha=0.6, clip=(Q_min, Q_max))
    ax_kde_y.axis("off")
    
    plt.savefig(f"../../{folder}/sensitivities/{null_net}_sensitivity_eta_{int(eta)}_{connectomes[data_idx]}.pdf", 
                dpi=600)
    plt.show()


def plot_sensitivity_hist(eta, null_net, normalized=False, labels=True):
    folder = 'figures' if labels else 'raw_figures'
    
    # Get sensitivity of connectomes
    Q1 = network_functions.compute_sensitivity(A, eta, normalized)
    Q2 = network_functions.compute_sensitivity(A_rand if null_net=='rand_weight' else A_poisson, eta, normalized)
    frac_lower = np.mean(Q1 > Q2)
    Q_min, Q_max = min(min(Q1),min(Q2)), max(max(Q1),max(Q2))

    fig, ax_scatter = plt.subplots(figsize=(.9*width,.9*height))
    
    # Create histogram
    xbins = np.linspace(Q_min, Q_max, 40)
    ybins = xbins.copy()
    counts, xedges, yedges, im = ax_scatter.hist2d(
        Q1, Q2,
        bins=[xbins, ybins],
        density=True,
        norm=LogNorm(),
        cmap=fade_to_color_cmap(con_colors[data_idx], alpha_min=0.2, name="fade_to_color")
    )
    ax_scatter.set_aspect('equal')
    
    # Use divider to attach KDE axes to the scatter's box
    divider = make_axes_locatable(ax_scatter)
    ax_kde_x = divider.append_axes('top', size='10%', pad=0.1, sharex=ax_scatter)
    ax_kde_y = divider.append_axes('right', size='10%', pad=0.1, sharey=ax_scatter)
    ax_cbar = divider.append_axes('right', size='5%', pad=0.1)
    
    # KDE plots
    sns.kdeplot(x=Q1, ax=ax_kde_x, fill=True, color=con_colors[data_idx], alpha=0.6,
                clip=(Q_min, Q_max))
    ax_kde_x.axis("off")
    
    sns.kdeplot(y=Q2, ax=ax_kde_y, fill=True, color=con_colors[data_idx], alpha=0.6,
                clip=(Q_min, Q_max))
    ax_kde_y.axis("off")
    
    
    
    # 7) re-enable all four spines on joint
    for loc in ["left","right","top","bottom"]:
        ax_scatter.spines[loc].set_visible(True)
    ax_scatter.set_frame_on(True)
    
    # 8) diagonal & labels & text
    ax_scatter.plot([Q_min, Q_max], [Q_min, Q_max], "--", c="k", lw=1)
    
    # 9) colorbar in its own column
    cb = fig.colorbar(im, cax=ax_cbar)
    
    # Add labels if necessary
    if labels:
        ax_scatter.set_xlabel('Connectome sensitivity')
        ax_scatter.set_ylabel('Random weight sensitivity' if null_net=='rand_weight' else 'Poisson sensitivity')
        scheme_label = f"$\eta = {eta}$"
        ax_scatter.text(Q_max, Q_min+0.1*(Q_max-Q_min), scheme_label, ha='right', va='bottom')
        ax_scatter.text(Q_max, Q_min, 'Frac. lower: %s'%("{:.3f}".format(frac_lower)), ha='right', va='bottom')
        
        cb.set_label("Density")
    
    # 10) tidy up & show
    for ax in (ax_kde_x, ax_kde_y):
        ax.tick_params(left=False, bottom=False)
    plt.savefig(
        f"../../{folder}/sensitivity_hists/{null_net}_sensitivity_{connectomes[data_idx]}.pdf",
        dpi=600
    )
    plt.show()


#------------------------------------------------------------------------------
# COMPARE SENSITIVITIES
#------------------------------------------------------------------------------
# Plot sensitivities
plot_sensitivities(1., 'rand_weight', normalized=True)
plot_sensitivity_hist(1., 'rand_weight', normalized=True)




