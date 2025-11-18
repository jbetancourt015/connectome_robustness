"""
    This script calculates statistics of single-neuron sensitivities
-------------------------------------------------------------------------------
created on:
    Tue 18 Feb 2025
-------------------------------------------------------------------------------
last change:
    Tue 18 Nov 2025
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
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from time import time
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter
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

# Nature figure size
width = 3.5
height = 3.2

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain',
               'drosophila_banc']

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

#------------------------------------------------------------------------------
# SENSITIVITY PLOTTING FUNCTIONS
#------------------------------------------------------------------------------
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
formatter.set_powerlimits((0, 0))

def log_kde_to_original(q_vals, n_grid=512, bw=None):
    u = np.log10(q_vals[q_vals > 0])
    kde = gaussian_kde(u, bw_method=bw)  # bw=None => Scott’s rule
    umin, umax = u.min(), u.max()
    u_grid = np.linspace(umin, umax, n_grid)
    x_grid = np.power(10.0, u_grid)                  # back to original Q
    f_u    = kde(u_grid)                             # density in log10-space
    f_x    = f_u / (x_grid * np.log(10.0))           # change-of-variables
    return x_grid, f_x

def plot_robustness(data_idx, A, A_rand, eta, null_net, normalized=False, log_axes=False):    
    # Get sensitivity of connectomes
    Q1 = network_functions.compute_robustness(A, eta, normalized)
    Q2 = network_functions.compute_robustness(A_rand, eta, normalized)
    frac_lower = np.mean(Q1 >= Q2)
    Q_min, Q_max = min(min(Q1),min(Q2)), max(max(Q1),max(Q2))
    
    # Set up subfigures
    fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
    
    # Plot sensitivities for random weight
    ax_scatter.scatter(Q1, Q2, color=con_colors[data_idx], alpha=.2, rasterized=True)
    ax_scatter.plot([Q_min,Q_max], [Q_min,Q_max], c='k', ls='--', lw=1)
    ax_scatter.set_aspect('equal')
    
    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Random weight robustness' if null_net=='rand_weight' else 'Poisson sensitivity')
    ax_scatter.text(0.05, 0.95, 'Frac. lower: %s'%("{:.3f}".format(frac_lower)),
                    ha='left',
                    va='top',
                    transform=ax_scatter.transAxes)
    
    if log_axes:
        # Set axis scales
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
        
        # Use divider to attach KDE axes to the scatter's box
        divider = make_axes_locatable(ax_scatter)
        ax_kde_x = divider.append_axes("top", size='10%', pad=0.1, sharex=ax_scatter)
        ax_kde_y = divider.append_axes("right", size='10%', pad=0.1, sharey=ax_scatter)
        
        # X-marginal (top): density vs Q1 on log-x
        x_grid, f_x = log_kde_to_original(Q1)
        ax_kde_x.plot(x_grid, f_x, color=con_colors[data_idx], lw=1.5)
        ax_kde_x.fill_between(x_grid, f_x, alpha=0.6, color=con_colors[data_idx])
        ax_kde_x.set_xscale('log')
        ax_kde_x.set_ylim(0, None)
        ax_kde_x.set_xlim(Q_min, Q_max)
        ax_kde_x.axis("off")
        
        # Y-marginal (right): density vs Q2 on log-y (plotted horizontally)
        y_grid, f_y = log_kde_to_original(Q2)
        ax_kde_y.plot(f_y, y_grid, color=con_colors[data_idx], lw=1.5)
        ax_kde_y.fill_betweenx(y_grid, 0*y_grid, f_y, alpha=0.6, color=con_colors[data_idx])
        ax_kde_y.set_yscale('log')
        ax_kde_y.set_xlim(0, None)
        ax_kde_y.set_ylim(Q_min, Q_max)
        ax_kde_y.axis("off")
    else:
        # Use divider to attach KDE axes to the scatter's box
        divider = make_axes_locatable(ax_scatter)
        ax_kde_x = divider.append_axes("top", size='10%', pad=0.1, sharex=ax_scatter)
        ax_kde_y = divider.append_axes("right", size='10%', pad=0.1, sharey=ax_scatter)
        
        # Plot marginal densities
        sns.kdeplot(x=Q1, ax=ax_kde_x, fill=True, color=con_colors[data_idx], 
                    alpha=0.6, clip=(Q_min, Q_max))
        sns.kdeplot(y=Q2, ax=ax_kde_y, fill=True, color=con_colors[data_idx], 
                    alpha=0.6, clip=(Q_min, Q_max))
    
    
    
    
    ax_kde_x.axis("off")
    ax_kde_y.axis("off")
    
    # plt.savefig(f"../../{folder}/robustness/{null_net}_robustness_eta_{int(eta)}_{connectomes[data_idx]}.pdf", 
    #             dpi=600)
    plt.show()


def plot_robustness_hist(data_idx, A, A_rand, eta, null_net, normalized=False, log_axes=False, labels=True):
    # Get sensitivity of connectomes
    Q1 = network_functions.compute_robustness(A, eta, normalized)
    Q2 = network_functions.compute_robustness(A_rand, eta, normalized)
    frac_lower = np.mean(Q1 >= Q2)
    Q_min, Q_max = min(min(Q1),min(Q2)), max(max(Q1),max(Q2))

    fig, ax_scatter = plt.subplots(figsize=(.9*width,.9*height))
    
    if log_axes:
        # Create histogram
        xbins = np.logspace(np.log10(Q_min), np.log10(Q_max), 40)
        ybins = xbins.copy()
        counts, xedges, yedges, im = ax_scatter.hist2d(
            Q1, Q2,
            bins=[xbins, ybins],
            density=True,
            norm=LogNorm(),
            cmap=fade_to_color_cmap(con_colors[data_idx], alpha_min=0.2, name="fade_to_color")
        )
        
        # Set axis scales
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
        
        # Use divider to attach KDE axes to the scatter's box
        divider = make_axes_locatable(ax_scatter)
        ax_kde_x = divider.append_axes('top', size='10%', pad=0.1, sharex=ax_scatter)
        ax_kde_y = divider.append_axes('right', size='10%', pad=0.1, sharey=ax_scatter)
        ax_cbar = divider.append_axes('right', size='5%', pad=0.1)
        
        # X-marginal (top): density vs Q1 on log-x
        x_grid, f_x = log_kde_to_original(Q1)
        ax_kde_x.plot(x_grid, f_x, color=con_colors[data_idx], lw=1.5)
        ax_kde_x.fill_between(x_grid, f_x, alpha=0.6, color=con_colors[data_idx])
        ax_kde_x.set_xscale('log')
        ax_kde_x.set_ylim(0, None)
        ax_kde_x.set_xlim(Q_min, Q_max)
        ax_kde_x.axis("off")
        
        # Y-marginal (right): density vs Q2 on log-y (plotted horizontally)
        y_grid, f_y = log_kde_to_original(Q2)
        ax_kde_y.plot(f_y, y_grid, color=con_colors[data_idx], lw=1.5)
        ax_kde_y.fill_betweenx(y_grid, 0*y_grid, f_y, alpha=0.6, color=con_colors[data_idx])
        ax_kde_y.set_yscale('log')
        ax_kde_y.set_xlim(0, None)
        ax_kde_y.set_ylim(Q_min, Q_max)
        ax_kde_y.axis("off")
        
    else:
        xbins = np.linspace(Q_min, Q_max, 40)
        ybins = xbins.copy()
        counts, xedges, yedges, im = ax_scatter.hist2d(
            Q1, Q2,
            bins=[xbins, ybins],
            density=True,
            norm=LogNorm(),
            cmap=fade_to_color_cmap(con_colors[data_idx], alpha_min=0.2, name="fade_to_color")
        )
        
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
    
    
    ax_scatter.set_aspect('equal')
    
    
    ax_kde_x.axis("off")
    ax_kde_y.axis("off")
    
    # 7) re-enable all four spines on joint
    for loc in ["left","right","top","bottom"]:
        ax_scatter.spines[loc].set_visible(True)
    ax_scatter.set_frame_on(True)
    
    # 8) diagonal & labels & text
    ax_scatter.plot([Q_min, Q_max], [Q_min, Q_max], "--", c="k", lw=1)
    
    # 9) colorbar in its own column
    cb = fig.colorbar(im, cax=ax_cbar)
    
    # Add labels
    ax_scatter.set_xlabel('Connectome robustness')
    ax_scatter.set_ylabel('Random weight robustness')
    ax_scatter.text(0.05, 0.95, 'Frac. lower: %s'%("{:.3f}".format(frac_lower)),
                    ha='left',
                    va='top',
                    transform=ax_scatter.transAxes)
    cb.set_label("Density")
    
    ax_kde_x.axis("off")
    ax_kde_y.axis("off")
    
    # 10) tidy up & show
    for ax in (ax_kde_x, ax_kde_y):
        ax.tick_params(left=False, bottom=False)
    plt.show()

#------------------------------------------------------------------------------
# COMPARE SENSITIVITIES
#------------------------------------------------------------------------------
thresholded = False
scheme = 'rand_weight'
norm = True
log_axes = False

for data_idx in range(len(connectomes)):
    # Get connectome
    A = network_functions.load_connectome(data_idx, thresholded=thresholded if data_idx==5 else False)
    N = A.shape[0]
    
    n_threshold = 1
    if data_idx == 5 and thresholded:
        n_threshold = 5
    elif data_idx == 6:
        n_threshold = 3

    if not((scheme!='rand_weight') and data_idx==4):
        # Get null network
        A_rand = network_functions.null_network(A, scheme=scheme, 
                                                conn_type='cont' if data_idx==4 else 'disc', 
                                                n_threshold=n_threshold)
    
        # Plot sensitivities
        plot_robustness(data_idx, A, A_rand, 1., 'rand_weight', normalized=norm, log_axes=log_axes)
        plot_robustness_hist(data_idx, A, A_rand, 1., 'rand_weight', normalized=norm, log_axes=log_axes)
        
# # Drosophila-specific plots
# weight_scheme = 'clump'

# # Get connectome
# A = network_functions.load_connectome(5, thresholded=thresholded, scheme=weight_scheme)
# N = A.shape[0]


# # Get null network
# A_rand = network_functions.null_network(A, scheme=scheme, 
#                                         conn_type='disc', 
#                                         thresholded=thresholded)

# # Plot sensitivities
# plot_robustness_hist(5, A, A_rand, 1., 'rand_weight', normalized=norm, log_axes=log_axes)



