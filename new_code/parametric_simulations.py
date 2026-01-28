"""
    This script analyzes simulations of loss for parametric distributions.
-------------------------------------------------------------------------------
created on:
    Tue 16 Dec 2025
-------------------------------------------------------------------------------
last change:
    Tue 20 Jan 2026
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
width = 2.5
height = 2.5

sim_dir = '../simulation_results/'
n_pred = 200

# Define predicted loss
def general_loss(mean, var):
    rob = np.sqrt(mean + var/mean)
    return (1/np.pi)*np.arccos((1.+1./rob**2)**(-1/2))

#------------------------------------------------------------------------------
# WRAPPER FUNCTION FOR PARAMETRIC SIMULATIONS
#------------------------------------------------------------------------------
def plot_parametric_loss(distribution, n_inputs):
    """
    Plot simulated loss for a given parametric distribution.
    
    Creates two plots:
        1. Loss vs mean (linear x-axis), colored by variance
        2. Loss vs variance (log x-axis), colored by mean
    
    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'lognormal', 'lomax')
    n_inputs : int
        Number of inputs used in the simulation
    """
    # Import simulated loss
    df = pd.read_parquet(sim_dir + f"{distribution}_sim_{n_inputs}.parquet")
    
    # Get values of mean and variance
    mean_vals = df['mean'].unique()
    var_vals = df['var'].unique()
    
    # Get predictions
    mean_min, mean_max = np.min(mean_vals), np.max(mean_vals)
    mean_pred = np.linspace(mean_min, mean_max, n_pred)
    
    var_min, var_max = np.min(var_vals), np.max(var_vals)
    var_pred = np.logspace(np.log10(var_min), np.log10(var_max), n_pred)
    
    #--------------------------------------------------------------------------
    # Plot 1: Loss vs mean, colored by variance
    #--------------------------------------------------------------------------
    cmap = plt.get_cmap('plasma', len(var_vals))
    
    fig = plt.figure(figsize=(.9*width, .9*height))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    
    for i, var in enumerate(var_vals):
        mask = df['var'] == var
        
        # Plot loss
        ax.plot(mean_pred, general_loss(mean_pred, var), c=cmap(i), lw=2, label=f"Var$={var:.0f}$", zorder=0)
        ax.scatter(df[mask]['mean'], df[mask]['sim_loss'], c='white', edgecolors=cmap(i), s=10, rasterized=True)
    
    plt.savefig(f"../../paper_figures/framework/{distribution}_simulation_mean.pdf", dpi=600)
    
    ax.legend()
    
    ax.set_xlabel('Mean')
    ax.set_ylabel('Simulated loss')
    
    plt.show()
    
    #--------------------------------------------------------------------------
    # Plot 2: Loss vs variance, colored by mean
    #--------------------------------------------------------------------------
    cmap = plt.get_cmap('plasma', len(mean_vals))
    
    fig = plt.figure(figsize=(.9*width, .9*height))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    
    for i, mean in enumerate(mean_vals):
        mask = df['mean'] == mean
        
        # Plot loss
        ax.plot(var_pred, general_loss(mean, var_pred), c=cmap(i), lw=2, label=f"Mean$={mean}$", zorder=0)
        ax.scatter(df[mask]['var'], df[mask]['sim_loss'], c='white', edgecolors=cmap(i), s=10, rasterized=True)
    
    ax.set_xscale('log')
    
    plt.savefig(f"../../paper_figures/framework/{distribution}_simulation_var.pdf", dpi=600)
    
    ax.legend()
        
    ax.set_xlabel('Variance')
    ax.set_ylabel('Simulated loss')
    
    plt.show()

#------------------------------------------------------------------------------
# RUN SIMULATIONS
#------------------------------------------------------------------------------
plot_parametric_loss('lognormal', 1000)
plot_parametric_loss('lomax', 1000)
plot_parametric_loss('gamma', 100)





