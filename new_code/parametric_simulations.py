"""
    This script analyzes simulations of loss for parametric distributions.
-------------------------------------------------------------------------------
created on:
    Tue 16 Dec 2025
-------------------------------------------------------------------------------
last change:
    Tue 16 Dec 2025
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

sim_dir = '../simulation_results/'
n_inputs = 1000
n_pred = 200

# Import simulated loss
# poisson_df = pd.read_parquet(sim_dir+f"poisson_sim_{n_inputs}.parquet")
lognormal_df = pd.read_parquet(sim_dir+f"lognormal_sim_{n_inputs}.parquet")

# Define predicted loss
def general_loss(mean, var):
    rob = np.sqrt(mean + var/mean)
    return (1/np.pi)*np.arccos((1.+1./rob**2)**(-1/2))

# #------------------------------------------------------------------------------
# # POISSON SIMULATIONS
# #------------------------------------------------------------------------------
# # Get values of the Poisson rate
# w0_vals = poisson_df['w0'].unique()

# # Define predicted loss
# def poisson_loss(w0):
#     rob = np.sqrt(1.+w0)
#     return (1/np.pi)*np.arccos((1.+1./rob**2)**(-1/2))

# # Get rates for prediction
# w_min, w_max = np.min(w0_vals), np.max(w0_vals)
# w0_pred = np.logspace(np.log10(w_min), np.log10(w_max), n_pred)

# # Plot simulation results and prediction
# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(w0_pred, poisson_loss(w0_pred), c=con_colors[3], lw=2, label='Prediction', zorder=0)
# ax.scatter(poisson_df['w0'], poisson_df['sim_loss'], c='white', edgecolors=con_colors[3], s=10, rasterized=True)

# ax.legend()

# ax.set_xscale('log')
    
# ax.set_xlabel('Average weight $w_0$')
# ax.set_ylabel('Simulated loss')

# plt.show()

#------------------------------------------------------------------------------
# LOGNORMAL SIMULATIONS
#------------------------------------------------------------------------------
# Get values of mu and sigma
mean_vals = lognormal_df['mean'].unique()
var_vals = lognormal_df['var'].unique()

# Get sigmas for prediction
mean_min, mean_max = np.min(mean_vals), np.max(mean_vals)
mean_pred = np.linspace(mean_min, mean_max, n_pred)

# Define colormap for each log-mean
cmap = plt.get_cmap('plasma', len(var_vals))

# Plot simulation results and prediction
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

for i, var in enumerate(var_vals):
    mask = lognormal_df['var'] == var
    
    # Plot loss
    ax.plot(mean_pred, general_loss(mean_pred, var), c=cmap(i), lw=2, label=f"Var.$={var}$", zorder=0)
    ax.scatter(lognormal_df[mask]['mean'], lognormal_df[mask]['sim_loss'], c='white', edgecolors=cmap(i), s=10, rasterized=True)

ax.legend()
    
ax.set_xlabel('Mean weight')
ax.set_ylabel('Simulated loss')

plt.show()

#------------------------------------------------------------------------------
# LOMAX SIMULATIONS
#------------------------------------------------------------------------------





