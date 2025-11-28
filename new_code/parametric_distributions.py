"""
    This script plots the robustness of different parametric distributions.
-------------------------------------------------------------------------------
created on:
    Fri 21 Nov 2025
-------------------------------------------------------------------------------
last change:
    Fri 21 Nov 2025
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
import seaborn as sns
import logging

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

colormap ='viridis'

# Nature figure size
width = 3.5
height = 3.2

#------------------------------------------------------------------------------
# ROBUSTNESS FUNCTIONS
#------------------------------------------------------------------------------
def rob_dirac(w0):
    return (w0)**(1/2)

def rob_gamma(alpha, w0):
    return (1+alpha*w0)*(1/2)

def rob_poisson(w0):
    return (1+w0)**(1/2)

def rob_pareto(gamma):
    return np.sqrt((gamma-2)/(gamma-3))

def rob_lognormal(mu,sigma):
    return np.exp((2*mu+3*(sigma**2))/4.)

#------------------------------------------------------------------------------
# DIFFERENT ROBUSTNESS PLOTS
#------------------------------------------------------------------------------
w0_vals = 10**(np.linspace(0.,2.,100))


# Gamma distribution
alpha_vals = [.2,.5,1.,2.,5.]
cmap = plt.get_cmap('viridis', len(alpha_vals))

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

for i, alpha in enumerate(alpha_vals):
    ax.plot(w0_vals, rob_gamma(alpha,w0_vals), c=cmap(i), label=f"$\\alpha={alpha}$")
    
ax.legend()    

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Shape parameter $w_0$')
ax.set_ylabel('Robustness')

plt.show()

# Poisson distribution
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.plot(w0_vals, rob_poisson(w0_vals), c=con_colors[0])

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Shape parameter $w_0$')
ax.set_ylabel('Robustness')

plt.show()

# Pareto distribution
gamma_vals = np.linspace(3.01,5.,100)
rob_vals = rob_pareto(gamma_vals)

fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.plot(gamma_vals, rob_vals, c=con_colors[1])
ax.plot([3.,3.], [1., np.max(rob_vals)], c='k', lw=1, ls='--')

ax.set_yscale('log')

ax.set_xlabel('Exponent $\\gamma$')
ax.set_ylabel('Robustness')

plt.show()

# Lognormal distribution
mu_vals = np.linspace(0.,3.,100)
sigma_vals = [.2,.4,.6,.8,1.]
cmap = plt.get_cmap('cool', len(sigma_vals))

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

for i, sigma in enumerate(sigma_vals):
    ax.plot(mu_vals, rob_lognormal(mu_vals,sigma), c=cmap(i), label=f"$\sigma={sigma}$")
    
ax.legend()    

# ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Log mean $\mu$')
ax.set_ylabel('Robustness')

plt.show()

#------------------------------------------------------------------------------
# ROBUSTNESS IN THE SAME PLOT
#------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.plot(w0_vals, rob_dirac(w0_vals), c=con_colors[0], lw=2, label='Dirac')
ax.plot(w0_vals, rob_poisson(w0_vals), c=con_colors[1], lw=2, label='Poisson')
ax.plot(w0_vals, rob_gamma(1.,w0_vals), c=con_colors[2], lw=2, label='Exponential')

ax.legend()

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('Mean')
ax.set_ylabel('Robustness')










