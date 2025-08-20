"""
    This script computes the expected loss for inputs that are not mean-zero
    in terms of the unperturbed firing rate
-------------------------------------------------------------------------------
created on:
    Tue 3 Jun 2024
-------------------------------------------------------------------------------
last change:
    Tue 3 Jun 2025
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.special import erfinv
from scipy.special import erf
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "font.serif": ["Garamond"],
    "text.latex.preamble": r'\usepackage{amsfonts}'
})

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

colormap = fade_to_color_cmap(con_colors[0],alpha_min=0.1, name="fade_to_color")
# colormap = 'rainbow'

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
def phi2(h, k, rho):
    mean = (0.0, 0.0)
    cov  = ((1.0, rho), (rho, 1.0))
    return multivariate_normal(mean, cov).cdf([h, k])

def norm_loc(p0):
    return -np.sqrt(2)*erfinv(1.-2*p0)

def loss(p0, rho):
    alpha = norm_loc(p0)
    l = 0.5*(1.+erf(alpha/np.sqrt(2)))
    l += 0.5*(1.+erf(rho*alpha/np.sqrt(2)))
    l -= 2*phi2(alpha, rho*alpha, rho)
    return l
    
#------------------------------------------------------------------------------
# PARAMETER SWEEP
#------------------------------------------------------------------------------
# Parameters
rho_vals = np.linspace(1e-5,1.-1e-5,1000)
p_min, p_max, n_p = 1e-4, .5-1e-5, 10
log_p_vals = np.linspace(np.log(p_min),np.log(p_max),n_p)
p_vals = np.exp(log_p_vals)

# Create colormap
cmap = plt.get_cmap(colormap, n_p)
norm = LogNorm(vmin= np.exp(np.log(p_min)-0.5*(np.log(p_max)-np.log(p_min))/(n_p-1)), 
               vmax= np.exp(np.log(p_max)+0.5*(np.log(p_max)-np.log(p_min))/(n_p-1)))
color = cmap(norm(p_vals))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# Plot loss values
fig                 = plt.figure(figsize=(6.4, 4.8*(1.6/2)))
ax                  = fig.add_subplot(111)

for i, p0 in enumerate(p_vals):    
    loss_vals = [loss(p0,rho) for rho in rho_vals]
    plt.plot(rho_vals, loss_vals, color=color[i])

cbar = fig.colorbar(sm)
cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
cbar.set_label('Firing probability $p_0$')

plt.xlabel('Correlation $\\rho({\\bf w})$')
plt.ylabel('Expected loss ${\cal E}({\\bf w}; p_0)$')
plt.savefig('../figures/sparse_loss.pdf', dpi=600, bbox_inches='tight')
plt.show()