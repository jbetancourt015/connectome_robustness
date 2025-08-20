"""
    This script simulates the sensitivity for various synthetic distributions
    and compares it to the LLN prediction
-------------------------------------------------------------------------------
created on:
    Mon 16 Jun 2025
-------------------------------------------------------------------------------
last change:
    Mon 16 Jun 2025
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
import seaborn as sns
from numba import njit
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

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
@njit
def compute_sensitivity(w, alpha, normalized=False):
    s_alpha = np.sum(w**alpha)
    s_2 = np.sum(w**2)
    if not normalized:
        return s_alpha/s_2
    else:
        s_1 = np.sum(w)
        n = len(w)
        return ((s_1/n)**(2-alpha))*s_alpha/s_2

@njit
def compute_asymp_sensitivity(dist='exponential', loc=1., scale=1., gamma=4.):
    if dist == 'exponential':
        return 1/2
    elif dist == 'poisson':
        return 1./(1.+1./loc)
    elif dist == 'lognormal':
        return np.exp(-scale**2)
    elif dist == 'pareto':
        return 1.-(gamma-2.)**(-2)

#------------------------------------------------------------------------------
# SIMULATION SETUP
#------------------------------------------------------------------------------
np.random.seed(42)

n_inputs_vals = [10,100,1000]
n_draws = int(1e5)

@njit
def run_sim(n_inputs, n_draws, alpha, dist='exponential', loc=1., scale=1., gamma=4.):
    # Initialize array of sensitivity values
    Q_vals = np.zeros(n_draws)
    # Simulate various input arrays
    for i in range(n_draws):
        if dist == 'exponential':
            w_sim = np.random.exponential(scale=loc, size=n_inputs)
        elif dist == 'poisson':
            w_aux = np.random.poisson(lam=loc, size=n_inputs)
            w_sim = w_aux.astype(np.float64)
        elif dist == 'lognormal':
            w_sim = np.random.normal(loc=np.log(loc), scale=scale, size=n_inputs)
            w_sim = np.exp(w_sim)
        elif dist == 'pareto':
            w_sim = np.random.rand(n_inputs)
            w_sim = (1.-w_sim)**(-1/(gamma-1))
        Q = compute_sensitivity(w_sim, alpha, normalized=True)
        Q_vals[i] = Q
    return Q_vals

#------------------------------------------------------------------------------
# PLOT SIMULATION 
#------------------------------------------------------------------------------
def plot_simulation(n_inputs_vals, n_draws, alpha, dist='exponential', loc=1., scale=1., gamma=4.):
    # Setup figure
    n_plots = len(n_inputs_vals)
    # Simulate for different number of inputs
    for i, n_inputs in enumerate(n_inputs_vals):
        Q = run_sim(n_inputs, n_draws, alpha, dist, loc, scale, gamma)
        sns.kdeplot(Q, color=con_colors[0], alpha=(i+1)/n_plots, label='$n = %s$'%(n_inputs))
    plt.legend()
    # Get predicted asymptotic sensitivity
    if gamma > 3:
        Q_asymp = compute_asymp_sensitivity(dist, loc, scale, gamma)
        plt.axvline(Q_asymp, c='k', ls='--', zorder=-1)
    plt.xlim([0,1])
    plt.xlabel('Normalized sensitivity $\\tilde{Q}_0({\\bf w})$')
    plt.ylabel('Density')
    plt.savefig('../figures/synthetic_distributions/%s_simulated_sensitivity.pdf'%(dist + ('_heavy' if gamma<3 else '')),
                dpi=600, bbox_inches='tight')
    plt.show()

plot_simulation(n_inputs_vals, n_draws, alpha=0., dist='exponential', loc=5.)
plot_simulation(n_inputs_vals, n_draws, alpha=0., dist='poisson', loc=5.)
plot_simulation(n_inputs_vals, n_draws, alpha=0., dist='lognormal', loc=5., scale=np.sqrt(np.log(2.)))
plot_simulation(n_inputs_vals, n_draws, alpha=0., dist='pareto', gamma=2.+np.sqrt(2.))

# Plot heavy-tailed Pareto
plot_simulation(n_inputs_vals, n_draws, alpha=0., dist='pareto', gamma=2.)