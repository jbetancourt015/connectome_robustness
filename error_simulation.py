"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
-------------------------------------------------------------------------------
created on:
    Mon 12 May 2024
-------------------------------------------------------------------------------
last change:
    Mon 8 Aug 2025
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
import logging
from numba import njit
from tqdm import tqdm


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

#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
np.random.seed(32)

n_draws = int(1e3)
n_perturb = int(1e5)
eps_vals = np.exp(np.log(10)*np.linspace(-2,3,20))

n_inputs = 10
strength_vals = [2., 4., 6.]

#------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
#------------------------------------------------------------------------------
@njit
def average_error(x, w, w_hat, eps):
    error = 0.
    for i in range(n_draws):
        for j in range(n_perturb):
            z = np.sum(w*x[:,i])
            z_tilde = np.sum((w+eps*w_hat[:,j])*x[:,i])

            error += (1-np.sign(z*z_tilde))/2
    error /= n_draws*n_perturb
    return error

@njit
def compute_sensitivity(w,alpha):
    s_alpha = np.sum(w**alpha)
    s_2 = np.sum(w**2)
    return s_alpha/s_2

#------------------------------------------------------------------------------
# SIMULATION
#------------------------------------------------------------------------------
errors = []
sensitivities = []
alpha=0.

for strength in strength_vals:
    # Draw network
    w = np.ones(n_inputs)
    w += np.random.poisson(strength-1., n_inputs)
    
    # Draw inputs and perturbations
    # x = np.random.normal(0.,size=(n_inputs,n_draws))
    x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
    w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
    
    # Loop over epsilons
    errors_strength = []
    for eps in tqdm(eps_vals):
        errors_strength.append(average_error(x,w,w_hat,eps))
        
    errors.append(errors_strength)
    sensitivities.append(compute_sensitivity(w,alpha))

# Set up subfigures
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.plot(eps_vals, errors[0], c='indigo', label='$\\langle w \\rangle = 2$')
ax.plot(eps_vals, np.arccos((1+(eps_vals**2)*sensitivities[0])**(-1/2))/np.pi, c='indigo', ls='--', alpha=0.7)
ax.plot(eps_vals, errors[1], c='b', label='$\\langle w \\rangle = 4$')
ax.plot(eps_vals, np.arccos((1+(eps_vals**2)*sensitivities[1])**(-1/2))/np.pi, c='b', ls='--', alpha=0.7)
ax.plot(eps_vals, errors[2], c='r', label='$\\langle w \\rangle = 6$')
ax.plot(eps_vals, np.arccos((1+(eps_vals**2)*sensitivities[2])**(-1/2))/np.pi, c='r', ls='--', alpha=0.7)
plt.legend()
plt.xlabel('Perturbation strength $\epsilon$')
plt.ylabel('Average loss ${\cal E}({\\bf w})$')
plt.xscale('log')
plt.yscale('log')
plt.savefig('../figures/simulated_sensitivities.pdf', dpi=600, bbox_inches='tight')
plt.show()