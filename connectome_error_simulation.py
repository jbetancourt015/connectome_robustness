"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
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
import network_functions
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

# Connectome list
connectomes = ['drosophila_central_brain','drosophila_optic_medulla','c_elegans',
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

data_idx = 5

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255;

# Get connectome
A = network_functions.load_connectome(data_idx)
N = A.shape[0]

#------------------------------------------------------------------------------
# OBTAIN SUBSET OF NEURONS
#------------------------------------------------------------------------------
np.random.seed(52)

n_samples = 100
sample_nodes = np.random.choice(N, size=n_samples, replace=False)

incoming_weights = [
    A.data[A.indptr[j] : A.indptr[j+1]]
    for j in sample_nodes
]

#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
n_draws = int(1e3)          # number of input draws
n_perturb = int(1e3)        # number of weight perturbation draws
eps_vals = np.exp(np.log(10)*np.linspace(-2,2,20))      # normalized perturbation strength

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

# Get vector of all weights
weights = np.concatenate(incoming_weights)
s_weights, P_weights = empirical_hist(weights)

plt.scatter(s_weights, P_weights, color=con_colors[0], rasterized=True)
plt.xlabel('Connection strength')
plt.ylabel('Probability')
plt.xscale('log')
plt.yscale('log')
plt.savefig('../figures/simulations/sampled_weight_dist.pdf', dpi=600, bbox_inches='tight')
plt.show()

# Generate raw plot
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_weights, P_weights, color=con_colors[0], rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('../raw_figures/simulations/sampled_weight_dist.pdf', dpi=600)
plt.show()

#------------------------------------------------------------------------------
# LOSS VS NEURON PARAMS
#------------------------------------------------------------------------------
# Initialize loss and parameter vectors
loss = []
mean = []
var = []

# Perturbation parameters
eta = 1.
eps = 1.

for i, w in enumerate(tqdm(incoming_weights)):
    # Get number of inputs
    n_inputs = len(w)
    Q = compute_sensitivity(w,eta)
        
    # Draw perturbations
    w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
    w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
    
    # Draw inputs
    x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
    
    # Save loss and vector statistics
    loss.append(average_error(x,w,w_hat,1.))
    mean.append(np.mean(w))
    var.append(np.var(w))

# Convert to numpy arrays
loss = np.array(loss)
mean = np.array(mean)
var = np.array(var)

# Get variance range
min_var, max_var = min(var), max(var)
min_mean, max_mean = min(mean), max(mean)

n_plot = 100
mean_vals = np.linspace(min_mean, max_mean, n_plot)

# Separate high and low variance values
high_mask = (var > np.median(var))
low_mask = (var <= np.median(var))

# Compare with model predictions
fig, ax = plt.subplots(figsize=(.9*width, .9*height))
ax.scatter(mean[high_mask], loss[high_mask], color=con_colors[1], rasterized=True)
ax.scatter(mean[low_mask], loss[low_mask], color=con_colors[0], rasterized=True)
ax.plot(mean_vals, np.arccos((1+(eps**2)*(mean_vals/(max_var + mean_vals**2)))**(-1/2))/np.pi, c=con_colors[1])
ax.plot(mean_vals, np.arccos((1+(eps**2)*(mean_vals/(min_var + mean_vals**2)))**(-1/2))/np.pi, c=con_colors[0])
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig(f"../raw_figures/simulations/loss_vs_mean_eta_{int(eta)}.pdf", dpi=600)
plt.show()


#------------------------------------------------------------------------------
# SIMULATION
#------------------------------------------------------------------------------
# gaussian_loss = []
# binary_loss = []
# eta = 2.

# for i, w in enumerate(tqdm(incoming_weights)):
#     # Get number of inputs
#     n_inputs = len(w)
#     Q = compute_sensitivity(w,eta)
        
#     # Draw perturbations
#     w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
#     w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
    
#     # Draw inputs
#     x_gaussian = np.random.normal(0.,size=(n_inputs,n_draws))
#     x_binary = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
    
#     # Loop over epsilons
#     gaussian_loss_eps = []
#     binary_loss_eps = []
#     for eps in eps_vals:
#         gaussian_loss_eps.append(average_error(x_gaussian,w,w_hat,eps/(Q**(1/2))))
#         binary_loss_eps.append(average_error(x_binary,w,w_hat,eps/(Q**(1/2))))
        
#     gaussian_loss.append(gaussian_loss_eps)
#     binary_loss.append(binary_loss_eps)

# # Get error statistics
# gaussian_loss = np.array(gaussian_loss)
# binary_loss = np.array(binary_loss)

# median_gaussian = np.median(gaussian_loss, axis=0)
# lower95_gaussian   = np.percentile(gaussian_loss, 2.5, axis=0)
# upper95_gaussian   = np.percentile(gaussian_loss, 97.5, axis=0)

# median_binary = np.median(binary_loss, axis=0)
# lower95_binary   = np.percentile(binary_loss, 2.5, axis=0)
# upper95_binary   = np.percentile(binary_loss, 97.5, axis=0)

# # Plot loss curves
# plt.plot(eps_vals, median_gaussian, c=con_colors[0], label='Median')
# plt.fill_between(eps_vals, lower95_gaussian, upper95_gaussian, color=con_colors[0], alpha=0.3, label='95\% range')
# plt.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7, label='Prediction')
# plt.legend()
# plt.xlabel('Perturbation strength $\epsilon Q_{%s}^{1/2}$'%(int(eta)))
# plt.ylabel('Average loss ${\cal E}({\\bf w})$')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"../figures/simulations/connectome_gaussian_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()

# plt.plot(eps_vals, median_binary, c=con_colors[0], label='Median')
# plt.fill_between(eps_vals, lower95_binary, upper95_binary, color=con_colors[0], alpha=0.3, label='95\% range')
# plt.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7, label='Prediction')
# plt.legend()
# plt.xlabel('Perturbation strength $\epsilon Q_{%s}^{1/2}$'%(int(eta)))
# plt.ylabel('Average loss ${\cal E}({\\bf w})$')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"../figures/simulations/connectome_binary_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()

# # Plot raw plots
# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(eps_vals, median_gaussian, c=con_colors[0])
# ax.fill_between(eps_vals, lower95_gaussian, upper95_gaussian, color=con_colors[0], alpha=0.3)
# ax.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.savefig(f"../raw_figures/simulations/connectome_gaussian_eta_{int(eta)}.pdf", dpi=600)
# plt.show()

# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(eps_vals, median_binary, c=con_colors[0])
# ax.fill_between(eps_vals, lower95_binary, upper95_binary, color=con_colors[0], alpha=0.3)
# ax.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.savefig(f"../raw_figures/simulations/connectome_binary_eta_{int(eta)}.pdf", dpi=600)
# plt.show()
