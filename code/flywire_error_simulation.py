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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

cmap=fade_to_color_cmap(con_colors[0], alpha_min=0.2, name="fade_to_color")

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
def sensitivity(w,eta):
    s_eta = np.sum(w**eta)
    s_2 = np.sum(w**2)
    return s_eta/s_2


@njit
def predicted_loss(w,eta,eps):
    Q = sensitivity(w,eta)
    return (1./np.pi)*np.arccos((1.+(eps**2)*Q)**(-1/2))

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
plt.savefig('../../figures/simulations/sampled_weight_dist.pdf', dpi=600, bbox_inches='tight')
plt.show()

# Generate raw plot
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_weights, P_weights, color=con_colors[0], rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('../../raw_figures/simulations/sampled_weight_dist.pdf', dpi=600)
plt.show()

#------------------------------------------------------------------------------
# LOSS VS NEURON STATISTICS
#------------------------------------------------------------------------------
# Initialize loss and parameter vectors
loss = []
mean = []
m2 = []

# Perturbation parameters
eta = 1.
eps = 1.

for i, w in enumerate(tqdm(incoming_weights)):
    # Get number of inputs
    n_inputs = len(w)
    Q = sensitivity(w,eta)
        
    # Draw perturbations
    w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
    w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
    
    # Draw inputs
    x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
    
    # Save loss and vector statistics
    loss.append(average_error(x,w,w_hat,1.))
    mean.append(np.mean(w))
    m2.append(np.mean(w**2))

# Convert to numpy arrays
loss = np.array(loss)
mean = np.array(mean)
m2 = np.array(m2)
fano = (m2/mean)-mean

# Get variance range
min_mean, max_mean = min(mean), max(mean)
min_m2, max_m2 = min(m2), max(m2)
min_fano, max_fano = min(fano[fano>0]), max(fano)

n_plot = 100
mean_vals = np.linspace(min_mean, max_mean, n_plot)
m2_vals = np.linspace(min_m2, max_m2, n_plot)
fano_vals = np.linspace(min_fano, max_fano, n_plot)


# FIGURE: LOSS VS MEAN---------------------------------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(mean, loss, c=m2, cmap='viridis', norm=LogNorm())
cbar = fig.colorbar(sc, cax=ax_cbar)

# Pick values of the second moment for analytical curves
n_curves = 3
m2_curves = np.logspace(np.log10(min_m2), np.log10(max_m2), n_curves)
cmap = sc.get_cmap()
norm = sc.norm

for x in m2_curves:
    ax_scatter.plot(mean_vals, np.arccos((1+(eps**2)*(mean_vals/x))**(-1/2))/np.pi, c=cmap(norm(x)))

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
plt.savefig(f"../../raw_figures/simulations/loss_vs_mean_eta_{int(eta)}.pdf", dpi=600)

# Add labels
ax_scatter.set_xlabel('Average strength')
ax_scatter.set_ylabel('Expected loss')
cbar.set_label('Average squared strength')

plt.savefig(f"../../figures/simulations/loss_vs_mean_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
plt.show()


# FIGURE: LOSS VS SECOND MOMENT------------------------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(m2, loss, c=mean, cmap='viridis', norm=LogNorm())
cbar = fig.colorbar(sc, cax=ax_cbar)

# Pick values of the second moment for analytical curves
n_curves = 3
mean_curves = np.logspace(np.log10(min_mean), np.log10(max_mean), n_curves)
cmap = sc.get_cmap()
norm = sc.norm

for x in mean_curves:
    ax_scatter.plot(m2_vals, np.arccos((1+(eps**2)*(x/m2_vals))**(-1/2))/np.pi, c=cmap(norm(x)))

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
plt.savefig(f"../../raw_figures/simulations/loss_vs_m2_eta_{int(eta)}.pdf", dpi=600)

# Add labels
ax_scatter.set_xlabel('Average squared strength')
ax_scatter.set_ylabel('Expected loss')
cbar.set_label('Average strength')

plt.savefig(f"../../figures/simulations/loss_vs_m2_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
plt.show()


# FIGURE: LOSS VS MEAN (BY FANO FACTOR)----------------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(mean, loss, c=fano, cmap='viridis', norm=LogNorm())
cbar = fig.colorbar(sc, cax=ax_cbar)

# Pick values of the second moment for analytical curves
n_curves = 3
fano_curves = np.logspace(np.log10(min_fano), np.log10(max_fano), n_curves)
cmap = sc.get_cmap()
norm = sc.norm

for x in fano_curves:
    ax_scatter.plot(mean_vals, np.arccos((1+(eps**2)*(1./(x+mean_vals)))**(-1/2))/np.pi, c=cmap(norm(x)))

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
plt.savefig(f"../../raw_figures/simulations/loss_vs_mean_fano_eta_{int(eta)}.pdf", dpi=600)

# Add labels
ax_scatter.set_xlabel('Average strength')
ax_scatter.set_ylabel('Expected loss')
cbar.set_label('Fano factor')

plt.savefig(f"../../figures/simulations/loss_vs_mean_fano_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
plt.show()


# FIGURE: LOSS VS FANO FACTOR--------------------------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(fano, loss, c=mean, cmap='viridis', norm=LogNorm())
cbar = fig.colorbar(sc, cax=ax_cbar)

# Pick values of the second moment for analytical curves
n_curves = 3
mean_curves = np.logspace(np.log10(min_mean), np.log10(max_mean), n_curves)
cmap = sc.get_cmap()
norm = sc.norm

for x in mean_curves:
    ax_scatter.plot(fano_vals, np.arccos((1+(eps**2)*(1./(fano_vals+x)))**(-1/2))/np.pi, c=cmap(norm(x)))

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
plt.savefig(f"../../raw_figures/simulations/loss_vs_fano_eta_{int(eta)}.pdf", dpi=600)

# Add labels
ax_scatter.set_xlabel('Fano factor')
ax_scatter.set_ylabel('Expected loss')
cbar.set_label('Average strength')

plt.savefig(f"../../figures/simulations/loss_vs_fano_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
#------------------------------------------------------------------------------
# # Initialize loss and parameter vectors
# sim_loss = []
# pred_loss = []
# baseline_loss = []

# # Perturbation parameters
# eta = 1.
# eps_vals = [.5,1.,2.]

# for i, w in enumerate(tqdm(incoming_weights)):
#     # Get number of inputs
#     n_inputs = len(w)
        
#     # Draw perturbations
#     w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
#     w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
    
#     # Draw inputs
#     x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
    
#     # Save loss for different values of perturbation strenght
#     sim_loss_vals = []
#     pred_loss_vals = []
    
#     for eps in eps_vals:
#         sim_loss_vals.append(average_error(x,w,w_hat,eps))
#         pred_loss_vals.append(predicted_loss(w,eta,eps))
    
#     # Save loss and vector statistics
#     sim_loss.append(sim_loss_vals)
#     pred_loss.append(pred_loss_vals)
#     baseline_loss.append(predicted_loss(w,eta,1.))

# # Convert to numpy arrays
# sim_loss = np.array(sim_loss)
# pred_loss = np.array(pred_loss)
# baseline_loss = np.array(baseline_loss)

# # Plot prediction vs simuated loss for different epsilon values
# for i, eps in enumerate(eps_vals):
#     # Get variance range
#     min_loss, max_loss = min(pred_loss[:,i]), max(pred_loss[:,i])
    
#     # FIGURE: PREDICTED VS SIMULATED LOSS
#     # Set up figure
#     fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
#     divider = make_axes_locatable(ax_scatter)
#     ax_cbar = divider.append_axes('right', size='5%', pad=0.1)
    
#     # Compare model prediction with simulated loss
#     sc = ax_scatter.scatter(pred_loss[:,i], sim_loss[:,i], c=baseline_loss, cmap='viridis', norm=LogNorm())
#     cbar = fig.colorbar(sc, cax=ax_cbar)
    
#     # Plot y=x line
#     ax_scatter.plot([min_loss,max_loss], [min_loss,max_loss], c='k', ls='--', lw=1)
    
#     cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
#     ax_scatter.set_xscale('log')
#     ax_scatter.set_yscale('log')
#     plt.savefig(f"../../raw_figures/simulations/pred_vs_sim_loss_eta_{int(eta)}_eps_{i}.pdf", dpi=600)
    
#     # Add labels
#     ax_scatter.set_xlabel('Predicted loss')
#     ax_scatter.set_ylabel('Simulated loss')
#     cbar.set_label('Baseline loss')
#     ax_scatter.text(max_loss/(max_loss/min_loss)**0.05, min_loss*(max_loss/min_loss)**0.05,
#                     f"$\epsilon$={eps}", ha="right", va="bottom")
    
#     plt.savefig(f"../../figures/simulations/pred_vs_sim_loss_eta_{int(eta)}_eps_{i}.pdf", dpi=600, bbox_inches='tight')
#     plt.show()
    

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
