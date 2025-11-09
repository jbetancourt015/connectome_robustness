"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
-------------------------------------------------------------------------------
created on:
    Tue 3 Jun 2024
-------------------------------------------------------------------------------
last change:
    Tue 14 Oct 2025
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
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit
from tqdm import tqdm
from matplotlib.collections import LineCollection
from scipy import sparse

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
               'platynereis_sensory_motor', 'mouse_retina', 'drosophila_whole_brain']

data_idx = 5
thresholded = False
scheme = 'remove'

suffix = '_thresholded' if thresholded else ''

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

cmap='viridis'

# Get connectome
A = network_functions.load_connectome(data_idx, thresholded=thresholded, scheme=scheme)
N = A.shape[0]

#------------------------------------------------------------------------------
# OBTAIN SUBSET OF NEURONS
#------------------------------------------------------------------------------
rng = np.random.default_rng(52)
n_samples = 100

# Vectorized in-degree: nnz per column
in_degree = np.diff(A.indptr)          # shape (N,)

k_min = 10
eligible = np.flatnonzero(in_degree >= k_min)

# Sample from the eligible set
take = min(n_samples, eligible.size)
sample_nodes = rng.choice(eligible, size=take, replace=False)

incoming_weights = [
    A.data[A.indptr[j] : A.indptr[j+1]]
    for j in sample_nodes
]

#------------------------------------------------------------------------------
# SIMULATION PARAMETERS
#------------------------------------------------------------------------------
n_draws = int(1e3)          # number of input draws
n_perturb = int(1e3)        # number of weight perturbation draws

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


def wilson_ci(p, n, z):
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z/denom) * np.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return center - half, center + half

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

# Generate raw plot
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

ax.scatter(s_weights, P_weights, color=con_colors[0], rasterized=True)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.savefig('../../figures/simulations/sampled_weight_dist.pdf', dpi=600, bbox_inches='tight')

# Add labels
ax.set_xlabel('Connection strength')
ax.set_ylabel('Probability')

# plt.savefig('../../raw_figures/simulations/sampled_weight_dist.pdf', dpi=600)
plt.show()

#------------------------------------------------------------------------------
# LOSS BY NEURON STATISTICS
#------------------------------------------------------------------------------
stats = {
    'mean':{'fn': lambda w: np.mean(w), 'label': 'Average incoming weight'},
    'var':{'fn': lambda w: np.var(w), 'label': 'Variance in weight'},
    'fano':{'fn': lambda w: np.var(w)/np.mean(w), 'label': 'Fano factor'}
    }

# Initialize vector of statistics
stat_vals = {x: [] for x in stats}
loss = []
pred_loss = []
num_inputs = []

# Perturbation params
plotting_eps = [.01,.1,1.]
eta = 1.

# Run simulation
for i, w in enumerate(tqdm(incoming_weights)):
    # Get number of inputs
    n_inputs = len(w)
    if n_inputs > 0:
        num_inputs.append(n_inputs)
        Q = sensitivity(w,eta)
            
        # Draw perturbations
        w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
        w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
        
        # Draw inputs
        x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
        
        # Compute loss
        loss_vec = []
        pred_loss_vec = []
        for eps in plotting_eps:
            l_hat = average_error(x,w,w_hat,eps)
            pred_l = predicted_loss(w,eta,eps)
            loss_vec.append(l_hat)
            pred_loss_vec.append(pred_l)
        loss.append(loss_vec)
        pred_loss.append(pred_loss_vec)
        
        # Compute statistics
        for stat in stats:
            stat_vals[stat].append(stats[stat]['fn'](w))

# Turn into numpy arrays
loss = np.array(loss)
pred_loss = np.array(pred_loss)
num_inputs = np.array(num_inputs)

for stat in stat_vals:
    stat_vals[stat] = np.array(stat_vals[stat])

# FIGURE: LOSS VS INCOMING WEIGHT (COLOR VAR)----------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Draw scatterplot (eps=1)
sc = ax_scatter.scatter(stat_vals['mean'], 
                        loss[:,2], 
                        c=stat_vals['var'], 
                        cmap='viridis', 
                        norm=LogNorm())
cbar = fig.colorbar(sc, cax=ax_cbar)
cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

# Add labels
ax_scatter.set_xlabel(stats['mean']['label'])
ax_scatter.set_ylabel('Simulated loss')
cbar.set_label(stats['var']['label'])

# plt.savefig(f"../../figures/candidate_figures/fig_1{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()

# FIGURE: LOSS VS INCOMING VARIANCE (COLOR MEAN)-------------------------------
# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Draw scatterplot (eps=1)
sc = ax_scatter.scatter(stat_vals['var'], 
                        loss[:,2], 
                        c=stat_vals['mean'], 
                        cmap='viridis')
cbar = fig.colorbar(sc, cax=ax_cbar)
cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

ax_scatter.set_xscale('log')

# Add labels
ax_scatter.set_xlabel(stats['var']['label'])
ax_scatter.set_ylabel('Simulated loss')
cbar.set_label(stats['mean']['label'])

# plt.savefig(f"../../figures/candidate_figures/fig_2{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()

#------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
#------------------------------------------------------------------------------
# Get loss range
min_loss, max_loss = min(pred_loss[:,2]), max(pred_loss[:,2])

# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(pred_loss[:,2], loss[:,2], c=stat_vals['mean'], cmap='viridis')
cbar = fig.colorbar(sc, cax=ax_cbar)

# Plot y=x line
ax_scatter.plot([min_loss,max_loss], [min_loss,max_loss], c='k', ls='--', lw=1)

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

# Add labels
ax_scatter.set_xlabel('Predicted loss')
ax_scatter.set_ylabel('Simulated loss')
cbar.set_label(stats['mean']['label'])

# plt.savefig(f"../../figures/candidate_figures/fig_3{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()

#------------------------------------------------------------------------------
# LOSS VS ROBUSTNESS
#------------------------------------------------------------------------------
# Get variance range
min_loss, max_loss = min(pred_loss[:,2]), max(pred_loss[:,2])

# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
robustness = ((stat_vals['var']+stat_vals['mean']**2)/stat_vals['mean'])**0.5
sc = ax_scatter.scatter(robustness, loss[:,2], c=stat_vals['mean'], cmap='viridis')
cbar = fig.colorbar(sc, cax=ax_cbar)

# Plot y=x line
r_vals = np.linspace(min(robustness), max(robustness), 100)
ax_scatter.plot(r_vals, (1./np.pi)*np.arccos((1.+r_vals**(-2))**(-0.5)), c='k', ls='--', lw=1, label='Prediction')
ax_scatter.legend()

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

# Add labels
ax_scatter.set_xlabel('Robustness')
ax_scatter.set_ylabel('Simulated loss')
cbar.set_label(stats['mean']['label'])

# plt.savefig(f"../../figures/candidate_figures/fig_4{suffix}.pdf", dpi=600, bbox_inches='tight')

plt.show()

#------------------------------------------------------------------------------
# RELATIVE ERROR VS NUMBER OF INCOMING CONNECTIONSo
#------------------------------------------------------------------------------
# Get relative error
rel_error = np.abs(loss[:,2]/pred_loss[:,2] - 1)
min_loss, max_loss = min(pred_loss[:,2]), max(pred_loss[:,2])

# Set up figure
fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
divider = make_axes_locatable(ax_scatter)
ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# Compare model prediction with simulated loss
sc = ax_scatter.scatter(num_inputs, rel_error, c=stat_vals['mean'], cmap='viridis')
cbar = fig.colorbar(sc, cax=ax_cbar)

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')

# Add labels
ax_scatter.set_xlabel('Number of inputs')
ax_scatter.set_ylabel('Relative error')
cbar.set_label(stats['mean']['label'])

plt.show()

# #------------------------------------------------------------------------------
# # PREDICTED LOSS PLOT
# #------------------------------------------------------------------------------
# # Get range of stats
# for stat in stats:
#     # Get full range
#     stats[stat]['range'] = [min(stat_vals[stat]), max(stat_vals[stat])]
#     # Get constrained range
#     stats[stat]['const_range'] = [min(stat_vals[stat][stat_vals[stat] > 0]), max(stat_vals[stat])]

# def pred_loss_stats(eps, mean, stat2, stat_name='var'):
#     if stat_name == 'var':
#         rho = (1.+mean/(stat2+mean**2))**(-1/2)
#     else:
#         rho = (1.+1/(stat2+mean))**(-1/2)
#     return (1/np.pi)*np.arccos(rho)

# # Define grid parameters
# n_grid = 100

# for i, eps in enumerate(plotting_eps):
#     for stat in ['var', 'fano']:
#         # Do log and not log
#         for log_x in [True, False]:
#             for log_y in [True, False]:
#                 for log_c in [True, False]:
#                     if log_x:
#                         s1_grid = np.exp(np.linspace(np.log(stats['mean']['const_range'][0]), np.log(stats['mean']['range'][1]), n_grid))
#                         x_min = stats['mean']['const_range'][0]
#                     else:
#                         s1_grid = np.linspace(stats['mean']['range'][0], stats['mean']['range'][1], n_grid)
#                         x_min = stats['mean']['range'][0]
                    
#                     if log_y:
#                         s2_grid = np.exp(np.linspace(np.log(stats[stat]['const_range'][0]), np.log(stats[stat]['range'][1]), n_grid))
#                         y_min = stats[stat]['const_range'][0]
#                     else:
#                         s2_grid = np.linspace(stats[stat]['range'][0], stats[stat]['range'][1], n_grid)
#                         y_min = stats[stat]['range'][0]
                    
#                     x_max = stats['mean']['range'][1]
#                     y_max = stats[stat]['range'][1]
                    
#                     S1, S2 = np.meshgrid(s1_grid, s2_grid, indexing="xy")
                    
#                     L = pred_loss_stats(eps, S1, S2, stat_name=stat)
                    
#                     fig, ax = plt.subplots(figsize=(.9*width, .9*height))
#                     norm = LogNorm(vmin=np.min(L), vmax=np.max(L)) if log_c else None
#                     im = ax.imshow(
#                         L,
#                         origin="lower",
#                         extent=[x_min, x_max, y_min, y_max],
#                         aspect="auto",
#                         interpolation="nearest",
#                         cmap=cmap,
#                         norm=norm,
#                     )
                    
#                     # Draw scatterplot
#                     sc = ax.scatter(stat_vals['mean'],
#                                     stat_vals[stat], 
#                                     c=loss[:,i], 
#                                     cmap=cmap, 
#                                     norm=LogNorm() if log_c else None,
#                                     edgecolors='k')
                    
#                     if log_x:
#                         ax.set_xscale('log')
#                     if log_y:
#                         ax.set_yscale('log')
                    
#                     ax.set_xlabel(stats['mean']['label'])
#                     ax.set_ylabel(stats[stat]['label'])
#                     cbar = fig.colorbar(im, ax=ax)
#                     cbar.set_label('Predicted loss')
                    
#                     ax.text(0.1, 0.1, f"$\epsilon$={eps}",
#                                     ha='left',
#                                     va='bottom',
#                                     transform=ax.transAxes)
                    
#                     plt.show()
