"""
    This script processes the raw connectome data and generates sparse datasets
    to speed up loading.
-------------------------------------------------------------------------------
created on:
    Tue 3 Jun 2024
-------------------------------------------------------------------------------
last change:
    Tue 9 Sep 2025
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
from matplotlib.collections import LineCollection

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
thresholded = True

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [255, 204, 0], [203, 41, 123], [0, 0, 0]])/255;

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    bottom = (*rgb, alpha_min)
    top    = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)

cmap='viridis'

# Get connectome
A = network_functions.load_connectome(data_idx, thresholded=thresholded)
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

# Perturbation params
plotting_eps = [.01,.1,1.]
eta = 1.

# Run simulation
for i, w in enumerate(tqdm(incoming_weights)):
    # Get number of inputs
    n_inputs = len(w)
    if n_inputs > 0:
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

for stat in stat_vals:
    stat_vals[stat] = np.array(stat_vals[stat])
    
# Plot loss vs parameters
for i, eps in enumerate(plotting_eps):
    for stat1 in stats:
        for stat2 in stats:
            if stat1 != stat2:
                # Do log and not log
                for log_x in [True, False]:
                    for log_y in [True, False]:
                        for log_c in [True, False]:
                            # Set up figure
                            fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
                            divider = make_axes_locatable(ax_scatter)
                            ax_cbar = divider.append_axes('right', size='5%', pad=0.1)
                            
                            # Draw scatterplot
                            sc = ax_scatter.scatter(stat_vals[stat1], 
                                                    loss[:,i], 
                                                    c=stat_vals[stat2], 
                                                    cmap='viridis', 
                                                    norm=LogNorm() if log_c else None)
                            cbar = fig.colorbar(sc, cax=ax_cbar)
                            cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
                            
                            if log_x:
                                ax_scatter.set_xscale('log')
                            if log_y:
                                ax_scatter.set_yscale('log')
                            
                            # Add labels
                            ax_scatter.set_xlabel(stats[stat1]['label'])
                            ax_scatter.set_ylabel('Simulated loss')
                            cbar.set_label(stats[stat2]['label'])
                            ax_scatter.text(0.1, 0.1, f"$\epsilon$={eps}",
                                            ha='left',
                                            va='bottom',
                                            transform=ax_scatter.transAxes)
                            
                            if (eps==1) and (stat1=='mean') and (stat2=='var') and (not log_x) and (not log_y) and log_c:
                                plt.savefig("../../figures/candidate_figures/fig_1.pdf", dpi=600, bbox_inches='tight')
                                
                            if (eps==1) and (stat1=='var') and (stat2=='mean') and log_x and (not log_y) and (not log_c):
                                plt.savefig("../../figures/candidate_figures/fig_2.pdf", dpi=600, bbox_inches='tight')
                            
                            plt.show()

#------------------------------------------------------------------------------
# PREDICTED LOSS PLOT
#------------------------------------------------------------------------------
# Get range of stats
for stat in stats:
    # Get full range
    stats[stat]['range'] = [min(stat_vals[stat]), max(stat_vals[stat])]
    # Get constrained range
    stats[stat]['const_range'] = [min(stat_vals[stat][stat_vals[stat] > 0]), max(stat_vals[stat])]

def pred_loss_stats(eps, mean, stat2, stat_name='var'):
    if stat_name == 'var':
        rho = (1.+mean/(stat2+mean**2))**(-1/2)
    else:
        rho = (1.+1/(stat2+mean))**(-1/2)
    return (1/np.pi)*np.arccos(rho)

# Define grid parameters
n_grid = 100

for i, eps in enumerate(plotting_eps):
    for stat in ['var', 'fano']:
        # Do log and not log
        for log_x in [True, False]:
            for log_y in [True, False]:
                for log_c in [True, False]:
                    if log_x:
                        s1_grid = np.exp(np.linspace(np.log(stats['mean']['const_range'][0]), np.log(stats['mean']['range'][1]), n_grid))
                        x_min = stats['mean']['const_range'][0]
                    else:
                        s1_grid = np.linspace(stats['mean']['range'][0], stats['mean']['range'][1], n_grid)
                        x_min = stats['mean']['range'][0]
                    
                    if log_y:
                        s2_grid = np.exp(np.linspace(np.log(stats[stat]['const_range'][0]), np.log(stats[stat]['range'][1]), n_grid))
                        y_min = stats[stat]['const_range'][0]
                    else:
                        s2_grid = np.linspace(stats[stat]['range'][0], stats[stat]['range'][1], n_grid)
                        y_min = stats[stat]['range'][0]
                    
                    x_max = stats['mean']['range'][1]
                    y_max = stats[stat]['range'][1]
                    
                    S1, S2 = np.meshgrid(s1_grid, s2_grid, indexing="xy")
                    
                    L = pred_loss_stats(eps, S1, S2, stat_name=stat)
                    
                    fig, ax = plt.subplots(figsize=(.9*width, .9*height))
                    norm = LogNorm(vmin=np.min(L), vmax=np.max(L)) if log_c else None
                    im = ax.imshow(
                        L,
                        origin="lower",
                        extent=[x_min, x_max, y_min, y_max],
                        aspect="auto",
                        interpolation="nearest",
                        cmap=cmap,
                        norm=norm,
                    )
                    
                    # Draw scatterplot
                    sc = ax.scatter(stat_vals['mean'],
                                    stat_vals[stat], 
                                    c=loss[:,i], 
                                    cmap=cmap, 
                                    norm=LogNorm() if log_c else None,
                                    edgecolors='k')
                    
                    if log_x:
                        ax.set_xscale('log')
                    if log_y:
                        ax.set_yscale('log')
                    
                    ax.set_xlabel(stats['mean']['label'])
                    ax.set_ylabel(stats[stat]['label'])
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Predicted loss')
                    
                    ax.text(0.1, 0.1, f"$\epsilon$={eps}",
                                    ha='left',
                                    va='bottom',
                                    transform=ax.transAxes)
                    
                    plt.show()

#------------------------------------------------------------------------------
# PREDICTED VS SIMULATED LOSS
#------------------------------------------------------------------------------
# Plot prediction vs simuated loss for different epsilon values
for i, eps in enumerate(plotting_eps):
    for stat in stats:
        for log_l in [True, False]:
            for log_c in [True, False]:
    
                # Get variance range
                min_loss, max_loss = min(pred_loss[:,i]), max(pred_loss[:,i])
                
                # Set up figure
                fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
                divider = make_axes_locatable(ax_scatter)
                ax_cbar = divider.append_axes('right', size='5%', pad=0.1)
                
                # Compare model prediction with simulated loss
                sc = ax_scatter.scatter(pred_loss[:,i], loss[:,i], c=stat_vals[stat], cmap='viridis', norm=LogNorm() if log_c else None)
                cbar = fig.colorbar(sc, cax=ax_cbar)
                
                # Plot y=x line
                ax_scatter.plot([min_loss,max_loss], [min_loss,max_loss], c='k', ls='--', lw=1)
                
                cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
                
                if log_l:
                    ax_scatter.set_xscale('log')
                    ax_scatter.set_yscale('log')
                
                # Add labels
                ax_scatter.set_xlabel('Predicted loss')
                ax_scatter.set_ylabel('Simulated loss')
                cbar.set_label(stats[stat]['label'])
                ax_scatter.text(0.1, 0.1, f"$\epsilon$={eps}",
                                ha='left',
                                va='bottom',
                                transform=ax_scatter.transAxes)
                
                
                if (eps==1) and (stat=='mean') and (not log_x) and (not log_y) and (not log_c):
                    plt.savefig("../../figures/candidate_figures/fig_3.pdf", dpi=600, bbox_inches='tight')
            
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
ax_scatter.plot(r_vals, (1./np.pi)*np.arccos((1.+r_vals**(-2))**(-0.5)), c='k', ls='--', lw=1)

cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)

# Add labels
ax_scatter.set_xlabel('Robustness')
ax_scatter.set_ylabel('Loss')
cbar.set_label(stats['mean']['label'])

plt.savefig("../../figures/candidate_figures/fig_4.pdf", dpi=600, bbox_inches='tight')

plt.show()


# #------------------------------------------------------------------------------
# # LOSS VS NEURON STATISTICS
# #------------------------------------------------------------------------------
# # Initialize loss and parameter vectors
# loss = []
# mean = []
# m2 = []
# wilson_lo = []
# wilson_hi = []

# # Perturbation parameters
# eta = 1.
# eps = 1.

# for i, w in enumerate(tqdm(incoming_weights)):
#     # Get number of inputs
#     n_inputs = len(w)
#     if n_inputs > 0:
#         Q = sensitivity(w,eta)
            
#         # Draw perturbations
#         w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
#         w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
        
#         # Draw inputs
#         x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
        
#         # Compute loss
#         l_hat = average_error(x,w,w_hat,1.)
        
#         # Get confidence interval
#         ci = wilson_ci(l_hat, n_draws*n_perturb, 2.)
        
#         # Save loss and vector statistics
#         loss.append(l_hat)
#         wilson_lo.append(ci[0])
#         wilson_hi.append(ci[1])
#         mean.append(np.mean(w))
#         m2.append(np.mean(w**2))

# # Convert to numpy arrays
# loss = np.array(loss)
# mean = np.array(mean)
# m2 = np.array(m2)
# fano = (m2/mean)-mean
# wilson_lo = np.array(wilson_lo)
# wilson_hi = np.array(wilson_hi)

# # Get variance range
# min_mean, max_mean = min(mean), max(mean)
# min_m2, max_m2 = min(m2), max(m2)
# min_fano, max_fano = min(fano[fano>0]), max(fano)

# n_plot = 100
# mean_vals = np.linspace(min_mean, max_mean, n_plot)
# m2_vals = np.linspace(min_m2, max_m2, n_plot)
# fano_vals = np.linspace(min_fano, max_fano, n_plot)

# # Compute error bars
# yerr = np.vstack([loss - wilson_lo, wilson_hi - loss])

# # FIGURE: LOSS VS MEAN---------------------------------------------------------
# # Set up figure
# fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
# divider = make_axes_locatable(ax_scatter)
# ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# # Compare model prediction with simulated loss
# sc = ax_scatter.scatter(mean, loss, c=m2, cmap='viridis', norm=LogNorm())
# cbar = fig.colorbar(sc, cax=ax_cbar)

# # Add Wilson confidence interval
# segments = np.stack([np.column_stack([mean, wilson_lo]), np.column_stack([mean, wilson_hi])], axis=1)
# colors = sc.cmap(sc.norm(m2))
# lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8, zorder=1)
# ax_scatter.add_collection(lc)

# # Pick values of the second moment for analytical curves
# n_curves = 3
# m2_curves = np.logspace(np.log10(min_m2), np.log10(max_m2), n_curves)
# cmap = sc.get_cmap()
# norm = sc.norm

# # for x in m2_curves:
# #     ax_scatter.plot(mean_vals, np.arccos((1+(eps**2)*(mean_vals/x))**(-1/2))/np.pi, c=cmap(norm(x)))

# cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
# ax_scatter.set_xscale('log')
# ax_scatter.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/loss_vs_mean_eta_{int(eta)}.pdf", dpi=600)

# # Add labels
# ax_scatter.set_xlabel('Average strength')
# ax_scatter.set_ylabel('Expected loss')
# cbar.set_label('Average squared strength')

# plt.savefig(f"../../figures/simulations/loss_vs_mean_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()


# # FIGURE: LOSS VS SECOND MOMENT------------------------------------------------
# # Set up figure
# fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
# divider = make_axes_locatable(ax_scatter)
# ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# # Compare model prediction with simulated loss
# sc = ax_scatter.scatter(m2, loss, c=mean, cmap='viridis', norm=LogNorm())
# cbar = fig.colorbar(sc, cax=ax_cbar)

# # Add Wilson confidence interval
# segments = np.stack([np.column_stack([m2, wilson_lo]), np.column_stack([m2, wilson_hi])], axis=1)
# colors = sc.cmap(sc.norm(mean))
# lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8, zorder=1)
# ax_scatter.add_collection(lc)

# # Pick values of the second moment for analytical curves
# n_curves = 3
# mean_curves = np.logspace(np.log10(min_mean), np.log10(max_mean), n_curves)
# cmap = sc.get_cmap()
# norm = sc.norm

# # for x in mean_curves:
# #     ax_scatter.plot(m2_vals, np.arccos((1+(eps**2)*(x/m2_vals))**(-1/2))/np.pi, c=cmap(norm(x)))

# cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
# ax_scatter.set_xscale('log')
# ax_scatter.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/loss_vs_m2_eta_{int(eta)}.pdf", dpi=600)

# # Add labels
# ax_scatter.set_xlabel('Average squared strength')
# ax_scatter.set_ylabel('Expected loss')
# cbar.set_label('Average strength')

# plt.savefig(f"../../figures/simulations/loss_vs_m2_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()


# # FIGURE: LOSS VS MEAN (BY FANO FACTOR)----------------------------------------
# # Set up figure
# fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
# divider = make_axes_locatable(ax_scatter)
# ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# # Compare model prediction with simulated loss
# sc = ax_scatter.scatter(mean, loss, c=fano, cmap='viridis', norm=LogNorm())
# cbar = fig.colorbar(sc, cax=ax_cbar)

# # Add Wilson confidence interval
# segments = np.stack([np.column_stack([mean, wilson_lo]), np.column_stack([mean, wilson_hi])], axis=1)
# colors = sc.cmap(sc.norm(fano))
# lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8, zorder=1)
# ax_scatter.add_collection(lc)

# # Pick values of the second moment for analytical curves
# n_curves = 3
# fano_curves = np.logspace(np.log10(min_fano), np.log10(max_fano), n_curves)
# cmap = sc.get_cmap()
# norm = sc.norm

# # for x in fano_curves:
# #     ax_scatter.plot(mean_vals, np.arccos((1+(eps**2)*(1./(x+mean_vals)))**(-1/2))/np.pi, c=cmap(norm(x)))

# cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
# ax_scatter.set_xscale('log')
# ax_scatter.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/loss_vs_mean_fano_eta_{int(eta)}.pdf", dpi=600)

# # Add labels
# ax_scatter.set_xlabel('Average strength')
# ax_scatter.set_ylabel('Expected loss')
# cbar.set_label('Fano factor')

# plt.savefig(f"../../figures/simulations/loss_vs_mean_fano_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()


# # FIGURE: LOSS VS FANO FACTOR--------------------------------------------------
# # Set up figure
# fig, ax_scatter = plt.subplots(figsize=(.9*width, .9*height))
# divider = make_axes_locatable(ax_scatter)
# ax_cbar = divider.append_axes('right', size='5%', pad=0.1)

# # Compare model prediction with simulated loss
# sc = ax_scatter.scatter(fano, loss, c=mean, cmap='viridis', norm=LogNorm())
# cbar = fig.colorbar(sc, cax=ax_cbar)

# # Add Wilson confidence interval
# segments = np.stack([np.column_stack([fano, wilson_lo]), np.column_stack([fano, wilson_hi])], axis=1)
# colors = sc.cmap(sc.norm(mean))
# lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8, zorder=1)
# ax_scatter.add_collection(lc)

# # Pick values of the second moment for analytical curves
# n_curves = 3
# mean_curves = np.logspace(np.log10(min_mean), np.log10(max_mean), n_curves)
# cmap = sc.get_cmap()
# norm = sc.norm

# # for x in mean_curves:
# #     ax_scatter.plot(fano_vals, np.arccos((1+(eps**2)*(1./(fano_vals+x)))**(-1/2))/np.pi, c=cmap(norm(x)))

# cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
# ax_scatter.set_xscale('log')
# ax_scatter.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/loss_vs_fano_eta_{int(eta)}.pdf", dpi=600)

# # Add labels
# ax_scatter.set_xlabel('Fano factor')
# ax_scatter.set_ylabel('Expected loss')
# cbar.set_label('Average strength')

# plt.savefig(f"../../figures/simulations/loss_vs_fano_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()

# #------------------------------------------------------------------------------
# # PREDICTED VS SIMULATED LOSS
# #------------------------------------------------------------------------------
# # Initialize loss and parameter vectors
# sim_loss = []
# pred_loss = []
# baseline_loss = []
# wilson_lo = []
# wilson_hi = []

# # Perturbation parameters
# eta = 1.
# eps_vals = [.5,1.,2.]

# for i, w in enumerate(tqdm(incoming_weights)):
#     # Get number of inputs
#     n_inputs = len(w)
#     if n_inputs > 0:
        
#         # Draw perturbations
#         w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
#         w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
        
#         # Draw inputs
#         x = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
        
#         # Save loss for different values of perturbation strenght
#         sim_loss_vals = []
#         pred_loss_vals = []
#         wilson_lo_vals = []
#         wilson_hi_vals = []
        
#         for eps in eps_vals:
#             l_hat = average_error(x,w,w_hat,eps)
#             ci = wilson_ci(l_hat, n_draws*n_perturb, 2.)
#             sim_loss_vals.append(l_hat)
#             pred_loss_vals.append(predicted_loss(w,eta,eps))
#             wilson_lo_vals.append(ci[0])
#             wilson_hi_vals.append(ci[1])
        
#         # Save loss and vector statistics
#         sim_loss.append(sim_loss_vals)
#         pred_loss.append(pred_loss_vals)
#         baseline_loss.append(predicted_loss(w,eta,1.))
#         wilson_lo.append(wilson_lo_vals)
#         wilson_hi.append(wilson_hi_vals)

# # Convert to numpy arrays
# sim_loss = np.array(sim_loss)
# pred_loss = np.array(pred_loss)
# baseline_loss = np.array(baseline_loss)
# wilson_lo = np.array(wilson_lo)
# wilson_hi = np.array(wilson_hi)

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
#     sc = ax_scatter.scatter(pred_loss[:,i], sim_loss[:,i], c=fano, cmap='viridis', norm=LogNorm())
#     cbar = fig.colorbar(sc, cax=ax_cbar)
    
#     # Add Wilson confidence interval
#     segments = np.stack([np.column_stack([pred_loss[:,i], wilson_lo[:,i]]), np.column_stack([pred_loss[:,i], wilson_hi[:,i]])], axis=1)
#     colors = sc.cmap(sc.norm(fano))
#     lc = LineCollection(segments, colors=colors, linewidths=0.8, alpha=0.8, zorder=1)
#     ax_scatter.add_collection(lc)
    
#     # Plot y=x line
#     ax_scatter.plot([min_loss,max_loss], [min_loss,max_loss], c='k', ls='--', lw=1)
    
#     cbar = fig.colorbar(sc, cax=ax_cbar, ax=ax_scatter)
#     ax_scatter.set_xscale('log')
#     ax_scatter.set_yscale('log')
#     plt.savefig(f"../../raw_figures/simulations/pred_vs_sim_loss_eta_{int(eta)}_eps_{i}.pdf", dpi=600)
    
#     # Add labels
#     ax_scatter.set_xlabel('Predicted loss')
#     ax_scatter.set_ylabel('Simulated loss')
#     cbar.set_label('Fano factor')
#     ax_scatter.text(max_loss/(max_loss/min_loss)**0.05, min_loss*(max_loss/min_loss)**0.05,
#                     f"$\epsilon$={eps}", ha="right", va="bottom")
    
#     plt.savefig(f"../../figures/simulations/pred_vs_sim_loss_eta_{int(eta)}_eps_{i}.pdf", dpi=600, bbox_inches='tight')
#     plt.show()
    

# #------------------------------------------------------------------------------
# # SIMULATION
# #------------------------------------------------------------------------------
# gaussian_loss = []
# binary_loss = []
# eta = 2.
# eps_vals = np.exp(np.log(10)*np.linspace(-2,2,20))      # normalized perturbation strength

# for i, w in enumerate(tqdm(incoming_weights)):
#     # Get number of inputs
#     n_inputs = len(w)
#     if n_inputs > 0:
#         Q = sensitivity(w,eta)
            
#         # Draw perturbations
#         w_hat = np.random.normal(0.,size=(n_inputs,n_perturb))
#         w_hat = w_hat*(w**(eta/2))[:,np.newaxis]
        
#         # Draw inputs
#         x_gaussian = np.random.normal(0.,size=(n_inputs,n_draws))
#         x_binary = np.random.choice([-1.,1.],size=(n_inputs,n_draws))
        
#         # Loop over epsilons
#         gaussian_loss_eps = []
#         binary_loss_eps = []
#         for eps in eps_vals:
#             gaussian_loss_eps.append(average_error(x_gaussian,w,w_hat,eps/(Q**(1/2))))
#             binary_loss_eps.append(average_error(x_binary,w,w_hat,eps/(Q**(1/2))))
            
#         gaussian_loss.append(gaussian_loss_eps)
#         binary_loss.append(binary_loss_eps)

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
# plt.savefig(f"../../figures/simulations/connectome_gaussian_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()

# plt.plot(eps_vals, median_binary, c=con_colors[0], label='Median')
# plt.fill_between(eps_vals, lower95_binary, upper95_binary, color=con_colors[0], alpha=0.3, label='95\% range')
# plt.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7, label='Prediction')
# plt.legend()
# plt.xlabel('Perturbation strength $\epsilon Q_{%s}^{1/2}$'%(int(eta)))
# plt.ylabel('Average loss ${\cal E}({\\bf w})$')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig(f"../../figures/simulations/connectome_binary_eta_{int(eta)}.pdf", dpi=600, bbox_inches='tight')
# plt.show()

# # Plot raw plots
# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(eps_vals, median_gaussian, c=con_colors[0])
# ax.fill_between(eps_vals, lower95_gaussian, upper95_gaussian, color=con_colors[0], alpha=0.3)
# ax.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/connectome_gaussian_eta_{int(eta)}.pdf", dpi=600)
# plt.show()

# fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# ax.plot(eps_vals, median_binary, c=con_colors[0])
# ax.fill_between(eps_vals, lower95_binary, upper95_binary, color=con_colors[0], alpha=0.3)
# ax.plot(eps_vals, np.arccos((1+(eps_vals**2))**(-1/2))/np.pi, c='k', ls='--', alpha=0.7)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.savefig(f"../../raw_figures/simulations/connectome_binary_eta_{int(eta)}.pdf", dpi=600)
# plt.show()
