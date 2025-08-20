"""
    This script calculates statistics of single-neuron sensitivities
-------------------------------------------------------------------------------
created on:
    Mon 24 Mar 2025
-------------------------------------------------------------------------------
last change:
    Sun 27 Apr 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import processing
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit
from scipy import spatial
from scipy import stats
from scipy.optimize import root
from tqdm import tqdm
from poisson_var_cdf import cdf_mc, cdf_exact

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "font.serif": ["Garamond"],
})

# Connectome list
data_files = ['Drosophila_central_brain','Drosophila_optic_medulla','Celegans',
              'Platynereis_sensory_motor', 'Mouse_retina']


# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

data_labels = ['$\it{Drosophila}$ central brain', '$\it{Drosophila}$ optic medulla',
               '$\it{C. elegans}$', '$\it{Platyneris}$ sensory motor', 'Mouse retina']


# #------------------------------------------------------------------------------
# # GET NETWORK STATISTICS
# #------------------------------------------------------------------------------
# # Get vectors of statistics
# fano_vals = []
# cv_vals = []
# dir_vals = []

# for data_idx in np.arange(5):
#     # Get network
#     G = processing.process_data(data_files[data_idx])
#     A = nx.to_numpy_array(G)
#     # Calculate moments
#     S0 = np.sum(A>0, axis=0)
#     S = [np.sum(A**i if i>0 else A>0,axis=0)[(S0>1)] for i in range(3)]
#     mean = S[1]/S[0]
#     var = S[2]/S[0] - mean**2
#     # Compute Fano factor
#     fano = (var+1)/mean
#     cv = var/(mean**2)
#     fano_vals.append(fano)
#     cv_vals.append(cv)
#     # Restrict set of neurons
#     non_triv = mean>1
#     dirichlet = 2*var[non_triv] / ((S[0][non_triv]-1)*((mean[non_triv]-1)**2)/2)
#     dir_vals.append(dirichlet)

# # First attempt: kde
# for data_idx in np.arange(5):
#     # Plot kernel density
#     kde = stats.gaussian_kde(fano_vals[data_idx])
#     x_eval = np.linspace(np.min(fano_vals[data_idx]), np.max(fano_vals[data_idx]), num=200)
#     plt.plot(x_eval, kde(x_eval), c=con_colors[data_idx], label=data_labels[data_idx])
    
# plt.legend(fontsize=12)
# plt.xlabel('Fano factor')
# plt.ylabel('Density')
# plt.xscale('log')
# plt.yscale('log')
# plt.axvline(x=1., color='k', ls='--', lw=1)
# plt.show()

# # Second attempt: 1-CDF
# @njit
# def compute_cdf(data, x_eval):
#     data_sort = np.copy(data)
#     np.sort(data_sort)
#     # Iteratively construct PDF
#     n_eval = len(x_eval)
#     n_data = len(data)
#     pdf = np.zeros(n_eval) # Note that this is not strictly a PDF, since it does not account for bin size
#     max_val = np.min(data)
#     max_idx = 0
#     for i, x in enumerate(x_eval):
#         mass = 0.
#         while (x >= max_val) and (max_idx < n_data):
#             mass += 1/n_data
#             max_idx += 1
#             max_val = data[max_idx]
#         pdf[i] = mass
#     return np.cumsum(pdf)

# #------------------------------------------------------------------------------
# # PLOT DISTRIBUTIONS
# #------------------------------------------------------------------------------
# for data_idx in np.arange(5):
#     # Plot kernel density
#     x_eval = np.linspace(np.min(fano_vals[data_idx]), np.max(fano_vals[data_idx]), num=200)
#     cdf = compute_cdf(fano_vals[data_idx], x_eval)
#     plt.plot(x_eval, cdf, c=con_colors[data_idx], label=data_labels[data_idx])

# plt.legend(fontsize=12)
# plt.xlabel('Shifted Fano factor')
# plt.ylabel('CDF')
# plt.xscale('log')
# plt.yscale('log')
# plt.axvline(x=1., color='k', ls='--', lw=1)
# plt.show()

# for data_idx in np.arange(5):
#     # Plot kernel density
#     x_eval = np.linspace(np.min(cv_vals[data_idx]), np.max(cv_vals[data_idx]), num=200)
#     cdf = compute_cdf(cv_vals[data_idx], x_eval)
#     plt.plot(x_eval, cdf, c=con_colors[data_idx], label=data_labels[data_idx])

# plt.legend(fontsize=12)
# plt.xlabel('Coefficient of Variation')
# plt.ylabel('CDF')
# plt.xscale('log')
# plt.yscale('log')
# plt.axvline(x=1., color='k', ls='--', lw=1)
# plt.show()


# for data_idx in np.arange(5):
#     # Plot kernel density
#     x_eval = np.linspace(np.min(dir_vals[data_idx]), np.max(dir_vals[data_idx]), num=200)
#     cdf = compute_cdf(dir_vals[data_idx], x_eval)
#     plt.plot(x_eval, cdf, c=con_colors[data_idx], label=data_labels[data_idx])

# plt.legend(fontsize=12)
# plt.xlabel('Dirichlet Statistic')
# plt.ylabel('CDF')
# plt.xscale('log')
# plt.yscale('log')
# plt.axvline(x=1., color='k', ls='--', lw=1)
# plt.show()

#------------------------------------------------------------------------------
# GET AVERAGE CDF
#------------------------------------------------------------------------------
# Get vector of scale values
n_psi = 3
scale_vals = np.linspace(0.5,1.5,n_psi)

# Get a vector of Psi for each connectome
psi_vecs = []

for data_idx in [1,2,3]:
    # Get network
    G = processing.process_data(data_files[data_idx])
    A = nx.to_numpy_array(G)
    # Calculate moments
    S0 = np.sum(A>0, axis=0)
    S = [np.sum(A**i if i>0 else A>0,axis=0)[(S0>1)] for i in range(3)]
    mean = S[1]/S[0]
    var = S[2]/S[0] - mean**2
    # Compute Psi values
    n_neurons = np.sum(S0>1)
    psi = np.zeros(n_psi)
    for i in tqdm(range(n_neurons)):
        if var[i] > 0:
            degree = S[0][i]
            # print('Degree:', degree)
            if degree < 8:
                for j, s in enumerate(scale_vals):
                    psi[j] += cdf_exact(degree, mean[i]-1, s*var[i])/n_neurons
            else:
                for j, s in enumerate(scale_vals):
                    psi[j] += cdf_mc(degree, mean[i]-1, s*var[i])/n_neurons
        else:
            psi += 1./n_neurons
    # Store Psi vector
    psi_vecs.append(1.-psi)
    
#------------------------------------------------------------------------------
# PLOT PSI STATISTIC
#------------------------------------------------------------------------------
for i, data_idx in enumerate([1,2,3]):
    # Plot kernel density
    plt.plot(scale_vals, psi_vecs[i], c=con_colors[data_idx], label=data_labels[data_idx])

plt.legend(fontsize=12)
plt.xlabel('Scale factor $s$')
plt.ylabel('Prob. low sensitivity $\Psi(s)$')
plt.axvline(x=1., color='k', ls='--', lw=1)
plt.show()
