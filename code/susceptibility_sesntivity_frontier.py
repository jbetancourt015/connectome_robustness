#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:25:12 2025

@author: jbet
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

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

con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

n_inputs = int(1e3)
n_samples = int(1e3)
r_vals = np.linspace(0.,1.,100)

def simulate_statistics(eta, n_inputs, n_samples, dist='uniform', gamma=2.):
    # Initialize statistics vectors
    q = []
    r = []
    for i in range(n_samples):
        if dist=='uniform':
            w = np.random.rand(n_inputs)
        elif dist=='pareto':
            w = np.random.rand(n_inputs)
            w = (1.-w)**(-1/(gamma-1))
        # Compute means
        w_1 = np.mean(w)
        w_2 = np.mean(w**2)
        w_eta = np.mean(w**eta)
        # Compute sesitivity and susceptibility 
        q.append((w_1**(2-eta))*w_eta/w_2)
        r.append(w_1/np.sqrt(w_2))
    return np.array(q), np.array(r)

def finite_frontier(r,eta,n_inputs):
    p = 1/n_inputs
    w1 = r + np.sqrt(r**2 + (n_inputs-1-n_inputs*(r**2)))
    w2 = np.sqrt((1.-p*(w1**2))/(1.-p))
    return (r**(2-eta))*(p*(w1**eta) + (1-p)*(w2**eta))

for eta in [.01, .1, .5, 1., 1.5, 1.8, 1.95, 1.99]:
    # Get statistics for both distributions
    q_uni, r_uni = simulate_statistics(eta, n_inputs, n_samples, dist='uniform')
    q_pareto, r_pareto = simulate_statistics(eta, n_inputs, n_samples, dist='pareto', gamma=2.5)
    
    plt.plot(r_vals, r_vals**(4-2*eta), c='k', lw=1, ls='--')
    plt.plot(r_vals, r_vals**2, c='k', lw=1, ls='--')
    plt.plot(r_vals[r_vals>np.sqrt(1/n_inputs)], finite_frontier(r_vals[r_vals>np.sqrt(1/n_inputs)],eta,n_inputs), ls='--', c='g')
    
    # Parametrized bound
    # t_vals = np.linspace(0.,1.,100)
    # r_param = 2*np.sqrt(t_vals*(1-t_vals))
    # q_param = (2**(2-eta))*(1-t_vals)*(t_vals**(1-eta))*(t_vals+((1-t_vals)**(1-eta))*(t_vals**eta))
    # plt.plot(r_param, q_param, c='k', lw=1, ls='--')
    
    plt.scatter(r_pareto, q_pareto, color=con_colors[1], alpha=.2, rasterized=True, label='Pareto $(\gamma = 2)$')
    plt.scatter(r_uni,q_uni, color=con_colors[0], alpha=.2, rasterized=True, label='Uniform')
    plt.text(1.,0.,'$\eta = %s$'%(eta), horizontalalignment='right', verticalalignment='bottom')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.xlabel('Susceptibility $r({\\bf w})$')
    plt.ylabel('Sensitivity $q({\\bf w})$')
    plt.show()

for eta in [.5, 1.5, 1.95]:
    # Draw the bound for various values of n
    plt.plot(r_vals, r_vals**(4-2*eta), c='k', lw=1, ls='--')
    plt.plot(r_vals, r_vals**2, c='k', lw=1, ls='--')
    for i, n in enumerate([1e1,1e2,1e3,1e4]):
        plt.plot(r_vals[r_vals>np.sqrt(1/n)], finite_frontier(r_vals[r_vals>np.sqrt(1/n)],eta,n), alpha=1.-i*.2, c=con_colors[0], label='$n=10^{%s}$'%(i+1))
    plt.text(1.,0.,'$\eta = %s$'%(eta), horizontalalignment='right', verticalalignment='bottom')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.xlabel('Susceptibility $r({\\bf w})$')
    plt.ylabel('Sensitivity $q({\\bf w})$')
    plt.show()
    
    
# Make raw figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

eta = 1.5
# Draw the bound for various values of n
ax.plot(r_vals, r_vals**(4-2*eta), c='k', lw=1, ls='--')
ax.plot(r_vals, r_vals**2, c='k', lw=1, ls='--')
for i, n in enumerate([1e1,1e2,1e3,1e4]):
    ax.plot(r_vals[r_vals>np.sqrt(1/n)], finite_frontier(r_vals[r_vals>np.sqrt(1/n)],eta,n), alpha=1.-i*.2, c=con_colors[0], label='$n=10^{%s}$'%(i+1))
plt.legend()
ax.set_aspect('equal')
plt.savefig(f"../raw_figures/susceptibility_frontier.pdf", dpi=600)
plt.show()