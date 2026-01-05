# %%
"""
    Visualization of parametric distribution simulation results.
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Path to simulation results
sim_dir = '../simulation_results/'

# Number of inputs values to plot
n_inputs_values = [10, 100, 1000]

# %%
# -----------------------------------------------------------------------------
# LOGNORMAL PLOTS
# -----------------------------------------------------------------------------
fig_ln, axes_ln = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig_ln.suptitle('Lognormal Simulations: Loss vs Mean (colored by Variance)', fontsize=14)

for idx, n_inputs in enumerate(n_inputs_values):
    try:
        df = pd.read_parquet(sim_dir + f"lognormal_sim_{n_inputs}.parquet")
        
        # Get unique variances for colormap
        unique_vars = np.sort(df['var'].unique())
        
        # Create scatter plot with color by variance
        scatter = axes_ln[idx].scatter(
            df['mean'], 
            df['sim_loss'], 
            c=df['var'], 
            cmap='viridis',
            norm=LogNorm(vmin=unique_vars.min(), vmax=unique_vars.max()),
            s=60,
            edgecolors='white',
            linewidth=0.5,
            alpha=0.8
        )
        
        axes_ln[idx].set_xlabel('Input Mean', fontsize=11)
        axes_ln[idx].set_title(f'n_inputs = {n_inputs}', fontsize=12)
        
    except FileNotFoundError:
        axes_ln[idx].text(0.5, 0.5, f'No data for\nn_inputs={n_inputs}', 
                          ha='center', va='center', transform=axes_ln[idx].transAxes)
        axes_ln[idx].set_title(f'n_inputs = {n_inputs}', fontsize=12)

axes_ln[0].set_ylabel('Simulated Loss', fontsize=11)

# Add colorbar
cbar_ln = fig_ln.colorbar(scatter, ax=axes_ln, label='Variance', shrink=0.8)
plt.tight_layout()
plt.show()

# %%
# -----------------------------------------------------------------------------
# LOMAX PLOTS
# -----------------------------------------------------------------------------
fig_lx, axes_lx = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
fig_lx.suptitle('Lomax Simulations: Loss vs Mean (colored by Variance)', fontsize=14)

for idx, n_inputs in enumerate(n_inputs_values):
    try:
        df = pd.read_parquet(sim_dir + f"lomax_sim_{n_inputs}.parquet")
        
        # Get unique variances for colormap
        unique_vars = np.sort(df['var'].unique())
        
        # Create scatter plot with color by variance
        scatter = axes_lx[idx].scatter(
            df['mean'], 
            df['sim_loss'], 
            c=df['var'], 
            cmap='plasma',
            norm=LogNorm(vmin=unique_vars.min(), vmax=unique_vars.max()),
            s=60,
            edgecolors='white',
            linewidth=0.5,
            alpha=0.8
        )
        
        axes_lx[idx].set_xlabel('Input Mean', fontsize=11)
        axes_lx[idx].set_title(f'n_inputs = {n_inputs}', fontsize=12)
        
    except FileNotFoundError:
        axes_lx[idx].text(0.5, 0.5, f'No data for\nn_inputs={n_inputs}', 
                          ha='center', va='center', transform=axes_lx[idx].transAxes)
        axes_lx[idx].set_title(f'n_inputs = {n_inputs}', fontsize=12)

axes_lx[0].set_ylabel('Simulated Loss', fontsize=11)

# Add colorbar
cbar_lx = fig_lx.colorbar(scatter, ax=axes_lx, label='Variance', shrink=0.8)
plt.tight_layout()
plt.show()
