"""
    This script generates 2D heatmap histograms of normalized local field values
    (z/sigma_z vs ztilde/sigma_ztilde) from simulation data.
-------------------------------------------------------------------------------
created on:
    Mon 12 Jan 2026
-------------------------------------------------------------------------------
last change:
    Mon 12 Jan 2026
-------------------------------------------------------------------------------
notes:
    Combines plotting style from robustness_comparison.py with axis labels
    from ellipse_rho.py. Data comes from z_ztilde parquet files.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import logging

plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'font.family': 'Helvetica',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

mpl.rcParams['figure.dpi'] = 300

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)

# Nature figure size
width = 2.
height = 2.

alpha_min = 0.2
n_bins = 20

# Data directory
sim_dir = '../simulation_results/'
fig_dir = '../paper_figures/local_field_hist/'

# File names
fnames = ['baseline', 'high_w', 'high_w2']

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                       [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]]) / 255


def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    """Create a colormap that fades from transparent to a solid color."""
    bottom = (*rgb, alpha_min)
    top = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)


def load_and_normalize(fname):
    """
    Load parquet file and normalize z and ztilde by their standard deviations.
    
    Parameters
    ----------
    fname : str
        Base filename (without path or extension).
    
    Returns
    -------
    z_norm : np.ndarray
        Normalized baseline local field values (z / sigma_z).
    ztilde_norm : np.ndarray
        Normalized perturbed local field values (ztilde / sigma_ztilde).
    """
    df = pd.read_parquet(sim_dir + f"z_ztilde_{fname}.parquet")
    
    z = df['z'].values
    ztilde = df['ztilde'].values
    
    print('Correlation:',np.corrcoef(z, ztilde))
    
    # Normalize by standard deviation
    z_norm = z / np.std(z)
    ztilde_norm = ztilde / np.std(ztilde)
    
    return z_norm, ztilde_norm


def plot_local_field_hist(z_norm, ztilde_norm, color, fname, labels=True):
    """
    Create a 2D heatmap histogram of normalized local field values.
    
    Parameters
    ----------
    z_norm : np.ndarray
        Normalized baseline local field values.
    ztilde_norm : np.ndarray
        Normalized perturbed local field values.
    color_idx : int
        Index into con_colors for the colormap.
    fname : str
        Filename suffix for saving.
    labels : bool
        Whether to add axis labels.
    """
    # Determine plot limits
    data_min = -3.
    data_max = 3.
    
    # Set up figure
    fig, ax_scatter = plt.subplots(figsize=(width, height))

    # Create histogram bins
    xbins = np.linspace(data_min, data_max, n_bins)
    ybins = xbins.copy()
    
    # Create 2D histogramo
    counts, xedges, yedges, im = ax_scatter.hist2d(
        z_norm, ztilde_norm,
        bins=[xbins, ybins],
        density=True,
        cmap=fade_to_color_cmap(color, alpha_min=0.)
    )
    
    # Enable all four spines
    for loc in ["left", "right", "top", "bottom"]:
        ax_scatter.spines[loc].set_visible(True)
    ax_scatter.set_frame_on(True)
    
    # Draw diagonal reference line
    ax_scatter.plot([0., 0.], [data_min, data_max], "--", c="k", lw=1)
    ax_scatter.plot([data_min, data_max], [0., 0.], "--", c="k", lw=1)
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Set equal aspect ratio
    ax_scatter.set_aspect('equal')
    
    # Save plot without labels
    # plt.savefig(fig_dir + f"{fname}_hist.pdf", dpi=600)
    
    if labels:
        # Add axis labels
        ax_scatter.set_xlabel(r'Local field $z/\sigma_z$')
        ax_scatter.set_ylabel(r'Perturbed local field $\tilde{z}/\sigma_{\tilde{z}}$')
    
    plt.show()


#------------------------------------------------------------------------------
# GENERATE HISTOGRAMS
#------------------------------------------------------------------------------
colors = [con_colors[2], con_colors[1], con_colors[4]]

for i, fname in enumerate(fnames):
    print(f"Processing {fname}...")
    
    # Load and normalize data
    z_norm, ztilde_norm = load_and_normalize(fname)
    
    # Plot histogram
    plot_local_field_hist(z_norm, ztilde_norm, color=colors[i], fname=fname)
    
    print(f"  Saved {fname}_hist.pdf")

