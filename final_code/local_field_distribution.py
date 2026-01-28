"""
    Master script for z and ztilde analysis: simulation, histogram, and cartoon.
-------------------------------------------------------------------------------
created on:
    Wed 22 Jan 2026
-------------------------------------------------------------------------------
last change:
    Wed 22 Jan 2026
-------------------------------------------------------------------------------
notes:
    This script combines functionality from:
    - z_ztilde_simulation.py: Computes z and ztilde values for gamma-distributed weights
    - local_field_hist.py: Generates 2D heatmap histograms of normalized local fields
    - ellipse_rho.py: Generates ellipse figures for correlation visualization
    
    Uses a single consolidated parquet file with parameter columns.
    Checks for existing parameter combinations before running new simulations.
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import logging

#------------------------------------------------------------------------------
# MATPLOTLIB CONFIGURATION
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# SHARED CONFIGURATION
#------------------------------------------------------------------------------
sim_dir = '../simulation_results/'
fig_dir = '../paper_figures/framework/'

# Single consolidated simulation file
sim_file = sim_dir + "z_ztilde_simulations.parquet"

# Parameter sets (mean, second_moment)
param_sets = [
    (1.0, 100.0),
    (1.0, 1.0),
    (1.0, 400.0),
]

# Simulation parameters
n_inputs = int(1e4)
eta = 1.0
eps = 10.0
n_draws = int(1e3)
n_perturb = int(1e3)

# Figure parameters
width = 1.5
height = 1.5
alpha_min = 0.2
n_bins = 20

# Plotting colors
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                       [120, 0, 150], [0, 168, 157], [203, 41, 123], [153, 153, 0]]) / 255

# Colors for each parameter set
colors = [con_colors[2], con_colors[1], con_colors[4]]

#------------------------------------------------------------------------------
# SIMULATION FUNCTIONS
#------------------------------------------------------------------------------
def compute_z_ztilde(
    mean,
    var,
    n_inputs,
    eta,
    eps,
    n_draws,
    n_perturb,
    rng=None,
):
    """
    Compute z and ztilde values for a single gamma-distributed weight vector.
    
    Parameters
    ----------
    mean : float
        Mean of the gamma distribution for weights.
    var : float
        Variance of the gamma distribution for weights.
    n_inputs : int
        Number of input dimensions (size of weight vector).
    eta : float
        Noise scaling exponent.
    eps : float
        Perturbation magnitude.
    n_draws : int
        Number of input draws (x instances).
    n_perturb : int
        Number of perturbation draws (w_hat instances).
    rng : numpy.random.Generator, optional
        Random number generator. If None, creates a new one.
    
    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        - mean: Mean parameter
        - second_moment: Second moment parameter
        - draw_idx: Index of the input x instance (0 to n_draws-1)
        - perturb_idx: Index of the perturbation instance (0 to n_perturb-1)
        - z: Baseline output value
        - ztilde: Perturbed output value
    w : np.ndarray
        The generated weight vector (for reference).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate gamma weights
    # Gamma distribution: mean = alpha * theta, var = alpha * theta^2
    # Solving: theta = var / mean, alpha = mean / theta = mean^2 / var
    theta = var / mean
    alpha = mean / theta
    w = rng.gamma(alpha, theta, n_inputs)

    # Draw inputs: n_inputs × n_draws
    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))

    # Draw base Gaussian noise: n_inputs × n_perturb
    base_noise = rng.normal(0.0, 1.0, size=(n_inputs, n_perturb))
    # Scale by w**(eta/2)
    w_hat = base_noise * (w**(eta / 2.0))[:, None]

    # Baseline output: shape (n_draws,)
    z = w @ x

    # Perturbed output: shape (n_perturb, n_draws)
    # ztilde[j, i] = z[i] + eps * (w_hat[:, j].T @ x[:, i])
    delta = eps * (w_hat.T @ x)  # (n_perturb, n_draws)
    ztilde = z[None, :] + delta  # (n_perturb, n_draws)

    # Build long-form DataFrame
    # Create index arrays
    draw_idx = np.tile(np.arange(n_draws), n_perturb)
    perturb_idx = np.repeat(np.arange(n_perturb), n_draws)
    z_flat = np.tile(z, n_perturb)
    ztilde_flat = ztilde.ravel()

    df = pd.DataFrame({
        'mean': mean,
        'variance': var,
        'draw_idx': draw_idx,
        'perturb_idx': perturb_idx,
        'z': z_flat,
        'ztilde': ztilde_flat,
    })

    return df, w


def get_missing_params(param_sets, existing_df=None):
    """
    Return parameter sets not already in the file.
    
    Parameters
    ----------
    param_sets : list of tuples
        List of (mean, second_moment) tuples.
    existing_df : pd.DataFrame, optional
        Existing simulation data. If None, all params are missing.
    
    Returns
    -------
    list of tuples
        Parameter sets that need to be simulated.
    """
    if existing_df is None:
        return param_sets
    existing = set(zip(existing_df['mean'], existing_df['variance']))
    return [(m, s) for m, s in param_sets if (m, s) not in existing]


#------------------------------------------------------------------------------
# HISTOGRAM FUNCTIONS
#------------------------------------------------------------------------------
def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    """Create a colormap that fades from transparent to a solid color."""
    bottom = (*rgb, alpha_min)
    top = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)


def load_and_normalize(df_all, mean, var):
    """
    Filter and normalize data for a specific parameter set.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        Full DataFrame with all simulations.
    mean : float
        Mean parameter to filter by.
    var : float
        Variance parameter to filter by.
    
    Returns
    -------
    z_norm : np.ndarray
        Normalized baseline local field values (z / sigma_z).
    ztilde_norm : np.ndarray
        Normalized perturbed local field values (ztilde / sigma_ztilde).
    """
    df = df_all[(df_all['mean'] == mean) & (df_all['variance'] == var)]
    
    z = df['z'].values
    ztilde = df['ztilde'].values
    
    print('Correlation:', np.corrcoef(z, ztilde)[0, 1])
    
    # Normalize by standard deviation
    z_norm = z / np.std(z)
    ztilde_norm = ztilde / np.std(ztilde)
    
    return z_norm, ztilde_norm


def plot_local_field_hist(z_norm, ztilde_norm, color, fname):
    """
    Create a 2D heatmap histogram of normalized local field values.
    
    Parameters
    ----------
    z_norm : np.ndarray
        Normalized baseline local field values.
    ztilde_norm : np.ndarray
        Normalized perturbed local field values.
    color : np.ndarray
        RGB color array for the colormap.
    fname : str
        Filename suffix for saving.
    """
    # Determine plot limits
    data_min = -3.
    data_max = 3.
    
    # Set up figure
    fig = plt.figure(figsize=(width, height))
    ax_scatter = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    # Create histogram bins
    xbins = np.linspace(data_min, data_max, n_bins)
    ybins = xbins.copy()
    
    # Create 2D histogram
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
    
    # Draw reference lines through origin
    ax_scatter.plot([0., 0.], [data_min, data_max], c="k", alpha=.5, lw=1)
    ax_scatter.plot([data_min, data_max], [0., 0.], c="k", alpha=.5, lw=1)
    
    # Use the same locator for x and y so ticks coincide
    locator = ax_scatter.yaxis.get_major_locator()
    ax_scatter.xaxis.set_major_locator(locator)
    
    # Set equal aspect ratio
    ax_scatter.set_aspect('equal')
    
    # Save raw figure
    plt.savefig(f"../../paper_figures/framework/hist_{fname}.pdf", dpi=600)
    
    # Add axis labels
    ax_scatter.set_xlabel(r'Local field $z/\sigma_z$')
    ax_scatter.set_ylabel(r'Perturbed local field $\tilde{z}/\sigma_{\tilde{z}}$')
    
    plt.show()


#------------------------------------------------------------------------------
# CARTOON FUNCTIONS
#------------------------------------------------------------------------------
def draw_ellipse(R, rho, ax, color, lim=3.):
    """
    Draw an ellipse satisfying x^2 + y^2 - 2*rho*x*y = R^2*sqrt(1 - rho^2).
    
    Parameters
    ----------
    R : float
        Radius scaling factor.
    rho : float
        Correlation parameter controlling ellipse shape.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    color : np.ndarray
        RGB color array.
    lim : float
        Plot limits.
    """
    # Create meshgrid
    xx = np.linspace(-lim, lim, 400)
    yy = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(xx, yy)

    # Evaluate the left-hand side of the equation: x^2 + y^2 - 2*rho*x*y
    Z = X**2 + Y**2 - 2*rho*X*Y

    # The right-hand side is R^2*sqrt(1 - rho^2)
    level = (R**2)*np.sqrt(1 - rho**2)

    # Fill the ellipse interior (where Z <= level)
    ax.contourf(X, Y, Z, levels=[0, level], colors=[color], alpha=0.3)

    # Draw the ellipse outline
    ax.contour(X, Y, Z, levels=[level], colors=[color], linewidths=2)


def plot_ellipse_cartoon(mean, var, color, fname, lim=3., R=1.5):
    """
    Plot a single ellipse cartoon for given parameters.
    
    Parameters
    ----------
    mean : float
        Mean parameter.
    var : float
        Variance parameter.
    color : np.ndarray
        RGB color array.
    lim : float
        Plot limits.
    R : float
        Radius scaling factor.
    """
    # Compute rho from parameters
    rho = (1. + (eps**2) * mean / (var + mean**2)) ** (-0.5)
    
    # Set up figure
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    
    # Draw coordinate axes (lines through origin)
    ax.plot([-lim, lim], [0., 0.], c='k', alpha=.5, lw=1)
    ax.plot([0., 0.], [-lim, lim], c='k', alpha=.5, lw=1)
    
    # Draw ellipse
    draw_ellipse(R, rho, ax, color, lim)
    
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    # Use the same locator for x and y so ticks coincide
    locator = ax.yaxis.get_major_locator()
    ax.xaxis.set_major_locator(locator)
    
    # Save raw figure
    plt.savefig(f"../../paper_figures/framework/ellipse_{fname}.pdf", dpi=600)
    
    # Labels
    ax.set_xlabel('Local field $z/\sigma_z$')
    ax.set_ylabel('Perturbed local field $\\tilde{z}/\sigma_{\\tilde{z}}$')
    
    plt.show()


#------------------------------------------------------------------------------
# MAIN EXECUTION
#------------------------------------------------------------------------------
rng = np.random.default_rng(1764)

#--------------------------------------------------------------------------
# 1. SIMULATION: Load existing or run missing simulations
#--------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: SIMULATION")
print("=" * 60)

if os.path.exists(sim_file):
    df_all = pd.read_parquet(sim_file)
    print(f"Loaded existing file with {len(df_all)} rows")
else:
    df_all = None
    print("No existing simulation file found")

# Check which parameter sets need to be simulated
missing = get_missing_params(param_sets, df_all)

if missing:
    print(f"Running simulations for {len(missing)} parameter sets...")
    new_dfs = []
    for mean, var in missing:
        print(f"  Simulating mean={mean}, variance={var}...")
        df, w = compute_z_ztilde(
            mean=mean,
            var=var,
            n_inputs=n_inputs,
            eta=eta,
            eps=eps,
            n_draws=n_draws,
            n_perturb=n_perturb,
            rng=rng,
        )
        new_dfs.append(df)
        print(f"    Generated {len(df)} rows")
    
    # Concatenate new results
    df_new = pd.concat(new_dfs, ignore_index=True)
    
    # Merge with existing data
    if df_all is not None:
        df_all = pd.concat([df_all, df_new], ignore_index=True)
    else:
        df_all = df_new
    
    # Save to file
    df_all.to_parquet(sim_file)
    print(f"Saved {len(df_all)} total rows to {sim_file}")
else:
    print("All parameter sets already simulated, skipping simulation step")

#--------------------------------------------------------------------------
# 2. HISTOGRAMS: Generate 2D heatmap histograms
#--------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: HISTOGRAMS")
print("=" * 60)

for i, (mean, var) in enumerate(param_sets):
    print(f"Processing histogram for mean={mean}, variance={var}...")
    
    # Load and normalize data
    z_norm, ztilde_norm = load_and_normalize(df_all, mean, var)
    
    # Plot histogram
    plot_local_field_hist(z_norm, ztilde_norm, color=colors[i], fname=f"{i}")
    
    print(f"  Generated histogram")

#--------------------------------------------------------------------------
# 3. CARTOONS: Generate ellipse cartoons
#--------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: ELLIPSE CARTOONS")
print("=" * 60)

for i, (mean, var) in enumerate(param_sets):
    print(f"Generating ellipse cartoon for mean={mean}, variance={var}...")
    
    # Plot ellipse cartoon
    plot_ellipse_cartoon(mean, var, colors[i], fname=f"{i}")
    
    print(f"  Generated cartoon")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

