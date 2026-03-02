"""
    Master script for framework figures: cartoons, local-field simulations,
    histograms, ellipse cartoons, Gaussian heatmaps, and parametric loss plots.
-------------------------------------------------------------------------------
created on:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
last change:
    Fri 6 Feb 2026
-------------------------------------------------------------------------------
notes:
    This script consolidates:
    - framework_cartoons.py: Classification plane, 1D local field, 2D
      misclassification cartoons
    - local_field_distribution.py: z/ztilde simulation, 2D histograms,
      ellipse cartoons, Gaussian heatmaps
    - parametric_simulations.py: Parametric distribution loss plots

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
import matplotlib.colors as mcolors
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
con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0],
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

# Directory paths
sim_dir = '../simulation_results/'
fig_dir = '../../paper_figures/framework/'

# Single consolidated simulation file for z/ztilde
sim_file = sim_dir + "z_ztilde_simulations.parquet"

# Figure sizes (mm -> inches)
mm_to_in = 25.4
width_sm = 32. / mm_to_in
height_sm = 32. / mm_to_in
width_md = 53. / mm_to_in
height_md = 53. / mm_to_in
width_lg = 65. / mm_to_in
height_lg = 65. / mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins_sm = dict(left=0.22, right=0.95, bottom=0.22, top=0.95)
fig_margins_md = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
fig_margins_lg = dict(left=0.12, right=0.95, bottom=0.12, top=0.95)

# Dark-cool colormap for parametric loss plots
dark_cool = mcolors.LinearSegmentedColormap.from_list(
    "dark_cool",
    ["#0b3c49", "#1f5c8b", "#4a4aa8", "#7a3fa0", "#9b2f8a"]
)
if "dark_cool" not in plt.colormaps:
    plt.colormaps.register(dark_cool)

# z/ztilde simulation parameters
# Parameter sets: (distribution, mean) or (distribution, mean, variance) for gamma
# Supported distributions: 'dirac', 'poisson', 'pareto', 'gamma'
# - dirac: all weights equal to mean (variance = 0)
# - poisson: weights drawn from Poisson(mean) (variance = mean)
# - pareto: weights drawn from Pareto with x_m=1, shape derived from mean (requires mean > 1)
# - gamma: weights drawn from Gamma with specified mean and variance
param_sets = [
    ('gamma', 1., 1.),
    ('gamma', 1., 100.),
    ('gamma', 1., 400.),
]

n_inputs = int(1e4)
eta = 1.0
eps = 10.0
n_draws = int(1e3)
n_perturb = int(1e3)

# Histogram parameters
alpha_min = 0.2
n_bins = 20

# Parametric loss parameters
n_pred = 200

# Colors for each parameter set (equally spaced from viridis)
n_colors = len(param_sets)
cmap = plt.get_cmap('viridis')
sim_colors = [plt.cm.viridis(i / (n_colors - 1))[:3] for i in range(n_colors)]


#==============================================================================
# SHARED HELPER FUNCTIONS
#==============================================================================
def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    """Create a colormap that fades from transparent to a solid color."""
    bottom = (*rgb, alpha_min)
    top = (*rgb, 1.0)
    return LinearSegmentedColormap.from_list(name, [bottom, top], N=256)


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


def draw_principal_axes(R, rho, ax, color_pos, color_neg, lw=1):
    """
    Draw dashed principal axes (y=x and y=-x) within ellipse bounds.

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
    lw : float
        Line width.
    """
    # For y=x: intersection at x where 2x^2(1-rho) = R^2*sqrt(1-rho^2)
    # For y=-x: intersection at x where 2x^2(1+rho) = R^2*sqrt(1-rho^2)
    level = R**2 * np.sqrt(1 - rho**2)
    x_pos = np.sqrt(level / (2*(1 - rho)))  # y=x line
    x_neg = np.sqrt(level / (2*(1 + rho)))  # y=-x line
    
    ax.annotate("", xytext=(0, 0), xy=(x_pos, x_pos), arrowprops=dict(color=color_pos, width=1, headwidth=5))
    ax.annotate("", xytext=(0, 0), xy=(x_neg, -x_neg), arrowprops=dict(color=color_neg, width=1, headwidth=5))

    ax.plot([0., x_pos], [0., x_pos], c=color_pos, lw=lw)
    ax.plot([0., x_neg], [0., -x_neg], c=color_neg, lw=lw)


def general_loss(mean, var):
    """Predicted loss from mean and variance."""
    rob = np.sqrt(mean + var/mean)
    return (1/np.pi)*np.arccos((1.+1./rob**2)**(-1/2))


#==============================================================================
# SIMULATION FUNCTIONS (z/ztilde)
#==============================================================================
def generate_weights(distribution, mean, n_inputs, rng, var=None):
    """
    Generate weights from the specified distribution.

    Parameters
    ----------
    distribution : str
        Distribution type: 'dirac', 'poisson', 'pareto', or 'gamma'.
    mean : float
        Mean of the distribution.
    n_inputs : int
        Number of weights to generate.
    rng : numpy.random.Generator
        Random number generator.
    var : float, optional
        Variance (required only for 'gamma').

    Returns
    -------
    w : np.ndarray
        Generated weight vector.
    computed_var : float
        The variance of the distribution (computed or provided).
    """
    if distribution == 'dirac':
        # All weights equal to mean
        w = np.full(n_inputs, mean)
        computed_var = 0.0

    elif distribution == 'poisson':
        # Poisson distribution: variance = mean
        w = rng.poisson(mean, n_inputs).astype(float)
        computed_var = mean

    elif distribution == 'pareto':
        # Pareto distribution with x_m = 1 (cutoff/scale = 1)
        # Mean = alpha / (alpha - 1) for alpha > 1
        # Solving for alpha: alpha = mean / (mean - 1)
        if mean <= 1:
            raise ValueError(f"Pareto distribution requires mean > 1, got {mean}")
        alpha = mean / (mean - 1)
        # numpy.random.pareto generates from Pareto Type I with x_m=1
        # The distribution is (x_m / x)^alpha, so samples are >= 1
        # Actually numpy's pareto(a) generates y where (1+y) follows Pareto(a,1)
        # So we need to add 1 to get the standard Pareto with x_m=1
        w = (rng.pareto(alpha, n_inputs) + 1)
        # Scale to achieve the desired mean
        # E[rng.pareto(alpha) + 1] = 1/(alpha-1) + 1 = alpha/(alpha-1) = mean
        # So no additional scaling needed

        # Compute variance: Var = x_m^2 * alpha / ((alpha-1)^2 * (alpha-2)) for alpha > 2
        if alpha > 2:
            computed_var = alpha / ((alpha - 1)**2 * (alpha - 2))
        else:
            # Variance is infinite for alpha <= 2 (mean <= 2)
            computed_var = np.inf

    elif distribution == 'gamma':
        if var is None:
            raise ValueError("Gamma distribution requires variance parameter")
        # Gamma distribution: mean = alpha * theta, var = alpha * theta^2
        # Solving: theta = var / mean, alpha = mean / theta = mean^2 / var
        theta = var / mean
        alpha = mean / theta
        w = rng.gamma(alpha, theta, n_inputs)
        computed_var = var

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return w, computed_var


def compute_z_ztilde(
    distribution,
    mean,
    n_inputs,
    eta,
    eps,
    n_draws,
    n_perturb,
    rng=None,
    var=None,
):
    """
    Compute z and ztilde values for a weight vector drawn from the specified distribution.

    Parameters
    ----------
    distribution : str
        Distribution type: 'dirac', 'poisson', 'pareto', or 'gamma'.
    mean : float
        Mean of the distribution for weights.
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
    var : float, optional
        Variance (required only for 'gamma' distribution).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with columns:
        - distribution: Distribution name
        - mean: Mean parameter
        - variance: Variance of the distribution
        - draw_idx: Index of the input x instance (0 to n_draws-1)
        - perturb_idx: Index of the perturbation instance (0 to n_perturb-1)
        - z: Baseline output value
        - ztilde: Perturbed output value
    w : np.ndarray
        The generated weight vector (for reference).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate weights from specified distribution
    w, computed_var = generate_weights(distribution, mean, n_inputs, rng, var)

    # Draw inputs: n_inputs x n_draws
    x = rng.choice([-1.0, 1.0], size=(n_inputs, n_draws))

    # Draw base Gaussian noise: n_inputs x n_perturb
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
        'distribution': distribution,
        'mean': mean,
        'variance': computed_var,
        'draw_idx': draw_idx,
        'perturb_idx': perturb_idx,
        'z': z_flat,
        'ztilde': ztilde_flat,
    })

    return df, w


def parse_param_set(param_set):
    """
    Parse a parameter set tuple into (distribution, mean, variance).

    Parameters
    ----------
    param_set : tuple
        Either (distribution, mean) or (distribution, mean, variance) for gamma.

    Returns
    -------
    distribution : str
        Distribution name.
    mean : float
        Mean parameter.
    var : float or None
        Variance parameter (None for non-gamma distributions).
    """
    if len(param_set) == 2:
        distribution, mean = param_set
        var = None
    elif len(param_set) == 3:
        distribution, mean, var = param_set
    else:
        raise ValueError(f"Invalid param_set format: {param_set}")
    return distribution, mean, var


def get_missing_params(param_sets, existing_df=None):
    """
    Return parameter sets not already in the file.

    Parameters
    ----------
    param_sets : list of tuples
        List of (distribution, mean) or (distribution, mean, variance) tuples.
    existing_df : pd.DataFrame, optional
        Existing simulation data. If None, all params are missing.

    Returns
    -------
    list of tuples
        Parameter sets that need to be simulated.
    """
    if existing_df is None:
        return list(param_sets)

    # Build set of existing (distribution, mean) combinations
    # For gamma, we also need to check variance
    missing = []
    for param_set in param_sets:
        distribution, mean, var = parse_param_set(param_set)

        if distribution == 'gamma':
            # For gamma, check distribution + mean + variance
            mask = (
                (existing_df['distribution'] == distribution) &
                (existing_df['mean'] == mean) &
                (existing_df['variance'] == var)
            )
        else:
            # For other distributions, check distribution + mean
            mask = (
                (existing_df['distribution'] == distribution) &
                (existing_df['mean'] == mean)
            )

        if not mask.any():
            missing.append(param_set)

    return missing


#==============================================================================
# HISTOGRAM / HEATMAP PLOT FUNCTIONS
#==============================================================================
def load_and_normalize(df_all, distribution, mean, var=None):
    """
    Filter and normalize data for a specific parameter set.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full DataFrame with all simulations.
    distribution : str
        Distribution name to filter by.
    mean : float
        Mean parameter to filter by.
    var : float, optional
        Variance parameter to filter by (only used for gamma).

    Returns
    -------
    z_norm : np.ndarray
        Normalized baseline local field values (z / sigma_z).
    ztilde_norm : np.ndarray
        Normalized perturbed local field values (ztilde / sigma_ztilde).
    computed_var : float
        The variance of the filtered data (from the DataFrame).
    """
    if distribution == 'gamma' and var is not None:
        df = df_all[
            (df_all['distribution'] == distribution) &
            (df_all['mean'] == mean) &
            (df_all['variance'] == var)
        ]
    else:
        df = df_all[
            (df_all['distribution'] == distribution) &
            (df_all['mean'] == mean)
        ]

    z = df['z'].values
    ztilde = df['ztilde'].values
    computed_var = df['variance'].iloc[0]  # Get the stored variance

    print('Correlation:', np.corrcoef(z, ztilde)[0, 1])

    # Normalize by standard deviation
    z_norm = z / np.std(z)
    ztilde_norm = ztilde / np.std(ztilde)

    return z_norm, ztilde_norm, computed_var


def plot_local_field_hist(z_norm, ztilde_norm, color, fname, index=0):
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
    index : int
        Plot index. Only the first plot (index=0) will have y ticks.
    """
    # Determine plot limits
    data_min = -3.
    data_max = 3.

    # Set up figure
    fig, ax_scatter = plt.subplots(figsize=(width_sm, height_sm))

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
    ax_scatter.set_xticks([-2,0,2])
    # Only first histogram has y ticks
    if index == 0:
        ax_scatter.set_yticks([-2,0,2])
    else:
        ax_scatter.set_yticks([])

    # Set equal aspect ratio
    ax_scatter.set_aspect('equal')

    # Save raw figure
    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"hist_{fname}.svg", dpi=600)

    # Add axis labels
    ax_scatter.set_xlabel(r'Local field $z/\sigma_z$')
    ax_scatter.set_ylabel(r'Perturbed local field $\tilde{z}/\sigma_{\tilde{z}}$')

    plt.show()


#==============================================================================
# ELLIPSE CARTOON FUNCTIONS
#==============================================================================
def plot_ellipse_cartoon(mean, var, color, fname, index=0, lim=3., R=1.5):
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
    fname : str
        Filename suffix for saving.
    index : int
        Plot index. No ellipses have x ticks; only the first (index=0) has y ticks.
    lim : float
        Plot limits.
    R : float
        Radius scaling factor.
    """
    # Compute rho from parameters
    # Handle infinite variance (for Pareto with mean <= 2)
    if np.isinf(var):
        rho = 1.0  # Perfect correlation when variance is infinite
    else:
        rho = (1. + (eps**2) * mean / (var + mean**2)) ** (-0.5)

    # Set up figure
    fig, ax = plt.subplots(figsize=(width_sm, height_sm))

    # Draw coordinate axes (lines through origin)
    ax.plot([-lim, lim], [0., 0.], c='k', alpha=.5, lw=1)
    ax.plot([0., 0.], [-lim, lim], c='k', alpha=.5, lw=1)

    # Draw ellipse
    draw_ellipse(R, rho, ax, color, lim)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    # No ellipses have x ticks; only first has y ticks
    ax.set_xticks([])
    if index == 0:
        ax.set_yticks([-2,0,2])
    else:
        ax.set_yticks([])

    # Save raw figure
    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"ellipse_{fname}.svg", dpi=600)

    # Labels
    ax.set_xlabel('Local field $z/\sigma_z$')
    ax.set_ylabel('Perturbed local field $\\tilde{z}/\sigma_{\\tilde{z}}$')

    plt.show()


#==============================================================================
# GAUSSIAN HEATMAP FUNCTIONS
#==============================================================================
def plot_gaussian_heatmap(rho, color, fname, index=0, lim=3., n_grid=100):
    """
    Plot an analytical 2D Gaussian density heatmap with unit variances and correlation rho.

    Parameters
    ----------
    rho : float
        Correlation parameter between the two variables.
    color : np.ndarray
        RGB color array for the colormap.
    fname : str
        Filename suffix for saving.
    index : int
        Plot index. Only the first plot (index=0) will have y ticks.
    lim : float
        Plot limits (symmetric around origin).
    n_grid : int
        Number of grid points in each dimension.
    """
    # Create grid
    x = np.linspace(-lim, lim, n_grid)
    y = np.linspace(-lim, lim, n_grid)
    X, Y = np.meshgrid(x, y)

    # Compute 2D Gaussian PDF with unit variances and correlation rho
    # f(x,y) = (1 / (2*pi*sqrt(1-rho^2))) * exp(-Q / (2*(1-rho^2)))
    # where Q = x^2 + y^2 - 2*rho*x*y
    Q = X**2 + Y**2 - 2*rho*X*Y
    Z = np.exp(-Q / (2*(1 - rho**2))) / (2*np.pi*np.sqrt(1 - rho**2))

    # Set up figure
    fig, ax = plt.subplots(figsize=(width_sm, height_sm))

    # Plot heatmap using pcolormesh with fade_to_color_cmap
    # rasterized=True embeds as raster image to reduce PDF size/memory in Inkscape
    im = ax.pcolormesh(X, Y, Z, cmap=fade_to_color_cmap(color, alpha_min=0.), shading='auto', rasterized=True)

    # Enable all four spines
    for loc in ["left", "right", "top", "bottom"]:
        ax.spines[loc].set_visible(True)
    ax.set_frame_on(True)

    # Draw reference lines through origin
    ax.plot([0., 0.], [-lim, lim], c="k", alpha=.5, lw=1)
    ax.plot([-lim, lim], [0., 0.], c="k", alpha=.5, lw=1)

    # Set axis limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # No gaussians have x ticks; only first has y ticks
    ax.set_xticks([])
    if index == 0:
        ax.set_yticks([-2,0,2])
    else:
        ax.set_yticks([])

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Save raw figure
    plt.subplots_adjust(**fig_margins_sm)
    plt.savefig(fig_dir + f"gaussian_{fname}.svg", dpi=600)

    # Add axis labels
    ax.set_xlabel(r'Local field $z/\sigma_z$')
    ax.set_ylabel(r'Perturbed local field $\tilde{z}/\sigma_{\tilde{z}}$')

    plt.show()


#==============================================================================
# PARAMETRIC LOSS PLOT FUNCTIONS
#==============================================================================
def plot_parametric_loss(distribution, n_inputs):
    """
    Plot simulated loss for a given parametric distribution.

    Creates a plot of loss vs variance (log x-axis), colored by mean.

    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'lognormal', 'lomax', 'gamma')
    n_inputs : int
        Number of inputs used in the simulation
    """
    # Import simulated loss
    df = pd.read_parquet(sim_dir + f"{distribution}_sim_{n_inputs}.parquet")

    # Get values of mean and variance
    mean_vals = df['mean'].unique()
    var_vals = df['var'].unique()

    # Get predictions
    var_min, var_max = np.min(var_vals), np.max(var_vals)
    var_pred = np.logspace(np.log10(var_min)-0.3, np.log10(var_max)+0.3, n_pred)

    #--------------------------------------------------------------------------
    # Loss vs variance, colored by mean
    #--------------------------------------------------------------------------
    cmap = plt.get_cmap('dark_cool', len(mean_vals))

    fig, ax = plt.subplots(figsize=(width_lg, height_lg))

    for i, mean in enumerate(mean_vals):
        mask = df['mean'] == mean

        # Plot loss
        ax.plot(var_pred, general_loss(mean, var_pred), c=cmap(i), lw=2, label=f"Mean$={mean}$", zorder=0)
        ax.scatter(df[mask]['var'], df[mask]['sim_loss'], c='white', edgecolors=cmap(i), s=20, rasterized=True)

    ax.set_xscale('log')
    # ax.set_ylim([0.,.2])
    # ax.set_yticks(np.linspace(0.,.2,3))
    
    ax.set_yscale('log')
    ax.set_ylim([1e-2,.2])
    ax.set_xlim([5e-1,1e4])
    
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)

    plt.subplots_adjust(**fig_margins_lg)
    plt.savefig(fig_dir + f"{distribution}_simulation_var.svg", dpi=600)

    ax.legend()

    ax.set_xlabel('Variance')
    ax.set_ylabel('Simulated loss')

    plt.show()


#==============================================================================
# MAIN EXECUTION
#==============================================================================

#------------------------------------------------------------------------------
# SECTION 1: CLASSIFICATION PLANE CARTOON
#------------------------------------------------------------------------------
lim = 1.

# Set up figure
fig, ax = plt.subplots(figsize=(width_md, height_md))

y1, y2 = .7, .2

# Axes
ax.plot([-1.,1.],[0.,0.], c='k', lw=1)
ax.plot([0.,0.],[-1.,1.], c='k', lw=1)

# Classification planes
ax.plot([-1.,1.],[y1,-y1], c=con_colors[0], lw=2, label='Unperturbed')
ax.plot([-1.,1.],[y2,-y2], c=con_colors[1], lw=2, label='Perturbed')

ax.annotate("", xytext=(0, 0), xy=(y1*.6/(1.+y1**2)**.5, .6/(1.+y1**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[0]))
ax.annotate("", xytext=(0, 0), xy=(y2*.6/(1.+y2**2)**.5, .6/(1.+y2**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[1]))

ax.fill_between([-1.,1.],[y1,-y1],[y2,-y2], color=con_colors[1], alpha=0.3)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + 'classification_plane.svg', dpi=600)

# Add lables
ax.text(0.25, 0.75, 'Fire', ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.35, 'Not fire', ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')

plt.show()

#------------------------------------------------------------------------------
# SECTION 2: DISTRIBUTION OF LOCAL FIELDS
#------------------------------------------------------------------------------
z_vals = np.linspace(-3,3,200)
p_vals = ((2*np.pi)**(-.5))*np.exp(-(z_vals**2)/2)

# Set up figure
fig, ax = plt.subplots(figsize=(width_md, height_md))

ax.plot(z_vals, p_vals, c=con_colors[0], lw=2)
ax.plot([0.,0.], [0.,1.1/np.sqrt(2*np.pi)], c='k', ls='--', lw=1)

# Add shading and label
mask = z_vals >= 0
ax.fill_between(z_vals[mask], p_vals[mask], 0, color=con_colors[0], alpha=0.2)

# Save raw figure
plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + 'local_field_distribution.svg', dpi=600)

ax.text(0.65, 0.15, 'Fire', ha='center', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.15, 'Not fire', ha='center', va='bottom', transform=ax.transAxes)

ax.set_xlabel('Local field $z$')
ax.set_ylabel('Probability distribution $p(z)$')

plt.show()

#------------------------------------------------------------------------------
# SECTION 3: MISCLASSIFICATION IN Z SPACE
#------------------------------------------------------------------------------
# Tunable parameters
rho = 0.7   # correlation parameter
R = 2.     # ellipse radius scaling
lim = 3.    # plot limits

# Set up figure
fig, ax = plt.subplots(figsize=(width_md, height_md))

# Coordinate axes
ax.plot([-lim, lim], [0., 0.], c='k', lw=1)
ax.plot([0., 0.], [-lim, lim], c='k', lw=1)

# Fill quadrants for misclassification
ax.fill_betweenx(np.linspace(0, lim, 2), -lim, 0, color=con_colors[1], alpha=0.3)
ax.fill_betweenx(np.linspace(-lim, 0, 2), 0, lim, color=con_colors[1], alpha=0.3)

# Draw ellipse (single level set)
draw_ellipse(R, rho, ax, cmap(0.5), lim=lim)

# Draw principal axes (dashed lines within ellipse)
draw_principal_axes(R, rho, ax, con_colors[0], con_colors[1], lw=1)

# Make axes coincide
locator = ax.yaxis.get_major_locator()
ax.xaxis.set_major_locator(locator)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')

# Save raw figure
plt.subplots_adjust(**fig_margins_md)
plt.savefig(fig_dir + '2d_local_field_distribution.svg', dpi=600)

# Annotations
ax.text(0.77, 0.23, 'Error', ha='center', va='center', transform=ax.transAxes)
ax.text(0.23, 0.77, 'Error', ha='center', va='center', transform=ax.transAxes)

# Set labels
ax.set_xlabel('Local field $z$')
ax.set_ylabel('Perturbed local field $\\tilde{z}$')

plt.show()

#------------------------------------------------------------------------------
# SECTION 4: z/ztilde SIMULATION
#------------------------------------------------------------------------------
rng = np.random.default_rng(1764)

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
    for param_set in missing:
        distribution, mean, var = parse_param_set(param_set)
        print(f"  Simulating distribution={distribution}, mean={mean}" +
              (f", variance={var}" if var is not None else "") + "...")
        df, w = compute_z_ztilde(
            distribution=distribution,
            mean=mean,
            n_inputs=n_inputs,
            eta=eta,
            eps=eps,
            n_draws=n_draws,
            n_perturb=n_perturb,
            rng=rng,
            var=var,
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

#------------------------------------------------------------------------------
# SECTION 5: HISTOGRAMS
#------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: HISTOGRAMS")
print("=" * 60)

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    print(f"Processing histogram for distribution={distribution}, mean={mean}...")

    # Load and normalize data
    z_norm, ztilde_norm, computed_var = load_and_normalize(df_all, distribution, mean, var)

    # Plot histogram (include distribution name in filename)
    plot_local_field_hist(z_norm, ztilde_norm, color=sim_colors[i], fname=f"{distribution}_{i}", index=i)

    print(f"  Generated histogram")

#------------------------------------------------------------------------------
# SECTION 6: ELLIPSE CARTOONS
#------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: ELLIPSE CARTOONS")
print("=" * 60)

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    # Load data to get computed variance (needed for rho calculation)
    _, _, computed_var = load_and_normalize(df_all, distribution, mean, var)
    print(f"Generating ellipse cartoon for distribution={distribution}, mean={mean}, variance={computed_var}...")

    # Plot ellipse cartoon (use computed variance for rho calculation)
    plot_ellipse_cartoon(mean, computed_var, sim_colors[i], fname=f"{distribution}_{i}", index=i)

    print(f"  Generated cartoon")

#------------------------------------------------------------------------------
# SECTION 7: GAUSSIAN HEATMAPS
#------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: GAUSSIAN HEATMAPS")
print("=" * 60)

for i, param_set in enumerate(param_sets):
    distribution, mean, var = parse_param_set(param_set)
    # Load data to get computed variance (needed for rho calculation)
    _, _, computed_var = load_and_normalize(df_all, distribution, mean, var)

    # Compute rho from parameters (same formula as in plot_ellipse_cartoon)
    # Handle infinite variance (for Pareto with mean <= 2)
    if np.isinf(computed_var):
        rho = 1.0  # Perfect correlation when variance is infinite
    else:
        rho = (1. + (eps**2) * mean / (computed_var + mean**2)) ** (-0.5)
    print(f"Generating Gaussian heatmap for distribution={distribution}, mean={mean}, variance={computed_var}, rho={rho:.4f}...")

    # Plot Gaussian heatmap (include distribution name in filename)
    plot_gaussian_heatmap(rho, sim_colors[i], fname=f"{distribution}_{i}", index=i)

    print(f"  Generated heatmap")

#------------------------------------------------------------------------------
# SECTION 8: PARAMETRIC LOSS PLOTS
#------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: PARAMETRIC LOSS PLOTS")
print("=" * 60)

plot_parametric_loss('gamma', 1000)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

