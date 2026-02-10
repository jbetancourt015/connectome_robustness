"""
    This script generates simple figures to illustrate the theoretical 
    framework for the robustness analysis
-------------------------------------------------------------------------------
created on:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
last change:
    Mon 26 Jan 2026
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'text.usetex': False,  # keep LaTeX off globally
    'mathtext.fontset': 'cm',   # or 'stixsans' for sans-serif
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
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


con_colors = np.array([[0, 77, 128], [181, 23, 0], [1, 113, 0], [242, 112, 0], 
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255;

# Nature figure size
mm_to_in = 25.4
width = 50./mm_to_in
height = 50./mm_to_in

# Fixed margins for consistent axes size across all single-panel figures
fig_margins = dict(left=0.20, right=0.95, bottom=0.18, top=0.95)

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    white = (1.0, 1.0, 1.0)
    # interpolate between white and rgb
    colors = [
        (*np.array(white), alpha_min),
        (*np.array(rgb), 1.0)
    ]
    return LinearSegmentedColormap.from_list(name, colors, N=256)


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


def draw_principal_axes(R, rho, ax, color, lw=1):
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
    
    ax.plot([-x_pos, x_pos], [-x_pos, x_pos], c=color, ls='--', lw=lw)
    ax.plot([-x_neg, x_neg], [x_neg, -x_neg], c=color, ls='--', lw=lw)

#------------------------------------------------------------------------------
# LINEAR CLASSIFIER
#------------------------------------------------------------------------------
lim = 1.

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width,.9*height))

y1, y2 = .7, .2

# Axes
ax.plot([-1.,1.],[0.,0.], c='k', lw=1)
ax.plot([0.,0.],[-1.,1.], c='k', lw=1)

# Classification planes
ax.plot([-1.,1.],[y1,-y1], c=con_colors[0], lw=2, label='Unperturbed')
ax.plot([-1.,1.],[y2,-y2], c=con_colors[1], lw=2, label='Perturbed')

ax.annotate("", xytext=(0, 0), xy=(y1*.6/(1.+y1**2)**.5, .6/(1.+y1**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[0]))
ax.annotate("", xytext=(0, 0), xy=(y2*.6/(1.+y2**2)**.5, .6/(1.+y2**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[1]))

ax.text(y1*.6/(1.+y1**2)**.5, .6/(1.+y1**2)**.5, '${\\bf w}$', ha='left', va='bottom', color=con_colors[0])
ax.text(y2*.6/(1.+y2**2)**.5, .6/(1.+y2**2)**.5, '$\\tilde{{\\bf w}}$', ha='left', va='bottom', color=con_colors[1])

ax.fill_between([-1.,1.],[y1,-y1],[y2,-y2], color=con_colors[1], alpha=0.3)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/framework/classification_plane.pdf', dpi=600)

# Add lables
ax.text(0.25, 0.75, 'Fire', ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.35, 'Not fire', ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')

plt.show()

#------------------------------------------------------------------------------
# DISTRIBUTION OF LOCAL FIELDS
#------------------------------------------------------------------------------
z_vals = np.linspace(-3,3,200)
p_vals = ((2*np.pi)**(-.5))*np.exp(-(z_vals**2)/2)

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width,.9*height))

ax.plot(z_vals, p_vals, c=con_colors[0], lw=2)
ax.plot([0.,0.], [0.,1.1/np.sqrt(2*np.pi)], c='k', ls='--', lw=1)

# Add shading and label
mask = z_vals >= 0
ax.fill_between(z_vals[mask], p_vals[mask], 0, color=con_colors[0], alpha=0.2)

# Save raw figure
plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/framework/local_field_distribution.pdf', dpi=600)

ax.text(0.65, 0.15, 'Fire', ha='center', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.15, 'Not fire', ha='center', va='bottom', transform=ax.transAxes)

ax.set_xlabel('Local field $z$')
ax.set_ylabel('Probability distribution $p(z)$')

plt.show()

#------------------------------------------------------------------------------
# MISCLASSIFICATION IN Z SPACE
#------------------------------------------------------------------------------
# Tunable parameters
rho = 0.7   # correlation parameter
R = 2.     # ellipse radius scaling
lim = 3.    # plot limits

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width,.9*height))

# Coordinate axes
ax.plot([-lim, lim], [0., 0.], c='k', lw=1)
ax.plot([0., 0.], [-lim, lim], c='k', lw=1)

# Fill quadrants for misclassification
ax.fill_betweenx(np.linspace(0, lim, 2), -lim, 0, color=con_colors[1], alpha=0.3)
ax.fill_betweenx(np.linspace(-lim, 0, 2), 0, lim, color=con_colors[1], alpha=0.3)

# Draw ellipse (single level set)
draw_ellipse(R, rho, ax, con_colors[2], lim=lim)

# Draw principal axes (dashed lines within ellipse)
draw_principal_axes(R, rho, ax, con_colors[2], lw=1)

# Make axes coincide
locator = ax.yaxis.get_major_locator()
ax.xaxis.set_major_locator(locator)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')

# Save raw figure
plt.subplots_adjust(**fig_margins)
plt.savefig('../../paper_figures/framework/2d_local_field_distribution.pdf', dpi=600)

# Annotations
ax.text(0.77, 0.23, 'Error', ha='center', va='center', transform=ax.transAxes)
ax.text(0.23, 0.77, 'Error', ha='center', va='center', transform=ax.transAxes)

# Set labels
ax.set_xlabel('Local field $z$')
ax.set_ylabel('Perturbed local field $\\tilde{z}$')

plt.show()












