"""
    This script generates simple figures to illustrate the theoretical 
    framework for the robustness analysis
-------------------------------------------------------------------------------
created on:
    Sun 9 Nov 2025
-------------------------------------------------------------------------------
last change:
    Sun 9 Nov 2025
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
import pandas as pd
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
width = 3.5
height = 3.5

def fade_to_color_cmap(rgb, alpha_min, name="fade_to_color"):
    white = (1.0, 1.0, 1.0)
    # interpolate between white and rgb
    colors = [
        (*np.array(white), alpha_min),
        (*np.array(rgb), 1.0)
    ]
    return LinearSegmentedColormap.from_list(name, colors, N=256)

#------------------------------------------------------------------------------
# LINEAR CLASSIFIER
#------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width,.9*height))

y1, y2 = .7, .2

# Axes
ax.plot([-1.,1.],[0.,0.], c='k', ls='--', lw=1)
ax.plot([0.,0.],[-1.,1.], c='k', ls='--', lw=1)

# Classification planes
ax.plot([-1.,1.],[y1,-y1], c=con_colors[0], lw=2, label='Unperturbed')
ax.plot([-1.,1.],[y2,-y2], c=con_colors[1], lw=2, label='Perturbed')
# ax.legend()

ax.annotate("", xytext=(0, 0), xy=(y1*.6/(1.+y1**2)**.5, .6/(1.+y1**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[0]))
ax.annotate("", xytext=(0, 0), xy=(y2*.6/(1.+y2**2)**.5, .6/(1.+y2**2)**.5), arrowprops=dict(arrowstyle="->", color=con_colors[1]))

ax.text(y1*.6/(1.+y1**2)**.5, .6/(1.+y1**2)**.5, '${\\bf w}$', ha='left', va='bottom', color=con_colors[0])
ax.text(y2*.6/(1.+y2**2)**.5, .6/(1.+y2**2)**.5, '$\\tilde{{\\bf w}}$', ha='left', va='bottom', color=con_colors[1])

ax.fill_between([-1.,1.],[y1,-y1],[y2,-y2], color='grey', alpha=0.3)

# Add lables
ax.text(0.25, 0.75, 'Fire', ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.35, 'Not fire', ha='right', va='top', transform=ax.transAxes)

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')

ax.set_xticks([-1,0,1])
ax.set_yticks([-1,0,1])

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
ax.text(0.65, 0.15, 'Fire', ha='center', va='bottom', transform=ax.transAxes)
ax.text(0.35, 0.15, 'Not fire', ha='center', va='bottom', transform=ax.transAxes)

ax.set_xlabel('Local field $z$')
ax.set_ylabel('Probability distribution $p(z)$')

plt.show()

#------------------------------------------------------------------------------
# MISCLASSIFICATION IN Z SPACE
#------------------------------------------------------------------------------
# Set up figure
fig, ax = plt.subplots(figsize=(.9*width,.9*height))

# Axes
ax.plot([-1.,1.],[0.,0.], c='k', lw=1)
ax.plot([0.,0.],[-1.,1.], c='k', lw=1)

# Fill quadrants for misclassification
ax.fill_betweenx(np.linspace(0, 1, 2), -1, 0, color=con_colors[1], alpha=0.3)
ax.fill_betweenx(np.linspace(-1, 0, 2), 0, 1, color=con_colors[1], alpha=0.3)

# Labels
ax.text(0.77, 0.23, 'Error', ha='center', va='center', transform=ax.transAxes)
ax.text(0.23, 0.77, 'Error', ha='center', va='center', transform=ax.transAxes)

# Gaussian contours
# Grid over plotting window
xx = np.linspace(-1, 1, 400)
yy = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(xx, yy)


# Ellipse parameters
theta = np.deg2rad(45)   # rotation so semimajor axis lies on y = x
sigma_major = 0.7
sigma_minor = 0.25
cx, cy = 0.0, 0.0

# Covariance aligned with ellipse axes
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
D = np.diag([sigma_major**2, sigma_minor**2])
Sigma = R @ D @ R.T
A = np.linalg.inv(Sigma)

# Evaluate Gaussian
x0, y0 = X - cx, Y - cy
quad = A[0,0]*x0**2 + 2*A[0,1]*x0*y0 + A[1,1]*y0**2
G = np.exp(-0.5 * quad)

# --- Filled contour plot ---
levels = np.linspace(0.2, 1.0, 6)  # choose how many ellipses you want
contourf = ax.contourf(
    X, Y, G/G.max(),
    levels=levels,
    cmap=fade_to_color_cmap(con_colors[0], 0.3),
    zorder=3
)
# Add contour lines on top for definition
contours = ax.contour(
    X, Y, G/G.max(),
    levels=levels,
    colors=[con_colors[0]],
    alpha=0.8,
    linewidths=1.2,
    zorder=4
)

# Make axes coincide
locator = ax.yaxis.get_major_locator()
ax.xaxis.set_major_locator(locator)

ax.set_xlabel('Local field $z$')
ax.set_ylabel('Perturbed local field $\\tilde{z}$')

plt.show()












