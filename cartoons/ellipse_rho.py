# %%
"""
    This script generates an ellipse figure satisfying the equation:
    x^2 + y^2 - 2*rho*x*y = sqrt(1 - rho^2)
-------------------------------------------------------------------------------
created on:
    Mon 5 Jan 2026
-------------------------------------------------------------------------------
last change:
    Mon 5 Jan 2026
-------------------------------------------------------------------------------
notes:
    The parameter rho controls the shape of the ellipse.
    For |rho| < 1, the equation describes an ellipse rotated 45 degrees.
    As rho -> 1, the ellipse becomes more elongated along the y=x diagonal.
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
                   [120, 0, 150], [0, 168, 157], [203, 41, 123], [0, 0, 0]])/255

# Nature figure size
width = 3.5
height = 3.2

#------------------------------------------------------------------------------
# ELLIPSE EQUATION: x^2 + y^2 - 2*rho*x*y = sqrt(1 - rho^2)
#------------------------------------------------------------------------------
lim = 1.5

def draw_ellipse(rho, ax, color):
    # Create meshgrid
    xx = np.linspace(-lim, lim, 400)
    yy = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(xx, yy)

    # Evaluate the left-hand side of the equation: x^2 + y^2 - 2*rho*x*y
    Z = X**2 + Y**2 - 2*rho*X*Y

    # The right-hand side is sqrt(1 - rho^2)
    level = np.sqrt(1 - rho**2)

    # Fill the ellipse interior (where Z <= level)
    ax.contourf(X, Y, Z, levels=[0, level], colors=[color], alpha=0.3)

    # Draw the ellipse outline
    ax.contour(X, Y, Z, levels=[level], colors=[color], linewidths=2)

# Set the correlation parameter rho (must satisfy |rho| < 1)
rho = 0.5

# Set up figure
fig, ax = plt.subplots(figsize=(.9*width, .9*height))

# Draw two ellipses
draw_ellipse(0.7, ax, con_colors[2])
draw_ellipse(0.1, ax, con_colors[4])

# Draw coordinate axes (dashed lines through origin)
ax.plot([-lim, lim], [0., 0.], c='k', lw=1)
ax.plot([0., 0.], [-lim, lim], c='k', lw=1)

# Labels
ax.set_xlabel('Local field $z/\sigma_z$')
ax.set_ylabel('Perturbed local field $\\tilde{z}/\sigma_{\\tilde{z}}$')

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')

plt.show()


# %%
