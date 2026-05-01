"""
    Shared figure formatting utilities for all figure scripts.
-------------------------------------------------------------------------------
created on:
    Thu 1 May 2026
-------------------------------------------------------------------------------
last change:
    Fri 1 May 2026
-------------------------------------------------------------------------------
notes:
    Centralises: matplotlib style setup, log-axis formatting, and the
    outer-tick helper. Figure sizes and margins stay per-script so each
    can be tweaked independently.

    Usage in any figure script:
        from figure_formatting import apply_style, log_format, log10_formatter, outer_tick
        apply_style()
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""

import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def apply_style():
    """Apply publication-quality matplotlib style used across all figure scripts."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
            "font.family": "Helvetica",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    mpl.rcParams["figure.dpi"] = 300
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)


def log_format(ax, format_x=True, format_y=True):
    """Set log scale on ax and force proper decade ticks with inter-decade minors."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    if format_x:
        ax.xaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
        )
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
    if format_y:
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
        )
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )


def log10_formatter(y, pos):
    """Format a log10-transformed tick value as 10^x with Unicode superscripts."""
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    if float(y).is_integer():
        return "10" + str(int(y)).translate(superscripts)
    return ""


def outer_tick(lim):
    """Return a clean tick value at ~75% of lim, rounded down to one significant figure."""
    val = 0.75 * lim
    exp = int(np.floor(np.log10(val)))
    return int(np.floor(val / 10**exp) * 10**exp)
