# -*- coding: utf-8 -*-

"""Plotting CCGs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

import matplotlib.pyplot as plt

from ._mpl_utils import _bottom_left_frame


#------------------------------------------------------------------------------
# CCG plotting
#------------------------------------------------------------------------------

def plot_ccg(ccg, baseline=None, bin=1., color=None, ax=None):
    """Plot a CCG with matplotlib and return an Axes instance."""
    if ax is None:
        ax = plt.subplot(111)
    assert ccg.ndim == 1
    n = ccg.shape[0]
    assert n % 2 == 1
    bin = float(bin)
    x_min = -n // 2 * bin - bin / 2
    x_max = (n // 2 - 1) * bin + bin / 2
    width = bin * 1.05
    left = np.linspace(x_min, x_max, n)
    ax.bar(left, ccg, facecolor=color, width=width, linewidth=0)
    if baseline is not None:
        ax.axhline(baseline, color='k', linewidth=2, linestyle='-')
    ax.axvline(color='k', linewidth=2, linestyle='--')

    ax.set_xlim(x_min, x_max + bin / 2)
    ax.set_ylim(0)

    # Only keep the bottom and left ticks.
    _bottom_left_frame(ax)

    return ax
