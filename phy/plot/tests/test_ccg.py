# -*- coding: utf-8 -*-

"""Test CCG plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from ..ccg import plot_ccg


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_plot_ccg():
    n_bins = 51
    ccg = np.random.randint(size=n_bins, low=10, high=50)
    plot_ccg(ccg, color='k')
    # plt.show()
