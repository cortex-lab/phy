# -*- coding: utf-8 -*-

"""matplotlib utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# matplotlib utilities
#------------------------------------------------------------------------------

def _bottom_left_frame(ax):
    """Only keep the bottom and left ticks in a matplotlib Axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
