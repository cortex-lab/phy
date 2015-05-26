# -*- coding: utf-8 -*-
# flake8: noqa

"""
phy is an open source electrophysiological data analysis package in Python
for neuronal recordings made with high-density multielectrode arrays
containing up to thousands of channels.
"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from pkg_resources import get_distribution, DistributionNotFound

from .utils.logging import _default_logger, set_level
from .utils.datasets import download_test_data
from .utils.dock import enable_qt, qt_app


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Kwik team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.1.0'


__all__ = ['debug', 'set_level']


# Set up the default logger.
_default_logger()


def debug(enable=True):
    """Enable debug logging mode."""
    if enable:
        set_level('debug')
    else:
        set_level('info')
