# -*- coding: utf-8 -*-
# flake8: noqa

"""Spike sorting and ephys data analysis for 1000 channels and beyond."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from pkg_resources import get_distribution, DistributionNotFound

from .utils.logging import _default_logger, set_level
from .utils.datasets import download_sample_data
from .utils._misc import _git_version
from .gui.qt import enable_qt, qt_app


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Kwik team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.2.0'
__version_git__ = __version__ + _git_version()


__all__ = ['debug', 'set_level']


# Set up the default logger.
_default_logger()


def debug(enable=True):
    """Enable debug logging mode."""
    if enable:
        set_level('debug')
    else:
        set_level('info')


def test():
    """Run the full testing suite of phy."""
    import pytest
    pytest.main()
