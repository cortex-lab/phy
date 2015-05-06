# -*- coding: utf-8 -*-
# flake8: noqa

"""Electrophysiological data analysis package in Python.

phy is an open source electrophysiological data analysis package in Python
for neuronal recordings made with high-density multielectrode arrays
containing tens, hundreds, or thousands of channels.

"""

from .utils.logging import default_logger, set_level

__author__ = 'Kwik Team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.1.0-alpha'

__all__ = ['debug', 'set_level']


# Set up the default logger.
default_logger()


def debug(enable=True):
    """Enable debug logging mode."""
    if enable:
        set_level('debug')
    else:
        set_level('info')
