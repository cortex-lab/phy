# -*- coding: utf-8 -*-
from .utils.logging import default_logger, set_level

__author__ = 'Kwik Team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.1.0-alpha'

# Set up the default logger.
default_logger()


def debug(enable=True):
    """Enable debug logging mode."""
    if enable:
        set_level('debug')
    else:
        set_level('info')
