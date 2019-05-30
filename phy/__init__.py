# -*- coding: utf-8 -*-
# flake8: noqa

"""Spike sorting and ephys data analysis for 1000 channels and beyond."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import atexit
import logging
import os.path as op
import sys

from io import StringIO

from phylib.utils._misc import _git_version
from phylib.utils.event import connect, unconnect, emit
from .utils.config import load_master_config
from .utils.plugin import IPlugin, get_plugin, discover_plugins


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Cyrille Rossant'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '2.0alpha'
__version_git__ = __version__ + _git_version()


# Set a null handler on the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())
logger.propagate = False


_logger_fmt = '%(asctime)s [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


@atexit.register
def on_exit():  # pragma: no cover
    # Close the logging handlers.
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def test():  # pragma: no cover
    """Run the full testing suite of phy."""
    import pytest
    pytest.main()
