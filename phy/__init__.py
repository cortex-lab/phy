# -*- coding: utf-8 -*-
# flake8: noqa

"""phy: interactive visualization and manual spike sorting of large-scale ephys data."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import atexit
import logging
import os.path as op
import sys

from io import StringIO

from phylib.utils import Bunch
from phylib.utils._misc import _git_version
from phylib.utils.event import connect, unconnect, emit
from .utils.config import load_master_config
from .utils.plugin import IPlugin, get_plugin, discover_plugins


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Cyrille Rossant'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '2.0b5'
__version_git__ = __version__ + _git_version()


# Set a null handler on the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())
logger.propagate = False


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
