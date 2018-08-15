# -*- coding: utf-8 -*-
# flake8: noqa

"""Spike sorting and ephys data analysis for 1000 channels and beyond."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
import sys

from six import StringIO

from .io.datasets import download_file
from .utils.config import load_master_config
from .utils._misc import _git_version
from .utils.plugin import IPlugin, get_plugin, discover_plugins
from .utils.testing import _enable_profiler


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Kwik team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '1.0.9'
__version_git__ = __version__ + _git_version()


# Set a null handler on the root logger
logger = logging.getLogger('phy')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


_logger_fmt = '%(asctime)s [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = {'D': '37', 'I': '0', 'W': '33', 'E': '31'}.get(record.levelname, '7')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


def add_default_handler(level='INFO', logger=logger):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt,
                           datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


DEBUG = False
if '--debug' in sys.argv:  # pragma: no cover
    DEBUG = True
    sys.argv.remove('--debug')


PDB = False
if '--pdb' in sys.argv:  # pragma: no cover
    PDB = True
    sys.argv.remove('--pdb')


IPYTHON = False
if '--ipython' in sys.argv:  # pragma: no cover
    IPYTHON = True
    sys.argv.remove('--ipython')


# Add `profile` in the builtins.
if '--lprof' in sys.argv or '--prof' in sys.argv:  # pragma: no cover
    _enable_profiler('--lprof' in sys.argv)
    if '--prof' in sys.argv:
        sys.argv.remove('--prof')
    if '--lprof' in sys.argv:
        sys.argv.remove('--lprof')


def test():  # pragma: no cover
    """Run the full testing suite of phy."""
    import pytest
    pytest.main()
