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

from .io.datasets import download_file, download_sample_data
from .utils._misc import _git_version
from .utils.plugin import IPlugin


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Kwik team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.3.0.dev0'
__version_git__ = __version__ + _git_version()


# Set a null handler on the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


_logger_fmt = '%(asctime)s  [%(levelname)s]  %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(16)
        return super(_Formatter, self).format(record)


def add_default_handler(level='INFO'):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt,
                           datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


if '--debug' in sys.argv:  # pragma: no cover
    add_default_handler('DEBUG')
    logger.info("Activate DEBUG level.")


# Force dask to use the synchronous scheduler: we'll use ipyparallel
# manually for parallel processing.
try:
    import dask.async
    from dask import set_options
    set_options(get=dask.async.get_sync)
except ImportError:  # pragma: no cover
    logger.debug("dask is not available.")


def test():  # pragma: no cover
    """Run the full testing suite of phy."""
    import pytest
    pytest.main()
