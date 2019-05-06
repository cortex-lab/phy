# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
import os
import os.path as op
import sys

from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from six import StringIO

from phy.io.datasets import download_file
from ._types import _is_array_like
from .config import _ensure_dir_exists, phy_config_dir

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def captured_logging(name=None):
    buffer = StringIO()
    logger = logging.getLogger(name)
    handlers = list(logger.handlers)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield buffer
    buffer.flush()
    logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)
    handler.close()


def _assert_equal(d_0, d_1):
    """Check that two objects are equal."""
    # Compare arrays.
    if _is_array_like(d_0):
        try:
            ae(d_0, d_1)
        except AssertionError:
            ac(d_0, d_1)
    # Compare dicts recursively.
    elif isinstance(d_0, dict):
        assert set(d_0) == set(d_1)
        for k_0 in d_0:
            _assert_equal(d_0[k_0], d_1[k_0])
    else:
        # General comparison.
        assert d_0 == d_1


def _in_travis():  # pragma: no cover
    return 'TRAVIS' in os.environ


_BASE_URL = 'https://raw.githubusercontent.com/kwikteam/phy-data/master/'


def download_test_file(name, config_dir=None, force=False):
    """Download a test file."""
    config_dir = config_dir or phy_config_dir()
    path = op.join(config_dir, 'test_data', name)
    _ensure_dir_exists(op.dirname(path))
    if not force and op.exists(path):
        return path
    url = _BASE_URL + name
    download_file(url, output_path=path)
    return path
