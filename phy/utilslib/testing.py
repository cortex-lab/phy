# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
import os
import sys

from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from six import StringIO

from ._types import _is_array_like

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
