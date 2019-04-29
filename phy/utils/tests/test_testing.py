# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy
import logging
import os.path as op
import time

import numpy as np
from pytest import mark

from ..testing import (benchmark, captured_output, captured_logging,
                       _assert_equal, _enable_profiler, _profile,
                       download_test_file
                       )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_captured_output():
    with captured_output() as (out, err):
        print('Hello world!')
    assert out.getvalue().strip() == 'Hello world!'


def test_captured_logging():
    logger = logging.getLogger()
    handlers = logger.handlers
    with captured_logging() as buf:
        logger.debug('Hello world!')
    assert 'Hello world!' in buf.getvalue()
    assert logger.handlers == handlers


def test_assert_equal():
    d = {'a': {'b': np.random.rand(5), 3: 'c'}, 'b': 2.}
    d_bis = deepcopy(d)
    d_bis['a']['b'] = d_bis['a']['b'] + 1e-10
    _assert_equal(d, d_bis)


def test_benchmark():
    with benchmark():
        time.sleep(.002)


@mark.parametrize('line_by_line', [False, True])
def test_profile(chdir_tempdir, line_by_line):
    # Remove the profile from the builtins.
    prof = _enable_profiler(line_by_line=line_by_line)
    _profile(prof, 'import time; time.sleep(.001)', {}, {})
    assert op.exists(op.join(chdir_tempdir, '.profile', 'stats.txt'))


def test_download_test_file(tempdir):
    name = 'test/test-4ch-1s.dat'
    path = download_test_file(name, config_dir=tempdir)
    assert op.exists(path)
    assert op.getsize(path) == 160000
    path = download_test_file(name, config_dir=tempdir)
