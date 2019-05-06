# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import time

from pytest import mark

from ..profiling import benchmark, _enable_profiler, _profile


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_benchmark():
    with benchmark():
        time.sleep(.002)


@mark.parametrize('line_by_line', [False, True])
def test_profile(chdir_tempdir, line_by_line):
    # Remove the profile from the builtins.
    prof = _enable_profiler(line_by_line=line_by_line)
    _profile(prof, 'import time; time.sleep(.001)', {}, {})
    assert op.exists(op.join(chdir_tempdir, '.profile', 'stats.txt'))
