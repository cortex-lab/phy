# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

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
def test_profile(tempdir, line_by_line):
    # Remove the profile from the builtins.
    prof = _enable_profiler(line_by_line=line_by_line)
    _profile(prof, 'import time; time.sleep(.001)', {}, {})
    assert (tempdir / '.profile/stats.txt').exists()
