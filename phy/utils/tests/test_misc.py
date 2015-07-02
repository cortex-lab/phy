# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import subprocess

from .._misc import _git_version


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_git_version():
    v = _git_version()

    # If this test file is tracked by git, then _git_version() should succeed
    filedir, _ = op.split(__file__)
    try:
        fnull = open(os.devnull, 'w')
        subprocess.check_output(['git', '-C', filedir, 'status'],
                                stderr=fnull)
        assert type(v) == str, "git_version failed to return"
        assert v[:5] == "git-v", "Git version does not begin in git-v"
    except (OSError, subprocess.CalledProcessError):
        assert v is False
