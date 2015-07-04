# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import subprocess

from .._misc import _git_version, _load_json, _save_json
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_json():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')
        d = {'a': 1, 'b': 'bb'}
        _save_json(path, d)
        d_bis = _load_json(path)
        assert d == d_bis


def test_git_version():
    v = _git_version()

    # If this test file is tracked by git, then _git_version() should succeed
    filedir, _ = op.split(__file__)
    try:
        fnull = open(os.devnull, 'w')
        subprocess.check_output(['git', '-C', filedir, 'status'],
                                stderr=fnull)
        assert v is not "", "git_version failed to return"
        assert v[:6] == "-git-v", "Git version does not begin in -git-v"
    except (OSError, subprocess.CalledProcessError):
        assert v == ""
