# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import subprocess

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises
from six import string_types

from .._misc import (_git_version, _load_json, _save_json,
                     _encode_qbytearray, _decode_qbytearray,
                     )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_qbytearray(tempdir):

    from phy.gui.qt import QtCore
    arr = QtCore.QByteArray()
    arr.append('1')
    arr.append('2')
    arr.append('3')

    encoded = _encode_qbytearray(arr)
    assert isinstance(encoded, string_types)
    decoded = _decode_qbytearray(encoded)
    assert arr == decoded

    # Test JSON serialization of QByteArray.
    d = {'arr': arr}
    path = op.join(tempdir, 'test')
    _save_json(path, d)
    d_bis = _load_json(path)
    assert d == d_bis


def test_json_simple(tempdir):
    d = {'a': 1, 'b': 'bb', 3: '33', 'mock': {'mock': True}}

    path = op.join(tempdir, 'test')
    _save_json(path, d)
    d_bis = _load_json(path)
    assert d == d_bis

    with open(path, 'w') as f:
        f.write('')
    assert _load_json(path) == {}
    with raises(IOError):
        _load_json(path + '_bis')


def test_json_numpy(tempdir):
    arr = np.arange(10).reshape((2, 5)).astype(np.float32)
    d = {'a': arr, 'b': arr.ravel()[0]}

    path = op.join(tempdir, 'test')
    _save_json(path, d)

    d_bis = _load_json(path)
    arr_bis = d_bis['a']

    assert arr_bis.dtype == arr.dtype
    assert arr_bis.shape == arr.shape
    ae(arr_bis, arr)

    assert d['b'] == d_bis['b']


def test_git_version():
    v = _git_version()

    # If this test file is tracked by git, then _git_version() should succeed
    filedir, _ = op.split(__file__)
    try:
        fnull = open(os.devnull, 'w')
        subprocess.check_output(['git', '-C', filedir, 'status'],
                                stderr=fnull)
        assert v is not "", "git_version failed to return"
        assert v[:5] == "-git-", "Git version does not begin in -git-"
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        assert v == ""
