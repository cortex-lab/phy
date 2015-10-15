# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import subprocess
from textwrap import dedent

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises
from six import string_types
from traitlets import Float
from traitlets.config import Configurable

from .._misc import (_git_version, _load_json, _save_json, _read_python,
                     _write_text,
                     _load_config, load_master_config,
                     _encode_qbytearray, _decode_qbytearray,
                     )
from .. import _misc


#------------------------------------------------------------------------------
# Misc tests
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


def test_read_python(tempdir):
    path = op.join(tempdir, 'mock.py')
    with open(path, 'w') as f:
        f.write("""a = {'b': 1}""")

    assert _read_python(path) == {'a': {'b': 1}}


def test_write_text(tempdir):
    for path in (op.join(tempdir, 'test_1'),
                 op.join(tempdir, 'test_dir/test_2.txt'),
                 ):
        _write_text(path, 'hello world')
        with open(path, 'r') as f:
            assert f.read() == 'hello world'


def test_temp_user_dir(temp_user_dir):
    assert _misc.phy_user_dir() == temp_user_dir


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


#------------------------------------------------------------------------------
# Config tests
#------------------------------------------------------------------------------

def test_load_config(tempdir):

    class MyConfigurable(Configurable):
        my_var = Float(0.0, config=True)

    assert MyConfigurable().my_var == 0.0

    # Create and load a config file.
    config_contents = dedent("""
       c = get_config()

       c.MyConfigurable.my_var = 1.0
       """)

    path = op.join(tempdir, 'config.py')
    with open(path, 'w') as f:
        f.write(config_contents)

    c = _load_config(path)
    assert c.MyConfigurable.my_var == 1.0

    # Create a new MyConfigurable instance.
    configurable = MyConfigurable()
    assert configurable.my_var == 0.0

    # Load the config object.
    configurable.update_config(c)
    assert configurable.my_var == 1.0


def test_load_master_config(temp_user_dir):
    # Create a config file in the temporary user directory.
    config_contents = dedent("""
       c = get_config()
       c.MyConfigurable.my_var = 1.0
       """)
    with open(op.join(temp_user_dir, 'phy_config.py'), 'w') as f:
        f.write(config_contents)

    # Load the master config file.
    c = load_master_config()
    assert c.MyConfigurable.my_var == 1.
