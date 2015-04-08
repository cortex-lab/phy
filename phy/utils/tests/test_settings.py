# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises

from ..settings import _Settings
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def test_settings_1():
    s = _Settings()

    # Namespaces are mandatory.
    with raises(ValueError):
        s._get('a')

    # None is returned if a key doesn't exist.
    assert s._get('test.a') is None

    s._set({'test.a': 3})
    assert s._get('test.a') == 3


def test_settings_2():
    s = _Settings()

    s._set({'test.a': 3}, scope='my_dataset')
    assert s._get('test.a') is None
    assert s._get('test.a', scope='my_dataset') == 3


def test_settings_path():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        # Create a simple settings file.
        contents = '''test.a = 4\ntest.b = 5\n'''
        with open(path, 'w') as f:
            f.write(contents)

        s = _Settings()
        # Need to set the namespace 'test' first.
        with raises(NameError):
            s._set(path=path)

        # Set the 'test' namespace.
        s._set({'test.a': 3, 'test.c': 6})
        assert s._get('test.a') == 3

        # Now, set the settings file.
        s._set(path=path)
        assert s._get('test.a') == 4
        assert s._get('test.b') == 5
        assert s._get('test.c') == 6
