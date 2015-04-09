# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises

from ..settings import BaseSettings, UserSettings, InternalSettings
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def test_base_settings_1():
    s = BaseSettings()

    # Namespaces are mandatory.
    with raises(ValueError):
        s.get('a')

    # None is returned if a key doesn't exist.
    assert s.get('test.a') is None

    s.set({'test.a': 3})
    assert s.get('test.a') == 3


def test_base_settings_2():
    s = BaseSettings()

    s.set({'test.a': 3}, scope='my_dataset')
    assert s.get('test.a') is None
    assert s.get('test.a', scope='my_dataset') == 3


def test_user_settings_path():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        # Create a simple settings file.
        contents = '''test.a = 4\ntest.b = 5\n'''
        with open(path, 'w') as f:
            f.write(contents)

        s = UserSettings()
        # Need to set the namespace 'test' first.
        with raises(NameError):
            s.set(path=path)

        # Set the 'test' namespace.
        s.set({'test.a': 3, 'test.c': 6})
        assert s.get('test.a') == 3

        # Now, set the settings file.
        s.set(path=path)
        assert s.get('test.a') == 4
        assert s.get('test.b') == 5
        assert s.get('test.c') == 6
