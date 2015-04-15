# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises

from ..logging import set_level
from ..settings import (BaseSettings,
                        UserSettings,
                        InternalSettings,
                        SettingsManager,
                        )
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def test_base_settings_1():
    s = BaseSettings()

    # Namespaces are mandatory.
    with raises(ValueError):
        s.get('a')

    # None is returned if a key doesn't exist.
    assert s.get('test.a') is None

    s.set('test.a', 3)
    assert s.get('test.a') == 3


def test_base_settings_2():
    s = BaseSettings()

    s.set('test.a', 3, scope='my_dataset')
    assert s.get('test.a') is None
    assert s.get('test.a', scope='my_dataset') == 3


def test_base_settings_3():
    s = BaseSettings()

    s.set('test.a', 3)
    assert s.get('test.a') == 3

    # The scope doesn't exist: fallback to global.
    assert s.get('test.a', scope='ds1') == 3

    # Now it works normally.
    s.set('test.a', 6, scope='ds1')
    assert s.get('test.a', scope='ds1') == 6
    assert s.get('test.a', scope='global') == 3

    # We set a new dataset.
    s.set('test.b', 2, scope='ds2')

    # The dataset exists, but not the requested field. We should still
    # fallback to global.
    assert s.get('test.a', scope='ds2') == 3

    assert s.get('test.b', scope='ds2') == 2
    assert s.get('test.b', scope='global') is None


def test_user_settings_1():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        # Create a simple settings file.
        contents = '''test.a = 4\ntest.b = 5\n'''
        with open(path, 'w') as f:
            f.write(contents)

        s = UserSettings()
        # Need to set the namespace 'test' first.
        with raises(NameError):
            s.load(path=path)

        # Set the 'test' namespace.
        s.set('test.a', 3)
        s.set('test.c', 6)
        assert s.get('test.a') == 3

        # Now, set the settings file.
        s.load(path=path)
        assert s.get('test.a') == 4
        assert s.get('test.b') == 5
        assert s.get('test.c') == 6


def test_user_settings_2():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        # Create a simple settings file.
        contents = '''test.a = 4\ntest.b = 5\n'''
        with open(path, 'w') as f:
            f.write(contents)

        s = UserSettings()
        # Need to set the namespace 'test' first.
        with raises(NameError):
            s.load(path=path)

        # Set the 'test' namespace.
        s.declare_namespace('test')
        s.load(path=path)
        assert s.get('test.a') == 4
        assert s.get('test.b') == 5

        s.set('test.a', 6)
        assert s.get('test.a') == 6


def test_internal_settings():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        s = InternalSettings()

        # Set the 'test' namespace.
        s.set('test.a', 3)
        s.set('test.c', 6)
        assert s.get('test.a') == 3
        assert s.get('test.c') == 6

        s.save(path)
        assert s.get('test.a') == 3
        assert s.get('test.c') == 6

        s = InternalSettings()
        assert s.get('test.a') is None

        s.load(path)
        assert s.get('test.a') == 3
        assert s.get('test.c') == 6


def test_settings_manager():
    with TemporaryDirectory() as tmpdir:
        sm = SettingsManager(tmpdir)

        # Check paths.
        assert sm.phy_user_dir == tmpdir
        assert sm.internal_settings_path('global') == op.join(tmpdir,
                                                              'internal_'
                                                              'settings')
        assert sm.user_settings_path('global') == op.join(tmpdir,
                                                          'user_settings.py')

        # User settings.
        assert sm.get_user_settings('test.a') is None
        sm.set_user_settings('test.a', 3)
        assert sm.get_user_settings('test.a') == 3

        # Internal settings.
        assert sm.get_internal_settings('internal.c') is None
        sm.set_internal_settings('internal.c', 5)
        assert sm.get_internal_settings('internal.c') == 5

        # Set an experiment path.
        path = op.join(tmpdir, 'myexperiment.dat')
        sm.set_experiment_path(path)
        assert sm.experiment_path == path
        assert sm.experiment_name == 'myexperiment'
        assert sm.phy_experiment_dir == op.join(tmpdir, 'myexperiment.phy')

        # User settings.
        assert sm.get_user_settings('test.a',
                                    scope='experiment') == 3
        sm.set_user_settings('test.a', 30,
                             scope='experiment')
        assert sm.get_user_settings('test.a',
                                    scope='experiment') == 30

        # Internal settings.
        assert sm.get_internal_settings('internal.c',
                                        scope='experiment') is None
        sm.set_internal_settings('internal.c', 50,
                                 scope='experiment')
        assert sm.get_internal_settings('internal.c',
                                        scope='experiment') == 50

        # Check persistence.
        sm.save()
        sm = SettingsManager(tmpdir)
        sm.set_experiment_path(path)
        assert sm.get_internal_settings('internal.c',
                                        scope='experiment') == 50
        assert sm.get_user_settings('test.a',
                                    scope='experiment') == 30
