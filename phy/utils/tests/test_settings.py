# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises

from ..settings import (BaseSettings,
                        Settings,
                        )
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def test_base_settings():
    s = BaseSettings()

    # Namespaces are mandatory.
    with raises(KeyError):
        s['a']

    s['a'] = 3
    assert s['a'] == 3


def test_user_settings():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test.py')

        # Create a simple settings file.
        contents = '''a = 4\nb = 5\nd = {'k1': 2, 'k2': 3}\n'''
        with open(path, 'w') as f:
            f.write(contents)

        s = BaseSettings()

        s['a'] = 3
        s['c'] = 6
        assert s['a'] == 3

        # Now, set the settings file.
        s.load(path=path)
        assert s['a'] == 4
        assert s['b'] == 5
        assert s['c'] == 6
        assert s['d'] == {'k1': 2, 'k2': 3}

        s = BaseSettings()
        s['d'] = {'k2': 30, 'k3': 40}
        s.load(path=path)
        assert s['d'] == {'k1': 2, 'k2': 3, 'k3': 40}


def test_internal_settings():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')

        s = BaseSettings()

        # Set the 'test' namespace.
        s['a'] = 3
        s['c'] = 6
        assert s['a'] == 3
        assert s['c'] == 6

        s.save(path)
        assert s['a'] == 3
        assert s['c'] == 6

        s = BaseSettings()
        with raises(KeyError):
            s['a']

        s.load(path)
        assert s['a'] == 3
        assert s['c'] == 6


def test_settings_manager():
    with TemporaryDirectory() as tmpdir:
        with TemporaryDirectory() as tmpdir_exp:
            sm = Settings(tmpdir)

            # Check paths.
            assert sm.phy_user_dir == tmpdir
            assert sm.internal_settings_path == op.join(tmpdir,
                                                        'internal_settings')
            assert sm.user_settings_path == op.join(tmpdir, 'user_settings.py')

            # User settings.
            with raises(KeyError):
                sm['a']
            # Artificially populate the user settings.
            sm._bs._store['a'] = 3
            assert sm['a'] == 3

            # Internal settings.
            sm['c'] = 5
            assert sm['c'] == 5

            # Set an experiment path.
            path = op.join(tmpdir_exp, 'myexperiment.dat')
            sm.on_open(path)
            assert op.realpath(sm.exp_path) == op.realpath(path)
            assert sm.exp_name == 'myexperiment'
            assert (op.realpath(sm.exp_settings_dir) ==
                    op.realpath(op.join(tmpdir_exp, 'myexperiment.phy')))
            assert (op.realpath(sm.exp_settings_path) ==
                    op.realpath(op.join(tmpdir_exp, 'myexperiment.phy/'
                                                    'user_settings.py')))

            # User settings.
            assert sm['a'] == 3
            sm._bs._store['a'] = 30
            assert sm['a'] == 30

            # Internal settings.
            sm['c'] = 50
            assert sm['c'] == 50

            # Check persistence.
            sm.save()
            sm = Settings(tmpdir)
            sm.on_open(path)
            assert sm['c'] == 50
            assert 'a' not in sm
