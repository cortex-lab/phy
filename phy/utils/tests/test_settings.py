# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import raises

from ..settings import (BaseSettings,
                        Settings,
                        _load_default_settings,
                        _recursive_dirs,
                        )


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def test_recursive_dirs():
    dirs = list(_recursive_dirs())
    assert len(dirs) >= 5
    root = op.join(op.realpath(op.dirname(__file__)), '../../')
    for dir in dirs:
        dir = op.relpath(dir, root)
        assert '.' not in dir
        assert '_' not in dir


def test_load_default_settings():
    settings = _load_default_settings()
    keys = settings.keys()
    assert 'log_file_level' in keys
    assert 'on_open' in keys
    assert 'spikedetekt' in keys
    assert 'klustakwik2' in keys
    assert 'traces' in keys
    assert 'cluster_manual_config' in keys


def test_base_settings():
    s = BaseSettings()

    # Namespaces are mandatory.
    with raises(KeyError):
        s['a']

    s['a'] = 3
    assert s['a'] == 3


def test_user_settings(tempdir):
    path = op.join(tempdir, 'test.py')

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


def test_internal_settings(tempdir):
    path = op.join(tempdir, 'test')

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


def test_settings_manager(tempdir, tempdir_bis):
    tempdir_exp = tempdir_bis
    sm = Settings(tempdir)

    # Check paths.
    assert sm.phy_user_dir == tempdir
    assert sm.internal_settings_path == op.join(tempdir,
                                                'internal_settings')
    assert sm.user_settings_path == op.join(tempdir, 'user_settings.py')

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
    path = op.join(tempdir_exp, 'myexperiment.dat')
    sm.on_open(path)
    assert op.realpath(sm.exp_path) == op.realpath(path)
    assert sm.exp_name == 'myexperiment'
    assert (op.realpath(sm.exp_settings_dir) ==
            op.realpath(op.join(tempdir_exp, 'myexperiment.phy')))
    assert (op.realpath(sm.exp_settings_path) ==
            op.realpath(op.join(tempdir_exp, 'myexperiment.phy/'
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
    sm = Settings(tempdir)
    sm.on_open(path)
    assert sm['c'] == 50
    assert 'a' not in sm
