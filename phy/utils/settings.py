# -*- coding: utf-8 -*-

"""Settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import re

from ..ext.six.moves.cPickle import load, dump
from ._misc import Bunch, _phy_user_dir, _ensure_path_exists
from .logging import debug, warn


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _split_namespace(name, namespace=None):
    if '.' in name:
        namespace, name = name.split('.')
    if namespace is None:
        raise ValueError("The namespace must be specified.")
    return namespace, name


#------------------------------------------------------------------------------
# Base Settings
#------------------------------------------------------------------------------

class BaseSettings(object):
    """Store key-value pairs."""
    def __init__(self):
        self._store = {'global': {}}

    @property
    def _namespaces(self):
        return sorted(self._store['global'])

    def get(self, name, namespace=None, scope='global'):
        """Get a settings value.

        Parameters
        ----------
        name : str
            The settings name
        scope : str (default is 'global')
            The scope for that setting. Can be 'global' or a dataset name.

        """
        namespace, name = _split_namespace(name, namespace=namespace)
        if scope not in self._store:
            scope = 'global'
        out = self._store[scope].get(namespace, {}).get(name, None)
        # Fallback to 'global' scope if the requested value is None.
        if scope != 'global' and out is None:
            out = self._store['global'].get(namespace, {}).get(name, None)
        return out

    def set(self, key, value, namespace=None, scope='global'):
        """Set some settings

        Parameters
        ----------
        key : str
        value : object
        namespace : str
            The namespace if it is not specified in the keys.
        scope : str (default is 'global')
            The scope for that setting. Can be 'global' or a dataset name.

        """
        if scope not in self._store:
            self._store[scope] = Bunch()
        namespace, name = _split_namespace(key, namespace=namespace)

        # Create dictionaries if they do not exist.
        if namespace not in self._store[scope]:
            self._store[scope][namespace] = Bunch()

        # Update the settings.
        self._store[scope][namespace][name] = value


#------------------------------------------------------------------------------
# User Settings
#------------------------------------------------------------------------------

class UserSettings(BaseSettings):
    """Support Python settings files."""

    def read_settings_file(self, path, file_namespace=None):
        """Return a dictionary {namespace: {key: value}} dictionary."""
        with open(path, 'r') as f:
            contents = f.read()
        if file_namespace is None:
            file_namespace = {}
        # Executing the code directly updates the internal store.
        # The current store is passed as a global namespace.
        try:
            exec(contents, Bunch(self._store['global']), file_namespace)
        except NameError as e:
            r = re.search("'([^']+)'", e.args[0])
            if r:
                name = r.group(1)
            namespaces = ', '.join(self._namespaces)
            raise NameError("Unknown namespace '{0:s}'. ".format(name) +
                            "Known namespaces are: {0:s}.".format(namespaces))

    def set(self, key=None, value=None, namespace=None, scope='global',
            path=None, file_namespace=None):
        """Set some settings

        Parameters
        ----------
        key : str
        value : str
        namespace : str
            The namespace if it is not specified in the keys.
        scope : str (default is 'global')
            The scope for that setting. Can be 'global' or a dataset name.
        path : str
            A path to a Python settings file.
        file_namespace : dict
            A namespace to pass to the Python settings file.

        """
        if path is not None:
            path = op.expanduser(path)
            path = op.realpath(path)
            assert op.exists(path)
            return self.read_settings_file(path,
                                           file_namespace=file_namespace)
        super(UserSettings, self).set(key, value,
                                      namespace=namespace,
                                      scope=scope,
                                      )


_USER_SETTINGS = UserSettings()


def get(*args, **kwargs):
    return _USER_SETTINGS.get(*args, **kwargs)


def set(*args, **kwargs):
    return _USER_SETTINGS.set(*args, **kwargs)


#------------------------------------------------------------------------------
# Internal Settings
#------------------------------------------------------------------------------

class InternalSettings(object):
    """Settings to be modified by the program, not the user."""

    def __init__(self):
        self._store = {}

    def load(self, path):
        path = op.realpath(op.expanduser(path))
        if not op.exists(path):
            raise ValueError("The file '{0}' doesn't exist.".format(path))
        try:
            with open(path, 'rb') as f:
                store = load(f)
        except Exception as e:
            warn("Unable to read the internal settings. "
                 "You may want to delete '{0}'.\n{1}".format(path, str(e)))
        assert isinstance(store, dict)
        debug("Loaded internal settings from '{0}'.".format(path))
        self._store = store

    def save(self, path):
        path = op.realpath(op.expanduser(path))
        with open(path, 'wb') as f:
            dump(self._store, f)
        debug("Saved internal settings to '{0}'.".format(path))

    def get(self, name):
        return self._store.get(name, None)

    def set(self, name, value):
        self._store[name] = value


#------------------------------------------------------------------------------
# Settings files manager
#------------------------------------------------------------------------------

def _create_internal_settings(path):
    # Initialize the global InternalSettings instance.
    internal_settings = InternalSettings()
    # Create the file if it doesn't exist.
    if not op.exists(path):
        internal_settings.save(path)
    else:
        internal_settings.load(path)
    return internal_settings


class SettingsManager(object):
    """Manage global and per-experiment user and internal settings."""

    def __init__(self, phy_user_dir=None):
        self.phy_experiment_dir = None

        # '.phy/' user directory, ~/.phy by default.
        if phy_user_dir is None:
            phy_user_dir = _phy_user_dir()
        self.phy_user_dir = phy_user_dir
        _ensure_path_exists(self.phy_user_dir)

        # Load global user settings.
        self._load_user_settings('global')

        # Initialize the global InternalSettings instance.
        path = self.internal_settings_path('global')
        self._internal_settings = {
            'global': _create_internal_settings(path),
            'experiment': None}

    def _load_user_settings(self, scope):
        path = self.user_settings_path(scope)
        if op.exists(path):
            set(path=path)

    def set_experiment_path(self, experiment_path):
        self.experiment_path = experiment_path
        self.experiment_name = op.splitext(op.basename(experiment_path))[0]

        # Experiment directory.
        self.experiment_dir = op.dirname(experiment_path)
        _ensure_path_exists(self.experiment_dir)

        # _phy subdirectory in the experiment directory.
        self.phy_experiment_dir = op.join(self.experiment_dir,
                                          self.experiment_name + '.phy')
        _ensure_path_exists(self.phy_experiment_dir)

        # Load per-experiment user settings.
        self._load_user_settings('experiment')

        # Initialize the experiment InternalSettings instance.
        path = self.internal_settings_path('experiment')
        self._internal_settings['experiment'] = _create_internal_settings(path)

    def user_settings_path(self, scope):
        assert scope in ('global', 'experiment')
        if scope == 'global':
            return op.join(self.phy_user_dir, 'user_settings.py')
        elif scope == 'experiment':
            return op.join(self.phy_experiment_dir, 'user_settings.py')

    def internal_settings_path(self, scope):
        assert scope in ('global', 'experiment')
        root = {'global': self.phy_user_dir,
                'experiment': self.phy_experiment_dir}
        return op.join(root[scope], 'internal_settings')

    def get_internal_settings(self, key, scope='global'):
        return self._internal_settings[scope].get(key)

    def set_internal_settings(self, key, value, scope='global'):
        return self._internal_settings[scope].set(key, value)

    def get_user_settings(self, key, scope='global'):
        if scope == 'experiment':
            scope = self.experiment_name
        return get(key, scope=scope)

    def set_user_settings(self, key, value, scope='global',
                          path=None, file_namespace=None):
        if scope == 'experiment':
            scope = self.experiment_name
        return set(key, value, scope=scope,
                   path=path, file_namespace=file_namespace)

    def save(self):
        for scope, settings in self._internal_settings.items():
            path = self.internal_settings_path(scope)
            settings.save(path)
