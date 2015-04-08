# -*- coding: utf-8 -*-

"""Settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

from ..ext import six
from ._misc import Bunch


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _read_settings_file(path):
    """Return a dictionary {namespace: {key: value}} dictionary."""
    pass


def _split_namespace(name, namespace=None):
    if '.' in name:
        namespace, name = name.split('.')
    if namespace is None:
        raise ValueError("The namespace must be specified.")
    return namespace, name


#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

class _Settings(object):
    def __init__(self):
        self._store = {'global': {}}

    def _get(self, name, namespace=None, scope='global'):
        namespace, name = _split_namespace(name, namespace=namespace)
        if scope not in self._store:
            scope = 'global'
        return self._store[scope].get(namespace, {}).get(name, None)

    def _set(self, key_values, namespace=None, scope='global', path=None):
        if path is not None and op.exists(path):
            return _read_settings_file(path)
        if scope not in self._store:
            self._store[scope] = {}
        for key, value in key_values.items():
            namespace, name = _split_namespace(key, namespace=namespace)

            # Create dictionaries if they do not exist.
            if namespace not in self._store[scope]:
                self._store[scope][namespace] = {}

            # Update the settings.
            self._store[scope][namespace][name] = value


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

_SETTINGS = _Settings()


def get(name, scope='global'):
    """Get a settings value.

    Parameters
    ----------
    name : str
        The settings name
    scope : str (default is 'global')
        The scope for that setting. Can be 'global' or a dataset name.

    """
    return _SETTINGS._get(name, scope)


def set(namespace_or_path, key_values=None, scope='global'):
    """Set some settings

    Parameters
    ----------
    namespace_or_path : str
        Either a path to a Python settings file, or a namespace.
    key_values : dict
        A {str: object} dictionary with key-value pair settings.
    scope : str (default is 'global')
        The scope for that setting. Can be 'global' or a dataset name.

    """
    return _SETTINGS._set(namespace_or_path, key_values, scope)
