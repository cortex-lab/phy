# -*- coding: utf-8 -*-

"""Settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

from ..ext import six
from ._misc import Bunch


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
# Settings
#------------------------------------------------------------------------------

class BaseSettings(object):
    """Store key-value pairs."""
    def __init__(self):
        self._store = {'global': {}}

    @property
    def _namespaces(self):
        return sorted(self._store['global'])

    def get(self, name, namespace=None, scope='global'):
        namespace, name = _split_namespace(name, namespace=namespace)
        if scope not in self._store:
            scope = 'global'
        return self._store[scope].get(namespace, {}).get(name, None)

    def set(self,
            key_values=None,
            namespace=None,
            scope='global',
            ):
        assert isinstance(key_values, dict)
        if scope not in self._store:
            self._store[scope] = Bunch({})
        for key, value in key_values.items():
            namespace, name = _split_namespace(key, namespace=namespace)

            # Create dictionaries if they do not exist.
            if namespace not in self._store[scope]:
                self._store[scope][namespace] = Bunch({})

            # Update the settings.
            self._store[scope][namespace][name] = value


class Settings(BaseSettings):
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

    def set(self,
            key_values=None,
            namespace=None,
            scope='global',
            path=None,
            file_namespace=None,
            ):
        if path is not None:
            path = op.expanduser(path)
            path = op.realpath(path)
            assert op.exists(path)
            return self.read_settings_file(path,
                                           file_namespace=file_namespace)
        super(Settings, self).set(key_values=key_values,
                                  namespace=namespace,
                                  scope=scope,
                                  )


#------------------------------------------------------------------------------
# Internal Settings
#------------------------------------------------------------------------------

class InternalSettings(BaseSettings):
    pass


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

_SETTINGS = Settings()


def get(name, scope='global'):
    """Get a settings value.

    Parameters
    ----------
    name : str
        The settings name
    scope : str (default is 'global')
        The scope for that setting. Can be 'global' or a dataset name.

    """
    return _SETTINGS.get(name, scope)


def set(key_values=None,
        namespace=None,
        scope='global',
        path=None,
        file_namespace=None,
        ):
    """Set some settings

    Parameters
    ----------
    key_values : dict
        A {str: object} dictionary with key-value pair settings.
    namespace : str
        The namespace if it is not specified in the keys.
    scope : str (default is 'global')
        The scope for that setting. Can be 'global' or a dataset name.
    path : str
        A path to a Python settings file.
    file_namespace : dict
        A namespace to pass to the Python settings file.

    """
    return _SETTINGS.set(key_values=key_values,
                          namespace=namespace,
                          scope=scope,
                          path=path,
                          file_namespace=file_namespace,
                          )
