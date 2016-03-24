# -*- coding: utf-8 -*-

"""Test plugin system."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from pytest import yield_fixture, raises

from ..plugin import (IPluginRegistry,
                      IPlugin,
                      get_plugin,
                      discover_plugins,
                      )
from .._misc import _write_text


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def no_native_plugins():
    # Save the plugins.
    plugins = IPluginRegistry.plugins
    IPluginRegistry.plugins = []
    yield
    IPluginRegistry.plugins = plugins


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_plugin_1(no_native_plugins):
    class MyPlugin(IPlugin):
        pass

    assert IPluginRegistry.plugins == [MyPlugin]
    assert get_plugin('MyPlugin').__name__ == 'MyPlugin'

    with raises(ValueError):
        get_plugin('unknown')


def test_discover_plugins(tempdir, no_native_plugins):
    path = op.join(tempdir, 'my_plugin.py')
    contents = '''from phy import IPlugin\nclass MyPlugin(IPlugin): pass'''
    _write_text(path, contents)

    plugins = discover_plugins([tempdir])
    assert plugins
    assert plugins[0].__name__ == 'MyPlugin'
