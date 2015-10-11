# -*- coding: utf-8 -*-

"""Test plugin system."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from ..plugin import (IPluginRegistry, IPlugin, get_plugin,
                      discover_plugins,
                      )

from pytest import yield_fixture, raises


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

    assert IPluginRegistry.plugins == [(MyPlugin,)]
    assert get_plugin('myplugin').__name__ == 'MyPlugin'

    with raises(ValueError):
        get_plugin('unknown')

    get_plugin('myplugin')().attach_to_gui(None)


def test_discover_plugins(tempdir, no_native_plugins):
    path = op.join(tempdir, 'my_plugin.py')
    contents = '''from phy import IPlugin\nclass MyPlugin(IPlugin): pass'''
    with open(path, 'w') as f:
        f.write(contents)

    plugins = discover_plugins([tempdir])
    assert plugins
    assert plugins[0][0].__name__ == 'MyPlugin'
