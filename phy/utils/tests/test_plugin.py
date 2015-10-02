# -*- coding: utf-8 -*-

"""Test plugin system."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from ..plugin import (IPluginRegistry, IPlugin, get_plugin,
                      iter_plugins_dirs, _load_all_native_plugins)

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

def test_plugin_registration(no_native_plugins):
    class MyPlugin(IPlugin):
        pass

    assert IPluginRegistry.plugins == [(MyPlugin, ())]


def test_get_plugin():
    # assert get_plugin('jso').__name__ == 'JSON'
    # assert get_plugin('JSO').__name__ == 'JSON'
    # assert get_plugin('JSON').__name__ == 'JSON'
    # assert get_plugin('json').__name__ == 'JSON'
    # assert get_plugin('.json').__name__ == 'JSON'

    # with raises(ValueError):
    #     assert get_plugin('.jso') is None
    # with raises(ValueError):
    #     assert get_plugin('jsonn') is None
    pass


def test_iter_plugins_dirs():
    # assert 'json' in [op.basename(plugin_dir)
    #                   for plugin_dir in iter_plugins_dirs()]
    pass


def test_load_all_native_plugins(no_native_plugins):
    _load_all_native_plugins()
