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


@yield_fixture(params=[(False, 'my_plugins/plugin.py'),
                       (True, 'plugins/plugin.py'),
                       ])
def plugin(no_native_plugins, temp_user_dir, request):
    path = op.join(temp_user_dir, request.param[1])
    contents = """
        from phy import IPlugin
        class MyPlugin(IPlugin):
            pass
    """
    _write_text(path, contents)
    yield temp_user_dir, request.param[0], request.param[1]


def _write_my_plugins_dir_in_config(temp_user_dir):
    # Now, we specify the path to the plugin in the phy config file.
    config_contents = """
       c = get_config()
       c.Plugins.dirs = [r'%s']
    """
    _write_text(op.join(temp_user_dir, 'phy_config.py'),
                config_contents % op.join(temp_user_dir, 'my_plugins/'))


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
