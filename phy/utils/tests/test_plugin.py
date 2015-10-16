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
                      get_all_plugins,
                      )
from .._misc import _write_text
from ..config import load_master_config


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
       c.Plugins.dirs = ['{}']
    """
    _write_text(op.join(temp_user_dir, 'phy_config.py'),
                config_contents,
                op.join(temp_user_dir, 'my_plugins/'))


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


def test_discover_plugins(tempdir, no_native_plugins):
    path = op.join(tempdir, 'my_plugin.py')
    contents = '''from phy import IPlugin\nclass MyPlugin(IPlugin): pass'''
    _write_text(path, contents)

    plugins = discover_plugins([tempdir])
    assert plugins
    assert plugins[0][0].__name__ == 'MyPlugin'


def test_get_all_plugins(plugin):
    temp_user_dir, in_default_dir, path = plugin
    n_builtin_plugins = 0

    plugins = get_all_plugins()

    def _assert_loaded():
        assert len(plugins) == n_builtin_plugins + 1
        p = plugins[-1]
        assert p.__name__ == 'MyPlugin'

    if in_default_dir:
        # Create a plugin in the default plugins directory: it will be
        # discovered and automatically loaded by get_all_plugins().
        _assert_loaded()
    else:
        assert len(plugins) == n_builtin_plugins

        # This time, we write the custom plugins path in the config file.
        _write_my_plugins_dir_in_config(temp_user_dir)

        # We reload all plugins with the master config object.
        config = load_master_config()
        plugins = get_all_plugins(config)

        # This time, the plugin should be found.
        _assert_loaded()
