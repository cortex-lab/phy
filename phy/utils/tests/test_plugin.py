# -*- coding: utf-8 -*-

"""Test plugin system."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from textwrap import dedent

from traitlets import List, Unicode
from pytest import yield_fixture, raises

from ..plugin import (IPluginRegistry,
                      IPlugin,
                      Plugins,
                      get_plugin,
                      discover_plugins,
                      get_all_plugins,
                      )


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

    get_plugin('myplugin')().attach_to_cli(None)
    get_plugin('myplugin')().attach_to_gui(None)


def test_discover_plugins(tempdir, no_native_plugins):
    path = op.join(tempdir, 'my_plugin.py')
    contents = '''from phy import IPlugin\nclass MyPlugin(IPlugin): pass'''
    with open(path, 'w') as f:
        f.write(contents)

    plugins = discover_plugins([tempdir])
    assert plugins
    assert plugins[0][0].__name__ == 'MyPlugin'


def test_get_all_plugins(temp_user_dir):

    n_builtin_plugins = 0

    plugins = get_all_plugins()
    assert len(plugins) == n_builtin_plugins

    plugin_contents = dedent("""
    from phy import IPlugin
    class MyPlugin(IPlugin):
        pass
    """)

    # Create a plugin in some directory.
    os.mkdir(op.join(temp_user_dir, 'myplugins/'))
    with open(op.join(temp_user_dir, 'myplugins/myplugin.py'), 'w') as f:
        f.write(plugin_contents)

    # By default, this directory has no reason to be scanned, and the
    # plugin is not loaded.
    plugins = get_all_plugins()
    assert len(plugins) == n_builtin_plugins

    # Specify the path to the plugin in the phy config file..
    config_contents = dedent("""
       c = get_config()
       c.Plugins.dirs = ['%s']
       """ % op.join(temp_user_dir, 'myplugins/'))
    with open(op.join(temp_user_dir, 'phy_config.py'), 'w') as f:
        f.write(config_contents)

    # Now, reload all plugins.
    plugins = get_all_plugins()

    # This time, the plugin will be found.
    assert len(plugins) == n_builtin_plugins + 1
    p = plugins[-1]
    assert p.__name__ == 'MyPlugin'
