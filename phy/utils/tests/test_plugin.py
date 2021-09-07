# -*- coding: utf-8 -*-

"""Test plugin system."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

from pytest import fixture, raises

from ..plugin import (IPluginRegistry,
                      IPlugin,
                      get_plugin,
                      discover_plugins,
                      attach_plugins
                      )
from phylib.utils._misc import write_text


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
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
    path = tempdir / 'my_plugin.py'
    contents = '''from phy import IPlugin\nclass MyPlugin(IPlugin): pass'''
    write_text(path, contents)

    plugins = discover_plugins([tempdir])
    assert plugins
    assert plugins[0].__name__ == 'MyPlugin'


def test_attach_plugins(tempdir):
    class MyController(object):
        pass

    write_text(tempdir / 'plugin1.py', dedent(
        '''
            from phy import IPlugin
            class MyPlugin1(IPlugin):
                def attach_to_controller(self, controller):
                    controller.plugin1 = True
        '''))

    class MyPlugin2(IPlugin):
        def attach_to_controller(self, controller):
            controller.plugin2 = True

    contents = dedent('''
    c = get_config()
    c.Plugins.dirs = ['%s']
    c.MyController.plugins = ['MyPlugin1']
    ''' % tempdir)
    write_text(tempdir / 'phy_config.py', contents)

    controller = MyController()
    attach_plugins(controller, plugins=['MyPlugin2'], config_dir=tempdir)

    assert controller.plugin1 == controller.plugin2 is True
