# -*- coding: utf-8 -*-

"""Plugin system.

Code from http://eli.thegreenplace.net/2012/08/07/fundamental-concepts-of-plugin-infrastructures  # noqa

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import imp
import logging
import os
import os.path as op

from six import with_metaclass

from . import _misc

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# IPlugin interface
#------------------------------------------------------------------------------

class IPluginRegistry(type):
    plugins = []

    def __init__(cls, name, bases, attrs):
        if name != 'IPlugin':
            logger.debug("Register plugin %s.", name)
            plugin_tuple = (cls,)
            if plugin_tuple not in IPluginRegistry.plugins:
                IPluginRegistry.plugins.append(plugin_tuple)


class IPlugin(with_metaclass(IPluginRegistry)):
    def attach_to_gui(self, gui):
        pass

    def attach_to_cli(self, cli):
        pass


def get_plugin(name):
    """Get a plugin class from its name."""
    name = name.lower()
    for (plugin,) in IPluginRegistry.plugins:
        if name in plugin.__name__.lower():
            return plugin
    raise ValueError("The plugin %s cannot be found." % name)


#------------------------------------------------------------------------------
# Plugins discovery
#------------------------------------------------------------------------------

def discover_plugins(dirs):
    """Discover the plugin classes contained in Python files.

    Parameters
    ----------

    dirs : list
        List of directory names to scan.

    Returns
    -------

    plugins : list
        List of plugin classes.

    """
    # Scan all subdirectories recursively.
    for plugin_dir in dirs:
        plugin_dir = op.realpath(plugin_dir)
        for subdir, dirs, files in os.walk(plugin_dir):
            # Skip test folders.
            base = op.basename(subdir)
            if 'test' in base or '__' in base:  # pragma: no cover
                continue
            logger.debug("Scanning %s.", subdir)
            for filename in files:
                if (filename.startswith('__') or
                        not filename.endswith('.py')):
                    continue  # pragma: no cover
                logger.debug("  Found %s.", filename)
                path = os.path.join(subdir, filename)
                modname, ext = op.splitext(filename)
                file, path, descr = imp.find_module(modname, [subdir])
                if file:
                    # Loading the module registers the plugin in
                    # IPluginRegistry.
                    try:
                        mod = imp.load_module(modname, file,
                                              path, descr)  # noqa
                    except Exception as e:
                        logger.exception(e)
    return IPluginRegistry.plugins


def _builtin_plugins_dir():
    return op.realpath(op.join(op.dirname(__file__), '../plugins/'))


def _user_plugins_dir():
    return op.expanduser(op.join(_misc.phy_user_dir(), 'plugins/'))


def get_all_plugins(config=None):
    """Load all builtin and user plugins."""
    # By default, builtin and default user plugin.
    dirs = [_builtin_plugins_dir(), _user_plugins_dir()]
    # Add Plugins.dir from the optionally-passed config object.
    if config:
        dirs += config.Plugins.dirs
    return [plugin for (plugin,) in discover_plugins(dirs)]
