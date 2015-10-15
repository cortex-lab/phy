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
from traitlets import List, Unicode
from traitlets.config import Configurable

from ._misc import load_master_config, PHY_USER_DIR

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


class Plugins(Configurable):
    """Configure the list of user plugin directories.

    By default, there is only `~/.phy/plugins/`.

    """
    dirs = List(Unicode,
                default_value=[op.expanduser(op.join(PHY_USER_DIR,
                                                     'plugins/'))],
                config=True,
                )


def get_all_plugins():
    """Load all builtin and user plugins."""

    # Builtin plugins.
    builtin_plugins_dir = [op.realpath(op.join(op.dirname(__file__),
                                               '../plugins/'))]

    # Load the plugin dirs from all config files.
    plugins_config = Plugins()
    c = load_master_config()
    plugins_config.update_config(c)

    # Add the builtin dirs.
    dirs = builtin_plugins_dir + plugins_config.dirs

    # Return all loaded plugins.
    return [plugin for (plugin,) in discover_plugins(dirs)]
