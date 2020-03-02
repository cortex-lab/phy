# -*- coding: utf-8 -*-

"""Simple plugin system.

Code from http://eli.thegreenplace.net/2012/08/07/fundamental-concepts-of-plugin-infrastructures

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import imp
import logging
import os
from pathlib import Path

from phylib.utils._misc import _fullname
from .config import load_master_config

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# IPlugin interface
#------------------------------------------------------------------------------

class IPluginRegistry(type):
    """Regjster all plugin instances."""

    plugins = []

    def __init__(cls, name, bases, attrs):
        if name != 'IPlugin':
            logger.debug("Register plugin `%s`.", _fullname(cls))
            if _fullname(cls) not in (_fullname(_) for _ in IPluginRegistry.plugins):
                IPluginRegistry.plugins.append(cls)


class IPlugin(metaclass=IPluginRegistry):
    """All plugin classes should derive from this class.

    Plugin classes should just implement a method `attach_to_controller(self, controller)`.

    """
    pass


def get_plugin(name):
    """Get a plugin class from its name."""
    for plugin in IPluginRegistry.plugins:
        if name in plugin.__name__:
            return plugin
    raise ValueError("The plugin %s cannot be found." % name)


#------------------------------------------------------------------------------
# Plugins discovery
#------------------------------------------------------------------------------

def _iter_plugin_files(dirs):
    """Iterate through all found plugin files."""
    for plugin_dir in dirs:
        plugin_dir = Path(plugin_dir).expanduser()
        if not plugin_dir.exists():  # pragma: no cover
            continue
        for subdir, dirs, files in os.walk(plugin_dir, followlinks=True):
            subdir = Path(subdir)
            # Skip test folders.
            base = subdir.name
            if 'test' in base or '__' in base or '.git' in str(subdir):  # pragma: no cover
                continue
            logger.debug("Scanning `%s`.", subdir)
            for filename in files:
                if (filename.startswith('__') or not filename.endswith('.py')):
                    continue  # pragma: no cover
                logger.debug("Found plugin module `%s`.", filename)
                yield subdir / filename


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
    for path in _iter_plugin_files(dirs):
        subdir = path.parent
        modname = path.stem
        if modname in ('phy_config', 'phycontrib_loader'):
            continue
        file, path, descr = imp.find_module(modname, [subdir])
        if file:
            # Loading the module registers the plugin in
            # IPluginRegistry.
            try:
                mod = imp.load_module(modname, file, path, descr)  # noqa
            except Exception as e:  # pragma: no cover
                logger.exception(e)
            finally:
                file.close()
    return IPluginRegistry.plugins


def attach_plugins(controller, plugins=None, config_dir=None, dirs=None):
    """Attach plugins to a controller object.

    Attached plugins are those found in the user configuration file for the given gui_name or
    class name of the Controller instance, plus those specified in the plugins keyword argument.

    Parameters
    ----------

    controller : object
        The controller object that will be passed to the `attach_to_controller()` plugins methods.
    plugins : list of str
        List of plugin names to attach in addition to those found in the user configuration file.
    config_dir : str
        Path to the user configuration file. By default, the directory is `~/.phy/`.

    """

    plugins = plugins or []
    config = load_master_config(config_dir=config_dir)
    name = getattr(controller, 'gui_name', None) or controller.__class__.__name__
    c = config.get(name)
    # Discover plugin files in the plugin directories, as specified in the phy config file.
    dirs = (dirs or []) + config.get('Plugins', {}).get('dirs', [])
    discover_plugins(dirs)
    default_plugins = c.plugins if c else []
    if len(default_plugins):
        plugins = default_plugins + plugins
    logger.debug("Loading %d plugins.", len(plugins))
    attached = []
    for plugin in plugins:
        try:
            p = get_plugin(plugin)()
        except ValueError:  # pragma: no cover
            logger.warning("The plugin %s couldn't be found.", plugin)
            continue
        try:
            p.attach_to_controller(controller)
            attached.append(plugin)
            logger.debug("Attached plugin %s.", plugin)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "An error occurred when attaching plugin %s: %s.", plugin, e)
    return attached
