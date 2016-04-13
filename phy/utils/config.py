# -*- coding: utf-8 -*-

"""Config."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
from textwrap import dedent

from traitlets.config import (Config,
                              PyFileConfigLoader,
                              JSONFileConfigLoader,
                              )

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Config
#------------------------------------------------------------------------------

def phy_config_dir():
    """Return the absolute path to the phy user directory."""
    home = op.realpath(op.expanduser('~'))
    return op.join(home, '.phy')


def _ensure_dir_exists(path):
    """Ensure a directory exists."""
    if not op.exists(path):
        os.makedirs(path)
    assert op.exists(path) and op.isdir(path)


def load_config(path):
    """Load a Python or JSON config file."""
    if not op.exists(path):
        return Config()
    path = op.realpath(path)
    dirpath, filename = op.split(path)
    file_ext = op.splitext(path)[1]
    logger.debug("Load config file `%s`.", path)
    if file_ext == '.py':
        config = PyFileConfigLoader(filename, dirpath,
                                    log=logger).load_config()
    elif file_ext == '.json':
        config = JSONFileConfigLoader(filename, dirpath,
                                      log=logger).load_config()
    return config


def _default_config(config_dir=None):
    path = op.join(config_dir or op.join('~', '.phy'), 'plugins/')
    return dedent("""
    # You can also put your plugins in ~/.phy/plugins/.

    from phy import IPlugin

    try:
        import phycontrib
    except:
        pass

    # Plugin example:
    #
    # class MyPlugin(IPlugin):
    #     def attach_to_cli(self, cli):
    #         # you can create phy subcommands here with click
    #         pass

    c = get_config()
    c.Plugins.dirs = [r'{}']
    """.format(path))


def load_master_config(config_dir=None):
    """Load a master Config file from `~/.phy/phy_config.py`."""
    config_dir = config_dir or phy_config_dir()
    path = op.join(config_dir, 'phy_config.py')
    # Create a default config file if necessary.
    if not op.exists(path):
        _ensure_dir_exists(op.dirname(path))
        logger.debug("Creating default phy config file at `%s`.", path)
        with open(path, 'w') as f:
            f.write(_default_config(config_dir=config_dir))
    assert op.exists(path)
    return load_config(path)


def save_config(path, config):
    """Save a config object to a JSON file."""
    import json
    config['version'] = 1
    with open(path, 'w') as f:
        json.dump(config, f)
