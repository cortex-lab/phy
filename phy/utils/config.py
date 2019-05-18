# -*- coding: utf-8 -*-

"""Config."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
from textwrap import dedent

from traitlets.config import (Config,
                              PyFileConfigLoader,
                              JSONFileConfigLoader,
                              )
from phylib.utils._misc import _ensure_dir_exists, phy_config_dir

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Config
#------------------------------------------------------------------------------

def load_config(path=None):
    """Load a Python or JSON config file."""
    if not path:
        return Config()
    path = Path(path)
    if not path.exists():  # pragma: no cover
        return Config()
    file_ext = path.suffix
    logger.debug("Load config file `%s`.", path)
    if file_ext == '.py':
        config = PyFileConfigLoader(path.name, str(path.parent), log=logger).load_config()
    elif file_ext == '.json':
        config = JSONFileConfigLoader(path.name, str(path.parent), log=logger).load_config()
    return config


def _default_config(config_dir=None):
    if not config_dir:  # pragma: no cover
        config_dir = Path.home() / '.phy'
    path = config_dir / 'plugins'
    return dedent("""
    # You can also put your plugins in ~/.phy/plugins/.

    from phy import IPlugin

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
    path = config_dir / 'phy_config.py'
    # Create a default config file if necessary.
    if not path.exists():
        _ensure_dir_exists(path.parent)
        logger.debug("Creating default phy config file at `%s`.", path)
        path.write_text(_default_config(config_dir=config_dir))
    assert path.exists()
    return load_config(path)


def save_config(path, config):
    """Save a config object to a JSON file."""
    import json
    config['version'] = 1
    with open(path, 'w') as f:
        json.dump(config, f)
