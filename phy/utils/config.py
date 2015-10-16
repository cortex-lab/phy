# -*- coding: utf-8 -*-

"""Config."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op

from traitlets.config import (Config,
                              PyFileConfigLoader,
                              JSONFileConfigLoader,
                              )

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Config
#------------------------------------------------------------------------------

def phy_user_dir():
    """Return the absolute path to the phy user directory."""
    return op.expanduser('~/.phy/')


def _ensure_dir_exists(path):
    """Ensure a directory exists."""
    if not op.exists(path):
        os.makedirs(path)
    assert op.exists(path) and op.isdir(path)


def _load_config(path):
    """Load a Python or JSON config file."""
    if not op.exists(path):
        return Config()
    path = op.realpath(path)
    dirpath, filename = op.split(path)
    file_ext = op.splitext(path)[1]
    if file_ext == '.py':
        config = PyFileConfigLoader(filename, dirpath).load_config()
    elif file_ext == '.json':
        config = JSONFileConfigLoader(filename, dirpath).load_config()
    return config


def load_master_config(user_dir=None):
    """Load a master Config file from `~/.phy/phy_config.py|json`."""
    user_dir = user_dir or phy_user_dir()
    c = Config()
    paths = [op.join(user_dir, 'phy_config.json'),
             op.join(user_dir, 'phy_config.py')]
    for path in paths:
        c.update(_load_config(path))
    return c


def save_config(path, config):
    """Save a config object to a JSON file."""
    import json
    config['version'] = 1
    with open(path, 'w') as f:
        json.dump(config, f)
