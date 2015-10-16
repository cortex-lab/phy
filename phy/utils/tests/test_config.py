# -*- coding: utf-8 -*-

"""Test config."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from textwrap import dedent

from traitlets import Float
from traitlets.config import Configurable

from .. import config as _config
from ..config import (_load_config,
                      load_master_config,
                      )


#------------------------------------------------------------------------------
# Test config
#------------------------------------------------------------------------------

def test_phy_user_dir():
    assert _config.phy_user_dir().endswith('.phy/')


def test_temp_user_dir(temp_user_dir):
    assert _config.phy_user_dir() == temp_user_dir


#------------------------------------------------------------------------------
# Config tests
#------------------------------------------------------------------------------

def test_load_config(tempdir):

    class MyConfigurable(Configurable):
        my_var = Float(0.0, config=True)

    assert MyConfigurable().my_var == 0.0

    # Create and load a config file.
    config_contents = dedent("""
       c = get_config()

       c.MyConfigurable.my_var = 1.0
       """)

    path = op.join(tempdir, 'config.py')
    with open(path, 'w') as f:
        f.write(config_contents)

    c = _load_config(path)
    assert c.MyConfigurable.my_var == 1.0

    # Create a new MyConfigurable instance.
    configurable = MyConfigurable()
    assert configurable.my_var == 0.0

    # Load the config object.
    configurable.update_config(c)
    assert configurable.my_var == 1.0


def test_load_master_config(temp_user_dir):
    # Create a config file in the temporary user directory.
    config_contents = dedent("""
       c = get_config()
       c.MyConfigurable.my_var = 1.0
       """)
    with open(op.join(temp_user_dir, 'phy_config.py'), 'w') as f:
        f.write(config_contents)

    # Load the master config file.
    c = load_master_config()
    assert c.MyConfigurable.my_var == 1.
