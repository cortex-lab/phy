# -*- coding: utf-8 -*-

"""Test config."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from textwrap import dedent

from pytest import yield_fixture
from traitlets import Float
from traitlets.config import Configurable

from .. import config as _config
from .._misc import _write_text
from ..config import (_ensure_dir_exists,
                      load_config,
                      load_master_config,
                      save_config,
                      )


#------------------------------------------------------------------------------
# Test config
#------------------------------------------------------------------------------

def test_phy_config_dir():
    assert _config.phy_config_dir().endswith('.phy')


def test_ensure_dir_exists(tempdir):
    path = op.join(tempdir, 'a/b/c')
    _ensure_dir_exists(path)
    assert op.isdir(path)


def test_temp_config_dir(temp_config_dir):
    assert _config.phy_config_dir() == temp_config_dir


#------------------------------------------------------------------------------
# Config tests
#------------------------------------------------------------------------------

@yield_fixture
def py_config(tempdir):
    # Create and load a config file.
    config_contents = """
       c = get_config()
       c.MyConfigurable.my_var = 1.0
       """
    path = op.join(tempdir, 'config.py')
    _write_text(path, config_contents)
    yield path


@yield_fixture
def json_config(tempdir):
    # Create and load a config file.
    config_contents = """
       {
          "MyConfigurable": {
            "my_var": 1.0
            }
        }
    """
    path = op.join(tempdir, 'config.json')
    _write_text(path, config_contents)
    yield path


@yield_fixture(params=['python', 'json'])
def config(py_config, json_config, request):
    if request.param == 'python':
        yield py_config
    elif request.param == 'json':
        yield json_config


def test_load_config(config):

    class MyConfigurable(Configurable):
        my_var = Float(0.0, config=True)

    assert MyConfigurable().my_var == 0.0

    c = load_config(config)
    assert c.MyConfigurable.my_var == 1.0

    # Create a new MyConfigurable instance.
    configurable = MyConfigurable()
    assert configurable.my_var == 0.0

    # Load the config object.
    configurable.update_config(c)
    assert configurable.my_var == 1.0


def test_load_master_config(temp_config_dir):
    # Create a config file in the temporary user directory.
    config_contents = dedent("""
       c = get_config()
       c.MyConfigurable.my_var = 1.0
       """)
    with open(op.join(temp_config_dir, 'phy_config.py'), 'w') as f:
        f.write(config_contents)

    # Load the master config file.
    c = load_master_config()
    assert c.MyConfigurable.my_var == 1.


def test_save_config(tempdir):
    c = {'A': {'b': 3.}}
    path = op.join(tempdir, 'config.json')
    save_config(path, c)

    c1 = load_config(path)
    assert c1.A.b == 3.
