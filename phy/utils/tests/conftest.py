# -*- coding: utf-8 -*-

"""py.test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

@fixture
def temp_config_dir(tempdir):
    """NOTE: the user directory should be loaded with:

    ```python
    from .. import config
    config.phy_config_dir()
    ```

    and not:

    ```python
    from config import phy_config_dir
    ```

    Otherwise, the monkey patching hack in tests won't work.

    """
    from phy.utils import config

    config_dir = config.phy_config_dir
    config.phy_config_dir = lambda: tempdir
    yield tempdir
    config.phy_config_dir = config_dir
