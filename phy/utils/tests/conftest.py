# -*- coding: utf-8 -*-

"""py.test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

@yield_fixture
def temp_user_dir(tempdir):
    """NOTE: the user directory should be loaded with:

    ```python
    from .. import settings
    settings.phy_user_dir()
    ```

    and not:

    ```python
    from settings import phy_user_dir
    ```

    Otherwise, the monkey patching hack in tests won't work.

    """
    from phy.utils import settings

    user_dir = settings.phy_user_dir
    settings.phy_user_dir = lambda: tempdir
    yield tempdir
    settings.phy_user_dir = user_dir
