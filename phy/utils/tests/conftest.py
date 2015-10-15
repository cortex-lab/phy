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
    from .. import _misc
    _misc.phy_user_dir()
    ```

    and not:

    ```python
    from _misc import phy_user_dir
    ```

    Otherwise, the monkey patching hack in tests won't work.

    """
    from phy.utils import _misc

    user_dir = _misc.phy_user_dir
    _misc.phy_user_dir = lambda: tempdir
    yield tempdir
    _misc.phy_user_dir = user_dir
