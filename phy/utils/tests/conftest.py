# -*- coding: utf-8 -*-

"""py.test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from phy.utils import _misc


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

@yield_fixture
def temp_user_dir(tempdir):
    user_dir = _misc.PHY_USER_DIR
    _misc.PHY_USER_DIR = tempdir
    yield tempdir
    _misc.PHY_USER_DIR = user_dir
