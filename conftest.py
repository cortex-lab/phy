# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from phy.utils.tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

@yield_fixture
def tempdir():
    with TemporaryDirectory() as tempdir:
        yield tempdir


@yield_fixture
def tempdir_bis():
    with TemporaryDirectory() as tempdir:
        yield tempdir
