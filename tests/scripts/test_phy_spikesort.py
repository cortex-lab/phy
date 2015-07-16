# -*- coding: utf-8 -*-1

"""Tests of phy spike sorting commands."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from subprocess import call

from phy.io.kwik.mock import create_mock_kwik


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _call(cmd):
    ret = call(cmd.split(' '))
    if ret != 0:
        raise RuntimeError()
