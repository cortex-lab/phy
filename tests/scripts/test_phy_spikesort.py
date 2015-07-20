# -*- coding: utf-8 -*-1

"""Tests of phy spike sorting commands."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.scripts import main


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_version():
    main('-v')


def test_quick_start(chdir_tempdir):
    import os
    print(os.getcwd())
    main('download hybrid_10sec.dat')
