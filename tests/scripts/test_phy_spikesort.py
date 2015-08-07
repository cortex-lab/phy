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


def test_cluster_auto_prm(chdir_tempdir):
    main('download hybrid_10sec.dat')
    main('download hybrid_10sec.prm')
    main('detect hybrid_10sec.prm')
    main('cluster-auto hybrid_10sec.prm --channel-group=0')


def test_quick_start(chdir_tempdir):
    main('download hybrid_10sec.dat')
    main('download hybrid_10sec.prm')
    main('spikesort hybrid_10sec.prm')
    # TODO: implement auto-close
    # main('cluster-manual hybrid_10sec.kwik')


# def test_traces(chdir_tempdir):
    # TODO: implement auto-close
    # main('download hybrid_10sec.dat')
    # main('traces --n-channels=32 --dtype=int16 '
    #      '--sample-rate=20000 --interval=0,3 hybrid_10sec.dat')
