# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..phy_script import ParserCreator, main


#------------------------------------------------------------------------------
# Script tests
#------------------------------------------------------------------------------

def test_parse_version():
    p = ParserCreator()
    p.parse(['--version'])


def test_parse_cluster_manual():
    p = ParserCreator()
    args = p.parse(['cluster-manual', 'test', '-i', '--debug'])
    assert args.command == 'cluster-manual'
    assert args.ipython
    assert args.debug
    assert not args.profiler
    assert not args.line_profiler


def test_parse_cluster_auto():
    p = ParserCreator()
    args = p.parse(['cluster-auto', 'test', '-lp'])
    assert args.command == 'cluster-auto'
    assert not args.ipython
    assert not args.debug
    assert not args.profiler
    assert args.line_profiler


def test_download(chdir_tempdir):
    main('download hybrid_10sec.prm')
