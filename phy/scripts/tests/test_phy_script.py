# -*- coding: utf-8 -*-1

"""Tests of the script."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..phy_script import ParserCreator


#------------------------------------------------------------------------------
# Script tests
#------------------------------------------------------------------------------

def test_script_parser():

    p = ParserCreator()

    kwik_path = 'test'

    args = p.parse(['cluster-manual', kwik_path, '-i', '--debug'])
    assert args.command == 'cluster-manual'
    assert args.ipython
    assert args.debug
    assert not args.profiler
    assert not args.line_profiler

    args = p.parse(['cluster-auto', kwik_path, '-lp'])
    assert args.command == 'cluster-auto'
    assert not args.ipython
    assert not args.debug
    assert not args.profiler
    assert args.line_profiler
