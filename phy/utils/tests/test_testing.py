# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import time

from vispy.app import Canvas

from ..testing import (benchmark, captured_output, show_test,
                       )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_captured_output():
    with captured_output() as (out, err):
        print('Hello world!')
    assert out.getvalue().strip() == 'Hello world!'


def test_benchmark():
    with benchmark():
        time.sleep(.002)


def test_canvas():
    c = Canvas(keys='interactive')
    with benchmark():
        show_test(c)
