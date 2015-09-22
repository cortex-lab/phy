# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import time

import numpy as np
from vispy.app import Canvas

from ..testing import (benchmark, captured_output, show_test,
                       _assert_equal,
                       )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_captured_output():
    with captured_output() as (out, err):
        print('Hello world!')
    assert out.getvalue().strip() == 'Hello world!'


def test_assert_equal():
    d = {'a': {'b': np.random.rand(5), 3: 'c'}, 'b': 2.}
    _assert_equal(d, d.copy())


def test_benchmark():
    with benchmark():
        time.sleep(.002)


def test_canvas():
    c = Canvas(keys='interactive')
    with benchmark():
        show_test(c)
