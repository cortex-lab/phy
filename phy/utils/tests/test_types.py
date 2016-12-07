# -*- coding: utf-8 -*-

"""Tests of misc type utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import raises

from .._types import (Bunch, _bunchify, _is_integer, _is_list, _is_float,
                      _as_list, _is_array_like, _as_array, _as_tuple,
                      _as_scalar, _as_scalars,
                      )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_bunch():
    obj = Bunch()
    obj['a'] = 1
    assert obj.a == 1
    obj.b = 2
    assert obj['b'] == 2
    assert obj.copy() == obj


def test_bunchify():
    d = {'a': {'b': 0}}
    b = _bunchify(d)
    assert isinstance(b, Bunch)
    assert isinstance(b['a'], Bunch)


def test_number():
    assert not _is_integer(None)
    assert not _is_integer(3.)
    assert _is_integer(3)
    assert _is_integer(np.arange(1)[0])

    assert not _is_float(None)
    assert not _is_float(3)
    assert not _is_float(np.array([3])[0])
    assert _is_float(3.)
    assert _is_float(np.array([3.])[0])


def test_list():
    assert not _is_list(None)
    assert not _is_list(())
    assert _is_list([])

    assert _as_list(None) is None
    assert _as_list(3) == [3]
    assert _as_list([3]) == [3]
    assert _as_list((3,)) == [3]
    assert _as_list('3') == ['3']
    assert np.all(_as_list(np.array([3])) == np.array([3]))


def test_as_tuple():
    assert _as_tuple(3) == (3,)
    assert _as_tuple((3,)) == (3,)
    assert _as_tuple(None) is None
    assert _as_tuple((None,)) == (None,)
    assert _as_tuple((3, 4)) == (3, 4)
    assert _as_tuple([3]) == ([3], )
    assert _as_tuple([3, 4]) == ([3, 4], )


def test_as_scalar():
    assert _as_scalar(1) == 1
    assert _as_scalar(np.ones(1)[0]) == 1.
    assert type(_as_scalar(np.ones(1)[0])) == float

    assert _as_scalars(np.arange(3)) == [0, 1, 2]


def test_array():
    def _check(arr):
        assert isinstance(arr, np.ndarray)
        assert np.all(arr == [3])

    _check(_as_array(3))
    _check(_as_array(3.))
    _check(_as_array([3]))

    _check(_as_array(3, np.float))
    _check(_as_array(3., np.float))
    _check(_as_array([3], np.float))
    _check(_as_array(np.array([3])))
    with raises(ValueError):
        _check(_as_array(np.array([3]), dtype=np.object))
    _check(_as_array(np.array([3]), np.float))

    assert _as_array(None) is None
    assert not _is_array_like(None)
    assert not _is_array_like(3)
    assert _is_array_like([3])
    assert _is_array_like(np.array([3]))
