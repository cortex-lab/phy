# -*- coding: utf-8 -*-

"""Test automaton."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..automaton import State, Transition
from phylib.utils import connect, Bunch, emit


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def data():
    n = 30
    _data = [
        {"id": i,
         "n_spikes": n * 10 - 10 * i,
         "group": {2: 'noise', 3: 'noise', 5: 'mua', 8: 'good'}.get(i, None),
         "is_masked": i in (2, 3, 5),
         } for i in range(n)]
    return _data


#------------------------------------------------------------------------------
# Test automaton
#------------------------------------------------------------------------------

def test_automaton_state_1():
    s = State(clusters=[1, 2])
    assert s.next_cluster is None
    assert s.similar == []
    assert s.next_similar is None
