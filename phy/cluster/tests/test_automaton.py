# -*- coding: utf-8 -*-

"""Test automaton."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import bisect
import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import fixture

from ..automaton import State, Transition, ClusterInfo, Automaton
from phylib.utils import connect, Bunch, emit


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

N = 30
CLUSTERS = [0, 1, 2, 10, 11, 20, 30]
#           i, g, N,  i,  g,  N, N
MASKED = [0, 10]
UNMASKED = [1, 2, 11, 20, 30]


def default_first():
    return 1


def default_last():
    return N


def default_similar(clusters):
    assert len(clusters) > 0
    if not clusters:
        return default_first()
    cl = clusters[0]
    if cl not in CLUSTERS:
        return default_last()
    return CLUSTERS[bisect.bisect_right(CLUSTERS, cl)]


def default_new_cluster_id():
    return N + 1


def default_next(clusters=None):
    if not clusters:
        return default_first()
    cl = clusters[0]
    if cl not in CLUSTERS:
        return None
    if cl == default_last():
        return default_last()
    return UNMASKED[bisect.bisect_right(UNMASKED, cl)]


def default_prev(clusters=None):
    if not clusters:
        return default_last()
    cl = clusters[0]
    i = UNMASKED.index(cl)
    if cl == default_first():
        return default_first()
    assert i >= 1
    return UNMASKED[i - 1]


def default_merge(clusters=None):
    return N + 1


def default_split(clusters=None):
    return [N + 1]


@fixture
def cluster_info():
    return ClusterInfo(
        first=default_first,
        last=default_last,
        similar=default_similar,
        new_cluster_id=default_new_cluster_id,
        next_best=default_next,
        prev_best=default_prev,
        next_similar=default_next,
        prev_similar=default_prev,
        merge=default_merge,
        split=default_split,
    )


@fixture
def automaton(cluster_info):
    s = State(clusters=[])
    return Automaton(s, cluster_info)


#------------------------------------------------------------------------------
# Test automaton
#------------------------------------------------------------------------------

def _assert(a: Automaton, cl: list[int], sim: list[int] = None):
    assert a.current_clusters() == cl
    assert a.current_similar() == (sim or [])


def test_automaton_state_1():
    s = State(clusters=[1, 2])
    assert s.similar == []


def test_automaton_transition_1():
    s = Transition(name='transition', before=None, after=None)


def test_automaton_1(cluster_info):
    s = State(clusters=[0])
    a = Automaton(s, cluster_info)
    _assert(a, [0], [])

    assert not a.can_undo()
    assert not a.can_redo()
    assert a.history_length() == 1

    a.set_state([1])
    _assert(a, [1], [])


def test_automaton_skip_best_1(automaton):
    a = automaton
    _assert(a, [], [])

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    for clu in UNMASKED:
        a.transition('next_best')
        _assert(a, [clu], [])

    for clu in UNMASKED[:-1:-1]:
        a.transition('prev_best')
        _assert(a, [clu], [])


def test_automaton_skip_best_2(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N

    a.set_state([0])
    _assert(a, [0])

    for clu in UNMASKED:
        a.transition('next_best')
        _assert(a, [clu], [])

    # One more next_best should not change the state if we're
    # already on the last best.
    a.transition('next_best')
    _assert(a, [clu], [])

    for clu in UNMASKED[:-1:-1]:
        a.transition('prev_best')
        _assert(a, [clu], [])


def test_automaton_skip_similar(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N

    a.set_state([0])
    _assert(a, [0])

    for clu in UNMASKED:
        a.transition('next')
        _assert(a, [0], [clu])

    for clu in UNMASKED[:-1:-1]:
        a.transition('prev')
        _assert(a, [0], [clu])


def test_automaton_merge_1(automaton):
    a = automaton

    a.set_state([30, 20], [])
    _assert(a, [30, 20])

    a.transition('merge')
    _assert(a, [31], [])


def test_automaton_merge_2(automaton):
    a = automaton

    a.set_state([30], [20])
    _assert(a, [30], [20])

    a.transition('merge')
    _assert(a, [31], [30])


def test_automaton_split_1(automaton):
    a = automaton

    a.set_state([30, 20], [])
    _assert(a, [30, 20])

    a.transition('split')
    _assert(a, [31], [])


def test_automaton_split_2(automaton):
    a = automaton

    a.set_state([30], [20])
    _assert(a, [30], [20])

    a.transition('split')
    _assert(a, [31], [30])
