# -*- coding: utf-8 -*-

"""Test automaton."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import bisect
from pprint import pprint
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
    if not clusters:
        return default_first()
    assert len(clusters) > 0
    cl = clusters[0]
    if cl not in CLUSTERS:
        return default_last()
    while cl in clusters:
        cl = CLUSTERS[bisect.bisect_right(CLUSTERS, cl)]
    assert cl not in clusters
    return cl


def default_new_cluster_id():
    return N + 1


def default_next(clusters=None):
    if not clusters:
        return default_first()
    cl = clusters[-1]
    if cl not in CLUSTERS or cl == default_last():
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


def default_merge(clusters=None, to=None):
    return to


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


def test_automaton_nav_0(automaton):
    a = automaton
    _assert(a, [], [])

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.first()
    _assert(a, [1], [])

    a.last()
    _assert(a, [30], [])

    a.set_state([10])
    _assert(a, [10])

    a.last()
    _assert(a, [30], [])

    a.first()
    _assert(a, [1], [])


def test_automaton_nav_best_1(automaton):
    a = automaton
    _assert(a, [], [])

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    for clu in UNMASKED:
        a.next_best()
        _assert(a, [clu], [])

    for clu in UNMASKED[:-1:-1]:
        a.prev_best()
        _assert(a, [clu], [])


def test_automaton_nav_best_2(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N

    a.set_state([0])
    _assert(a, [0])

    for clu in UNMASKED:
        a.next_best()
        _assert(a, [clu], [])

    # One more next_best should not change the state if we're
    # already on the last best.
    a.next_best()
    _assert(a, [clu], [])

    for clu in UNMASKED[:-1:-1]:
        a.prev_best()
        _assert(a, [clu], [])


def test_automaton_nav_similar(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N

    a.set_state([0])
    _assert(a, [0])

    for clu in UNMASKED:
        a.next()
        _assert(a, [0], [clu])

    for clu in UNMASKED[:-1:-1]:
        a.prev()
        _assert(a, [0], [clu])


def test_automaton_nav_next(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N
    # UNMASKED = [1, 2, 11, 20, 30]

    # Wizard.
    a.next()
    _assert(a, [1], [2])

    a.next()
    _assert(a, [1], [11])

    a.next()
    _assert(a, [1], [20])

    a.next()
    _assert(a, [1], [30])

    a.next()
    _assert(a, [2], [10])


def test_automaton_merge_1(automaton):
    a = automaton

    a.set_state([30, 20], [])
    _assert(a, [30, 20])

    _l = []

    @a.connect
    def on_merge(before, after, **kwargs):
        _l.append((before, after))

    a.merge(to=31)
    _assert(a, [31], [])

    assert _l[0][0].clusters == [30, 20]
    assert _l[0][1].clusters == [31]


def test_automaton_merge_2(automaton):
    a = automaton

    a.set_state([30], [20])
    _assert(a, [30], [20])

    a.merge(to=31)
    _assert(a, [31], [30])


def test_automaton_merge_move(automaton):
    a = automaton

    a.set_state([20, 11], [])
    _assert(a, [20, 11])

    a.merge(to=31)
    _assert(a, [31], [])

    a.move(group='good', which='all')
    _assert(a, [30], [])


def test_automaton_split_1(automaton):
    a = automaton

    a.set_state([30, 20], [])
    _assert(a, [30, 20])

    a.split()
    _assert(a, [31], [])


def test_automaton_split_2(automaton):
    a = automaton

    a.set_state([30], [20])
    _assert(a, [30], [20])

    a.split()
    _assert(a, [31], [30])


def test_automaton_label(automaton):
    a = automaton
    a.set_state([30, 20], [])
    a.label()
    _assert(a, [30, 20])


def test_automaton_move_1(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.set_state([1, 2])

    a.move(group='good', which='best')
    _assert(a, [11])


def test_automaton_move_2(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.set_state([1, 2], [0])

    a.move(group='good', which='all')
    _assert(a, [11], [20])


def test_automaton_move_3(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N, N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.set_state([1, 2], [0])

    a.move(group='good', which='similar')
    _assert(a, [1, 2], [1])


def test_automaton_history_1(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N
    # UNMASKED = [1, 2, 11, 20, 30]

    # These should not add anything to the undo stack (history).
    a.next()
    a.next()
    a.next()

    # Wizard.
    a.set_state([2], [10])

    _l = []

    @a.connect
    def on_undo(name, before, after, **kwargs):
        _l.append(name)

    # Merge.
    assert not a.can_undo()
    assert not a.can_redo()
    a.merge(to=31)
    _assert(a, [31], [30])
    assert not _l

    # Undo.
    assert a.can_undo()
    assert not a.can_redo()
    a.undo()
    _assert(a, [2], [10])
    assert _l == ['merge']

    _l = []

    @a.connect
    def on_redo(name, before, after, **kwargs):
        _l.append(name)

    # Redo.
    assert not a.can_undo()
    assert a.can_redo()
    a.redo()
    _assert(a, [31], [30])
    assert _l == ['merge']


def test_automaton_history_move_1(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.set_state([10])
    _assert(a, [10])

    a.move(which='all', group='noise')
    _assert(a, [11])

    a.undo()
    _assert(a, [10])

    a.redo()
    _assert(a, [11])


def test_automaton_history_move_2(automaton):
    a = automaton

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N
    # UNMASKED = [1, 2, 11, 20, 30]

    a.set_state([20], [2])
    _assert(a, [20], [2])

    a.move(which='similar', group='noise')
    _assert(a, [20], [11])

    a.undo()
    _assert(a, [20], [2])

    a.redo()
    _assert(a, [20], [11])
