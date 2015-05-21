# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from .._utils import UpdateInfo
from ..wizard import (_previous,
                      _next,
                      Wizard,
                      )


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_utils():
    l = [2, 3, 5, 7, 11]

    def func(x):
        return x in (2, 5)

    with raises(RuntimeError):
        _previous(l, 1, func)
    with raises(RuntimeError):
        _previous(l, 15, func)

    assert _previous(l, 2, func) == 2
    assert _previous(l, 3, func) == 2
    assert _previous(l, 5, func) == 2
    assert _previous(l, 7, func) == 5
    assert _previous(l, 11, func) == 5

    with raises(RuntimeError):
        _next(l, 1, func)
    with raises(RuntimeError):
        _next(l, 15, func)

    assert _next(l, 2, func) == 5
    assert _next(l, 3, func) == 5
    assert _next(l, 5, func) == 5
    assert _next(l, 7, func) == 7
    assert _next(l, 11, func) == 11


def test_wizard_core():

    wizard = Wizard([2, 3, 5])

    @wizard.set_quality_function
    def quality(cluster):
        return {2: .9,
                3: .3,
                5: .6,
                }[cluster]

    @wizard.set_similarity_function
    def similarity(cluster, other):
        cluster, other = min((cluster, other)), max((cluster, other))
        return {(2, 3): 1,
                (2, 5): 2,
                (3, 5): 3}[cluster, other]

    assert wizard.best_clusters() == [2, 5, 3]
    assert wizard.best_clusters(n_max=0) == [2, 5, 3]
    assert wizard.best_clusters(n_max=None) == [2, 5, 3]
    assert wizard.best_clusters(n_max=2) == [2, 5]

    assert wizard.best_clusters(n_max=1) == [2]

    assert wizard.most_similar_clusters() == [5, 3]
    assert wizard.most_similar_clusters(2) == [5, 3]

    assert wizard.most_similar_clusters(n_max=0) == [5, 3]
    assert wizard.most_similar_clusters(n_max=None) == [5, 3]
    assert wizard.most_similar_clusters(n_max=1) == [5]


def test_wizard_nav():

    groups = {2: None, 3: None, 5: 'ignored', 7: 'good'}
    wizard = Wizard(groups)

    @wizard.set_quality_function
    def quality(cluster):
        return {2: .2,
                3: .3,
                5: .5,
                7: .7,
                }[cluster]

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return 1. + quality(cluster) - quality(other)

    # Loop over the best clusters.
    wizard.start()
    assert wizard.best == 3
    assert wizard.match is None

    wizard.next()
    assert wizard.best == 2

    wizard.previous()
    assert wizard.best == 3

    wizard.previous_best()
    assert wizard.best == 3

    wizard.next()
    assert wizard.best == 2

    wizard.next()
    assert wizard.best == 7

    wizard.next_best()
    assert wizard.best == 5

    wizard.next()
    assert wizard.best == 5

    # Now we start again.
    wizard.start()
    assert wizard.best == 3
    assert wizard.match is None

    # The match are sorted by group first (unsorted, good, and ignored),
    # and similarity second.
    wizard.pin()
    assert wizard.best == 3
    assert wizard.match == 2

    wizard.next()
    assert wizard.best == 3
    assert wizard.match == 7

    wizard.next_match()
    assert wizard.best == 3
    assert wizard.match == 5

    wizard.previous_match()
    assert wizard.best == 3
    assert wizard.match == 7

    wizard.previous()
    assert wizard.best == 3
    assert wizard.match == 2

    # Test HTML representation.
    html = wizard.get_panel()
    assert '>3<' in html
    assert '>2<' in html

    wizard.unpin()
    assert wizard.best == 3
    assert wizard.match is None


def test_wizard_actions():

    groups = {2: None, 3: None, 5: 'ignored', 7: 'good'}
    wizard = Wizard(groups)

    @wizard.set_quality_function
    def quality(cluster):
        return cluster / 50.

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return 1. + quality(cluster) - quality(other)

    def _assert_wizard(best, match):
        assert (wizard.best, wizard.match) == (best, match)

    def _move(clusters, group):
        info = UpdateInfo(description='metadata_group',
                          metadata_changed=clusters,
                          metadata_value=group,
                          )
        wizard.on_cluster(info)

    def _undo_move(clusters, old_group):
        info = UpdateInfo(description='metadata_group',
                          metadata_changed=clusters,
                          metadata_value=old_group,
                          history='undo',
                          )
        wizard.on_cluster(info)

    def _redo_move(clusters, group):
        info = UpdateInfo(description='metadata_group',
                          metadata_changed=clusters,
                          metadata_value=group,
                          history='redo',
                          )
        wizard.on_cluster(info)

    def _merge(clusters, new):
        info = UpdateInfo(description='merge',
                          added=[new],
                          deleted=clusters,
                          descendants=[(x, new) for x in clusters],
                          )
        wizard.on_cluster(info)

    def _undo_merge(clusters, new):
        info = UpdateInfo(description='assign',
                          added=clusters,
                          deleted=[new],
                          history='undo',
                          descendants=[(new, x) for x in clusters],
                          )
        wizard.on_cluster(info)

    def _redo_merge(clusters, new):
        info = UpdateInfo(description='assign',
                          added=[new],
                          deleted=clusters,
                          history='redo',
                          descendants=[(x, new) for x in clusters],
                          )
        wizard.on_cluster(info)

    # Loop over the best clusters.
    wizard.start()
    wizard.next()
    wizard.pin()
    _assert_wizard(2, 3)
    assert wizard.match_list == [3, 7, 5]

    _merge([2, 3], 20)
    _assert_wizard(20, 7)
    assert wizard.best_list == [20, 7, 5]
    assert wizard.match_list == [7, 5]

    # Simulate an undo and redo.
    _undo_merge([2, 3], 20)
    assert wizard.match_list == [3, 7, 5]
    _assert_wizard(2, 3)

    _redo_merge([2, 3], 20)
    _assert_wizard(20, 7)

    assert wizard.best_list == [20, 7, 5]
    assert wizard.match_list == [7, 5]

    wizard.next()
    _assert_wizard(20, 5)

    # Move.
    _move([20], 'good')
    _assert_wizard(7, 20)

    # Undo twice.
    _undo_move([20], None)
    _assert_wizard(20, 5)

    _undo_merge([2, 3], 20)
    _assert_wizard(2, 3)

    # Redo twice.
    _redo_merge([2, 3], 20)
    _assert_wizard(20, 7)

    _redo_move([20], 'good')
    _assert_wizard(7, 20)

    # End of wizard.
    wizard.last()
    _assert_wizard(7, 5)

    _move([5], 'noise')
    _assert_wizard(7, 5)

    wizard.next()
    _assert_wizard(7, 5)

    wizard.previous()
    _assert_wizard(7, 20)

    wizard.previous()
    _assert_wizard(7, 20)
