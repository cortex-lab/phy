# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..wizard import (_previous,
                      _next,
                      Wizard,
                      WizardPanel,
                      )


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_utils():
    l = [2, 3, 5, 7, 11]

    func = lambda x: x in (2, 5)

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


def test_panel():
    panel = WizardPanel()
    assert panel.html

    panel.best = 3
    assert panel.html

    panel.match = 10
    assert panel.html


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

    # Loop over the best clusters.
    wizard.start()
    wizard.next()
    wizard.pin()
    assert wizard.best == 2
    assert wizard.match == 3

    wizard.merge([2, 3], 20, None)
    assert wizard.best_list == [20, 7, 5]
    assert wizard.best == 20
    assert wizard.match == 7
    assert wizard.match_list == [7, 5]

    wizard.next()
    assert wizard.best == 20
    assert wizard.match == 5

    wizard.move(20, 'good')
    assert wizard.best == 7
    assert wizard.match == 20
