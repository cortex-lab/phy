# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_array_equal as ae

from ..wizard import (_previous,
                      _next,
                      _find_first,
                      Wizard,
                      _wizard_group,
                      )


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_wizard_group():
    assert _wizard_group('noise') == 'ignored'
    assert _wizard_group('mua') == 'ignored'
    assert _wizard_group('good') == 'good'
    assert _wizard_group('unknown') is None
    assert _wizard_group(None) is None


def test_utils():
    l = [2, 3, 5, 7, 11]

    def func(x):
        return x in (2, 5)

    _find_first([], None)

    _previous([], None)
    _previous([0, 1], 1, lambda x: x > 0)
    # Error: log and do nothing.
    _previous(l, 1, func)
    _previous(l, 15, func)

    assert _previous(l, 2, func) == 2
    assert _previous(l, 3, func) == 2
    assert _previous(l, 5, func) == 2
    assert _previous(l, 7, func) == 5
    assert _previous(l, 11, func) == 5

    _next([], None)
    # Error: log and do nothing.
    _next(l, 1, func)
    _next(l, 15, func)

    assert _next(l, 2, func) == 5
    assert _next(l, 3, func) == 5
    assert _next(l, 5, func) == 5
    assert _next(l, 7, func) == 7
    assert _next(l, 11, func) == 11


def test_wizard_core():

    wizard = Wizard()
    wizard.set_cluster_ids_function(lambda: [2, 3, 5])

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


def test_wizard_nav(wizard):

    # Loop over the best clusters.
    wizard.start()

    assert wizard.n_clusters == 4
    assert wizard.best_list == [3, 2, 7, 5]

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
    assert wizard.match_list == [2, 7, 5]

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

    wizard.first()
    assert wizard.selection == [3, 2]
    wizard.last()
    assert wizard.selection == [3, 5]

    wizard.unpin()
    assert wizard.best == 3
    assert wizard.match is None

    assert wizard.n_processed == 2


def test_wizard_update_simple(wizard, clustering, cluster_meta):
    # 2: none, 3: none, 5: ignored, 7: good
    wizard.attach(clustering, cluster_meta)

    wizard.first()
    wizard.last()

    wizard.start()

    wizard.first()
    wizard.last()

    wizard.pin()

    wizard.first()
    wizard.last()

    wizard.pin()
    wizard.previous_best()
    wizard.next_best()


def test_wizard_update_group(wizard, clustering, cluster_meta):
    wizard.attach(clustering, cluster_meta)

    wizard.start()

    def _check_best_match(b, m):
        assert wizard.selection == [b, m]
        assert wizard.best == b
        assert wizard.match == m

    wizard.pin()
    _check_best_match(3, 2)
    # print(wizard.best_list)

    # Ignore the currently-pinned cluster.
    cluster_meta.set('group', 3, 'noise')
    # 2: none, 3: ignored, 5: ignored, 7: good
    _check_best_match(5, 2)
    return

    # Ignore the current match and move to next.
    cluster_meta.set('group', 2, 'mua')
    _check_best_match(5, 7)
    # 2: ignored, 3: ignored, 5: ignored, 7: good

    cluster_meta.undo()
    _check_best_match(5, 2)

    cluster_meta.redo()
    _check_best_match(5, 7)

    # Now move 3 to good.
    for _ in range(5):
        cluster_meta.undo()
    wizard.selection = (3, 2)
    _check_best_match(3, 2)
    cluster_meta.set('group', 3, 'good')
    _check_best_match(5, 2)


def test_wizard_update_clustering(wizard, clustering, cluster_meta):
    # 2: none, 3: none, 5: ignored, 7: good
    wizard.attach(clustering, cluster_meta)
    wizard.start()

    def _check_best_match(b, m):
        assert wizard.selection == [b, m]
        assert wizard.best == b
        assert wizard.match == m

    assert wizard.best_list == [3, 2, 7, 5]
    wizard.next()
    wizard.pin()

    _check_best_match(2, 3)
    cluster_meta.set('group', 2, 'good')
    wizard.selection = [2, 3]

    ################################

    assert wizard.cluster_status(2) == 'good'
    assert wizard.cluster_status(3) is None
    clustering.merge([2, 3])  # => 8
    _check_best_match(8, 7)
    assert wizard.best_list == [8, 7, 5]
    assert wizard.cluster_status(8) is None
    assert wizard.cluster_status(7) == 'good'
    assert wizard.cluster_status(2) == 'good'

    # Undo merge.
    clustering.undo()
    _check_best_match(2, 3)
    assert wizard.cluster_status(2) == 'good'
    assert wizard.cluster_status(3) is None

    # Make a selection.
    wizard.selection = [1, 5, 7, 8]
    _check_best_match(5, 7)

    # Redo merge.
    clustering.redo()
    _check_best_match(8, 7)
    assert wizard.cluster_status(8) is None
    assert wizard.cluster_status(7) == 'good'

    ################################

    # Split.
    ae(clustering.spike_clusters, [8, 8, 5, 7])
    clustering.split([1, 2])  # ==> 9, 10
    ae(clustering.spike_clusters, [10, 9, 9, 7])
    _check_best_match(9, 10)
    assert wizard.cluster_status(10) is None
    assert wizard.cluster_status(9) is None

    # Ignore a cluster.
    cluster_meta.set('group', 9, 'noise')
    assert wizard.cluster_status(9) == 'ignored'

    # Undo split.
    up = clustering.undo()
    _check_best_match(8, 7)
    assert up.description == 'assign'
    assert up.history == 'undo'

    # Redo split.
    up = clustering.redo()
    _check_best_match(9, 10)
    assert up.description == 'assign'
    assert up.history == 'redo'
    assert wizard.cluster_status(9) == 'ignored'

    ################################

    # Split (=merge).
    ae(clustering.spike_clusters, [10, 9, 9, 7])
    up = clustering.split([1, 2])
    ae(clustering.spike_clusters, [10, 11, 11, 7])
    _check_best_match(11, 7)
    assert up.description == 'merge'
    assert up.history is None
    assert wizard.cluster_status(11) == 'ignored'

    # Undo split (=merge).
    up = clustering.undo()
    _check_best_match(9, 10)
    assert up.description == 'merge'
    assert up.history == 'undo'

    # Redo split (=merge).
    up = clustering.redo()
    _check_best_match(11, 7)
    assert up.description == 'merge'
    assert up.history == 'redo'
