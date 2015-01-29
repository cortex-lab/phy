# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ....datasets.mock import artificial_spike_clusters, MockExperiment
from ..session import Session


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_session():

    # Mock experiment.
    exp = MockExperiment()
    n_clusters = exp.n_clusters

    with raises(ValueError):
        Session(None)

    session = Session(exp)

    # Views.
    view = session.show_waveforms()

    # Selection.
    session.select([1, 2])

    clusters_0 = np.array(np.arange(n_clusters))
    ae(session.clustering.cluster_labels, clusters_0)

    # Action 1: merge.
    session.merge([0, 1])
    clusters_1 = np.arange(2, n_clusters + 1)
    ae(session.clustering.cluster_labels, clusters_1)

    # Action 2: merge.
    session.merge([10, 2])
    clusters_2 = np.array([3, 4, 5, 6, 7, 8, 9, 11])
    ae(session.clustering.cluster_labels, clusters_2)

    # Undo 2.
    session.undo()
    ae(session.clustering.cluster_labels, clusters_1)

    # Undo 1.
    session.undo()
    ae(session.clustering.cluster_labels, clusters_0)

    # Redo 1.
    session.redo()
    ae(session.clustering.cluster_labels, clusters_1)

    # New set of actions.
    # Split.
    session.split([10, 20, 30])
    clusters_2_bis = np.arange(2, 12)
    ae(session.clustering.cluster_labels, clusters_2_bis)

    # Undo
    session.undo()
    ae(session.clustering.cluster_labels, clusters_1)

    # Redo
    session.redo()
    ae(session.clustering.cluster_labels, clusters_2_bis)

    # Move two clusters.
    session.move([9, 10], 1)
    assert session.cluster_metadata.get(9, 'group') == 1
    assert session.cluster_metadata.get(10, 'group') == 1

    session.undo()
    assert session.cluster_metadata.get(9, 'group') == 3
    assert session.cluster_metadata.get(10, 'group') == 3

    session.redo()
    assert session.cluster_metadata.get(9, 'group') == 1
    assert session.cluster_metadata.get(10, 'group') == 1

    # Wizard actions not implemented yet.
    with raises(NotImplementedError):
        session.wizard_start()
    with raises(NotImplementedError):
        session.wizard_next()
    with raises(NotImplementedError):
        session.wizard_previous()
    with raises(NotImplementedError):
        session.wizard_reset()

    session.unregister_view(view)
