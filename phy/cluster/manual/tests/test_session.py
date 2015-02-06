# -*- coding: utf-8 -*-

"""Tests of session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ....datasets.mock import artificial_spike_clusters, MockModel
from ..session import Session, CallbackManager


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

class _BaseView(object):
    is_loaded = False
    is_selected = False
    is_clustered = False

    def show(self):
        pass


class _MyView(_BaseView):
    pass


class _MyViewBis(_BaseView):
    pass


def test_callback_manager():
    model = MockModel()
    session = Session(model)

    cm = session._callback_manager

    # Create views.
    @cm.create("Show me")
    def show_me():
        view = _MyView()
        view.show()
        return view

    @cm.create("Show me bis")
    def show_me_bis():
        view = _MyViewBis()
        view.show()
        return view

    view = session.show_me()
    session.show_me()

    assert len(session._views) == 2

    view_bis = session.show_me_bis()

    # Test loading.
    @cm.load(_MyView)
    def loaded(view):
        assert isinstance(view, _MyView)
        view.is_loaded = True

    assert not view.is_loaded
    session._update_after_load()
    assert view.is_loaded

    # Test selection.
    @cm.select()
    def selected(view):
        assert isinstance(view, (_MyView, _MyViewBis))
        view.is_selected = True

    assert not view.is_selected
    session.select([0])
    assert view.is_selected

    # Test cluster.
    @cm.cluster(_MyViewBis)
    def clustered(view, up=None):
        assert isinstance(view, _MyViewBis)
        view.is_clustered = True

    assert not view_bis.is_clustered
    session.merge([0])
    assert view_bis.is_clustered


def test_session():

    # Mock model.
    model = MockModel()
    n_clusters = model.n_clusters

    with raises(ValueError):
        Session(None)

    session = Session(model)

    session._update_after_load()
    ae(session.cluster_labels, np.arange(n_clusters))
    assert len(session.cluster_colors) == n_clusters

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
