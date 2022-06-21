# -*- coding: utf-8 -*-

"""Test controller."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import bisect
from pprint import pprint
import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import fixture

from ..controller import Controller
from phylib.utils import connect, Bunch, emit
from phy.utils.context import Context


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def controller(
        cluster_ids, cluster_groups, cluster_labels, similarity, tempdir):

    spike_clusters = np.repeat(cluster_ids, 2 + np.arange(len(cluster_ids)))

    s = Controller(
        spike_clusters=spike_clusters,
        cluster_groups=cluster_groups,
        cluster_labels=cluster_labels,
        similarity=similarity,
        context=Context(tempdir),
    )
    return s


#------------------------------------------------------------------------------
# Test controller
#------------------------------------------------------------------------------

def test_controller_1(controller):
    c = controller

    assert len(c.cluster_info()) == 7
    assert c.n_spikes(30) == 8
    assert c.fields == ('test_label',)


def test_controller_merge_1(controller):
    c = controller

    # Split some spikes.
    up = c.merge([30, 20])
    assert up.description == 'merge'
    assert up.deleted == [20, 30]
    assert up.added == [31]

    assert list(c.cluster_ids) == [0, 1, 2, 10, 11, 31]

    # Undo the merge.
    c.undo()
    assert list(c.cluster_ids) == [0, 1, 2, 10, 11, 20, 30]

    # Redo the merge.
    c.redo()
    assert list(c.cluster_ids) == [0, 1, 2, 10, 11, 31]


def test_controller_split_1(controller):
    c = controller

    # Split some spikes.
    up = c.split(spike_ids=[1, 2])
    assert up.description == 'assign'
    assert up.deleted == [0, 1]
    assert up.added == [31, 32, 33]

    assert list(c.cluster_ids) == [2, 10, 11, 20, 30, 31, 32, 33]

    # Undo the split.
    c.undo()
    assert list(c.cluster_ids) == [0, 1, 2, 10, 11, 20, 30]

    # Redo the split.
    c.redo()
    assert list(c.cluster_ids) == [2, 10, 11, 20, 30, 31, 32, 33]


def test_controller_label_1(controller):
    c = controller

    c.label(cluster_ids=20, name="my_field", value=3.14)
    c.label(cluster_ids=30, name="my_field", value=1.23)

    assert 'my_field' in c.fields
    assert c.get_labels('my_field')[20] == 3.14
    assert c.get_labels('my_field')[30] == 1.23


def test_controller_label_merge_1(controller):
    c = controller

    c.label(cluster_ids=[20, 30], name="my_field", value=3.14)

    # Same value for the old clusters.
    l = c.get_labels('my_field')
    assert l[20] == l[30] == 3.14

    up = c.merge([20, 30])

    assert c.get_labels('my_field')[up.added[0]] == 3.14


def test_controller_label_merge_2(controller):
    c = controller

    c.label(cluster_ids=[20], name="my_field", value=3.14)

    # One of the parents.
    l = c.get_labels('my_field')
    assert l[20] == 3.14
    assert l[30] is None

    up = c.merge([20, 30])

    assert c.get_labels('my_field')[up.added[0]] == 3.14


def test_controller_label_merge_3(controller):
    c = controller

    # Conflict: largest cluster wins.
    c.label(cluster_ids=[20, 30], name="my_field", value=3.14)

    # Create merged cluster from 20 and 30.
    up = c.merge(cluster_ids=[20, 30])
    new = up.added[0]

    # It fot the label of its parents.
    assert c.get_labels('my_field')[new] == 3.14

    # Now, we label a smaller cluster.
    c.label(cluster_ids=[10], name="my_field", value=2.718)

    # We merge the large and small cluster together.
    up = c.merge(cluster_ids=up.added + [10])

    # The new cluster should have the value of the first, merged big cluster, i.e. 3.14.
    assert c.get_labels('my_field')[up.added[0]] == 3.14
