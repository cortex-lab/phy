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
