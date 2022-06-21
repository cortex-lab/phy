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

def test_action_controller_1(controller):
    c = controller
