# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from phy.utils import Bunch
from phy.cluster.tests.conftest import MockController


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

@fixture
def state(tempdir):
    # Save a test GUI state JSON file in the tempdir.
    state = Bunch()
    state.WaveformView0 = Bunch(overlap=False)
    state.TraceView0 = Bunch(scaling=1.)
    state.FeatureView0 = Bunch(feature_scaling=.5)
    state.CorrelogramView0 = Bunch(uniform_normalization=True)
    return state


@fixture
def gui(tempdir, state):
    controller = MockController(config_dir=tempdir)
    return controller.create_gui(add_default_views=False, **state)


def _select_clusters(gui):
    gui.show()
    mc = gui.controller.picker
    assert mc
    mc.select([])
    mc.select([0])
    mc.select([0, 2])
    mc.select([0, 2, 3])
