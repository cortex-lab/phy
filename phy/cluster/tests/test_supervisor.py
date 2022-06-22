# -*- coding: utf-8 -*-

"""Test supervisor."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pprint import pprint

from pytest import fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from .. import supervisor as _supervisor
from ..supervisor import (
    Supervisor, ClusterView, SimilarityView, ActionCreator, TableController)
from phy.gui import GUI
from phy.gui.qt import qInstallMessageHandler
from phy.utils.context import Context
from phylib.utils import connect, Bunch, emit


def handler(msg_type, msg_log_context, msg_string):
    pass


qInstallMessageHandler(handler)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def gui(tempdir, qtbot):
    # NOTE: mock patch show box exec_
    _supervisor._show_box = lambda _: _

    gui = GUI(position=(200, 100), size=(800, 600), config_dir=tempdir)
    gui.set_default_actions()
    gui.show()
    qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(5)
    gui.close()
    del gui
    qtbot.wait(5)


@fixture
def controller(gui, cluster_ids, cluster_groups, cluster_labels, cluster_metrics, similarity):
    c = TableController(
        gui,
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        cluster_labels=cluster_labels,
        cluster_metrics=cluster_metrics,
        similarity=similarity,
    )
    return c


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_action_creator_1(qtbot, gui):
    ac = ActionCreator()
    ac.attach(gui)
    gui.show()
    # qtbot.stop()


def test_table_controller_1(qtbot, gui, controller):
    c = controller
    assert len(c.cluster_info()) == 7

    qtbot.stop()
