# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import shutil

from pytest import fixture

from phylib.io.datasets import download_test_file
from phylib.utils import connect, reset
from phylib.utils.testing import captured_output
from phy.cluster.views import WaveformView, TraceView
from phy.gui.widgets import Barrier
from phy.plot.tests import key_press, mouse_click
from ..gui import KwikController, kwik_describe

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def controller(tempdir):
    # Download the dataset.
    paths = list(map(download_test_file, ('kwik/hybrid_10sec.kwik',
                                          'kwik/hybrid_10sec.kwx',
                                          'kwik/hybrid_10sec.dat')))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path, tempdir / path.name)
    kwik_path = tempdir / paths[0].name
    c = KwikController(kwik_path, channel_group=0)
    yield c
    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()


def _wait_controller(controller):
    mc = controller.supervisor
    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=mc.cluster_view)
    connect(b('similarity_view'), event='ready', sender=mc.similarity_view)
    b.wait()


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_describe(controller):
    with captured_output() as (stdout, stderr):
        kwik_describe(controller.model.kwik_path)
    assert 'main*' in stdout.getvalue()


def test_gui_1(qtbot, tempdir, controller):
    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)
    _wait_controller(controller)

    wv = gui.list_views(WaveformView)[0]
    tv = gui.list_views(TraceView)[0]

    tv.actions.go_to_next_spike()
    s.block()

    s.actions.next()
    s.block()

    clu_moved = s.selected[0]
    s.actions.move_best_to_good()
    s.block()
    assert len(s.selected) == 1

    s.actions.next()
    s.block()

    clu_to_merge = s.selected
    assert len(clu_to_merge) == 2

    # Ensure the template feature view is updated.
    s.actions.merge()
    s.block()
    clu_merged = s.selected[0]

    s.actions.move_all_to_mua()
    s.block()

    s.actions.next()
    s.block()

    clu = s.selected[0]

    wv.actions.toggle_mean_waveforms(True)
    tv.actions.toggle_highlighted_spikes(True)
    tv.actions.go_to_next_spike()
    tv.actions.go_to_previous_spike()

    mouse_click(qtbot, tv.canvas, (100, 100), modifiers=('Control',))

    s.save()
    gui.close()

    # Create a new controller and a new GUI with the same data.
    controller = KwikController(controller.model.kwik_path, config_dir=tempdir)

    gui = controller.create_gui()
    s = controller.supervisor
    gui.show()
    qtbot.waitForWindowShown(gui)

    assert s.cluster_meta.get('group', clu_moved) == 'good'
    for clu in clu_to_merge:
        assert clu not in s.clustering.cluster_ids
    assert clu_merged in s.clustering.cluster_ids
    gui.close()


def test_kwik_gui_2(qtbot, controller):
    gui = controller.create_gui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitForWindowShown(gui)

    key_press(qtbot, gui, 'Down')
    key_press(qtbot, gui, 'Down')
    key_press(qtbot, gui, 'Space')
    key_press(qtbot, gui, 'G')
    key_press(qtbot, gui, 'Space')
    key_press(qtbot, gui, 'G', modifiers=('Alt',))
    key_press(qtbot, gui, 'Z')
    key_press(qtbot, gui, 'N', modifiers=('Alt',))
    key_press(qtbot, gui, 'Space')
    # Recluster.
    key_press(qtbot, gui, 'Colon')
    for char in 'RECLUSTER':
        key_press(qtbot, gui, char)
    key_press(qtbot, gui, 'Enter')
    controller.supervisor.block()

    key_press(qtbot, gui, 'S', modifiers=('Control',))

    gui.close()
