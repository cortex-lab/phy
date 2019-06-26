# -*- coding: utf-8 -*-

"""Testing the Base controller."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import cycle, islice
import logging
import os

import numpy as np
from pytest import fixture

from phylib.utils import connect, Bunch, reset
from phylib.io.mock import (
    artificial_features, artificial_traces, artificial_spike_clusters, artificial_spike_samples,
    artificial_waveforms
)
from phy.cluster.views import WaveformView
from phy.gui.qt import Debouncer
from phy.gui.widgets import Barrier
from ..base import BaseController, WaveformMixin, FeatureMixin, TraceMixin, TemplateMixin

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

class MyModel(object):
    n_channels = 8
    n_spikes = 20000
    n_clusters = 32
    n_templates = n_clusters
    n_pcs = 5
    n_samples_waveforms = 100
    channel_vertical_order = None
    channel_positions = np.random.normal(size=(n_channels, 2))
    channel_shanks = np.zeros(n_channels, dtype=np.int32)
    features = artificial_features(n_spikes, n_channels, n_pcs)
    metadata = {'group': {3: 'noise', 4: 'mua', 5: 'good'}}
    sample_rate = 10000
    spike_attributes = {}
    amplitudes = np.random.normal(size=n_spikes, loc=1, scale=.1)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_templates = spike_clusters
    spike_times = artificial_spike_samples(n_spikes) / sample_rate
    duration = spike_times[-1]
    traces = artificial_traces(int(sample_rate * duration), n_channels)

    def _get_some_channels(self, offset, size):
        return list(islice(cycle(range(self.n_channels)), offset, offset + size))

    def get_features(self, spike_ids, channel_ids):
        return artificial_features(len(spike_ids), len(channel_ids), self.n_pcs)

    def get_waveforms(self, spike_ids, channel_ids):
        return artificial_waveforms(len(spike_ids), self.n_samples_waveforms, len(channel_ids))

    def get_template(self, template_id):
        nc = self.n_channels // 2
        return Bunch(
            template=artificial_waveforms(1, self.n_samples_waveforms, nc)[0, ...],
            channel_ids=self._get_some_channels(template_id, nc))

    def save_spike_clusters(self, spike_clusters):
        pass

    def save_metadata(self, name, values):
        pass


class MyController(BaseController):
    """Default controller."""
    def get_best_channels(self, cluster_id):
        return self.model._get_some_channels(cluster_id, 5)


class MyControllerW(WaveformMixin, MyController):
    """With waveform view."""
    pass


class MyControllerF(FeatureMixin, MyController):
    """With feature view."""
    pass


class MyControllerT(TraceMixin, MyController):
    """With trace view."""
    pass


class MyControllerTmp(TemplateMixin, MyController):
    """With templates."""
    pass


class MyControllerFull(TemplateMixin, WaveformMixin, FeatureMixin, TraceMixin, MyController):
    """With everything."""
    pass


def _controller(qtbot, tempdir, cls):
    Debouncer.delay = 1

    model = MyModel()
    controller = cls(
        dir_path=tempdir, config_dir=tempdir / 'config', model=model,
        clear_cache=True, enable_threading=False)
    gui = controller.create_gui(do_prompt_save=False)

    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=controller.supervisor.cluster_view)
    connect(b('similarity_view'), event='ready', sender=controller.supervisor.similarity_view)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitForWindowShown(gui)
    b.wait()

    yield controller, gui
    gui.close()

    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()


@fixture(params=[
    MyController,
    MyControllerW,
    MyControllerF,
    MyControllerT,
    MyControllerTmp,
    MyControllerFull,
])
def controller(request, qtbot, tempdir):
    yield from _controller(qtbot, tempdir, request.param)


@fixture
def controller_full(qtbot, tempdir):
    yield from _controller(qtbot, tempdir, MyControllerFull)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_base_gui_0(qtbot, tempdir, controller):
    controller, gui = controller
    s = controller.supervisor

    delay = 50

    # Select and view actions.
    for actions in (s.select_actions, s.view_actions):
        for name, action_obj in actions._actions_dict.items():
            if not action_obj.prompt:
                logger.info(name)
                action_obj.qaction.trigger()
                qtbot.wait(delay)

    # Merge action.
    s.select_actions.next()
    s.block()

    s.actions.merge()
    s.block()

    # Succession of undo and redo.
    for _ in range(3):
        s.actions.undo()
        s.block()

        s.actions.redo()
        s.block()

    qtbot.wait(delay)

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()


def test_base_gui_1(qtbot, tempdir, controller_full):
    controller, gui = controller_full

    for view_name in ('IPythonView', 'ProbeView', 'RasterView', 'TemplateView'):
        gui._create_and_add_view(view_name)
        qtbot.wait(50)

    # Close a view
    gui.get_view(WaveformView).close()
