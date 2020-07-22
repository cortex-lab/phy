# -*- coding: utf-8 -*-

"""Integration tests for the GUIs."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import cycle, islice
import logging
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from pytestqt.plugin import QtBot

from phylib.io.mock import (
    artificial_features, artificial_traces, artificial_spike_clusters, artificial_spike_samples,
    artificial_waveforms
)

from phylib.utils import connect, unconnect, Bunch, reset, emit

from phy.cluster.views import (
    WaveformView, FeatureView, AmplitudeView, TraceView, TemplateView,
)
from phy.gui.qt import Debouncer, create_app
from phy.gui.widgets import Barrier
from phy.plot.tests import mouse_click
from ..base import BaseController, WaveformMixin, FeatureMixin, TraceMixin, TemplateMixin

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Mock models and controller classes
#------------------------------------------------------------------------------

class MyModel(object):
    seed = np.random.seed(0)
    n_channels = 8
    n_spikes = 20000
    n_clusters = 32
    n_templates = n_clusters
    n_pcs = 5
    n_samples_waveforms = 100
    channel_positions = np.random.normal(size=(n_channels, 2))
    channel_mapping = np.arange(0, n_channels)
    channel_shanks = np.zeros(n_channels, dtype=np.int32)
    features = artificial_features(n_spikes, n_channels, n_pcs)
    metadata = {'group': {3: 'noise', 4: 'mua', 5: 'good'}}
    sample_rate = 10000
    spike_attributes = {}
    amplitudes = np.random.normal(size=n_spikes, loc=1, scale=.1)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_templates = spike_clusters
    spike_samples = artificial_spike_samples(n_spikes)
    spike_times = spike_samples / sample_rate
    spike_times_reordered = artificial_spike_samples(n_spikes) / sample_rate
    duration = spike_times[-1]
    spike_waveforms = None
    traces = artificial_traces(int(sample_rate * duration), n_channels)

    def _get_some_channels(self, offset, size):
        return list(islice(cycle(range(self.n_channels)), offset, offset + size))

    def get_features(self, spike_ids, channel_ids):
        return artificial_features(len(spike_ids), len(channel_ids), self.n_pcs)

    def get_waveforms(self, spike_ids, channel_ids):
        n_channels = len(channel_ids) if channel_ids else self.n_channels
        return artificial_waveforms(len(spike_ids), self.n_samples_waveforms, n_channels)

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

    def get_channel_amplitudes(self, cluster_id):
        return self.model._get_some_channels(cluster_id, 5), np.ones(5)


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


def _mock_controller(tempdir, cls):
    model = MyModel()
    return cls(
        dir_path=tempdir, config_dir=tempdir / 'config', model=model,
        clear_cache=True, enable_threading=False)


#------------------------------------------------------------------------------
# Base classes
#------------------------------------------------------------------------------

class MinimalControllerTests(object):

    # Methods to override
    #--------------------------------------------------------------------------

    @classmethod
    def get_controller(cls, tempdir):
        raise NotImplementedError()

    # Convenient properties
    #--------------------------------------------------------------------------

    @property
    def qtbot(self):
        return self.__class__._qtbot

    @property
    def controller(self):
        return self.__class__._controller

    @property
    def model(self):
        return self.__class__._controller.model

    @property
    def supervisor(self):
        return self.controller.supervisor

    @property
    def cluster_view(self):
        return self.supervisor.cluster_view

    @property
    def similarity_view(self):
        return self.supervisor.similarity_view

    @property
    def cluster_ids(self):
        return self.supervisor.clustering.cluster_ids

    @property
    def gui(self):
        return self.__class__._gui

    @property
    def selected(self):
        return self.supervisor.selected

    @property
    def amplitude_view(self):
        return self.gui.list_views(AmplitudeView)[0]

    # Convenience methods
    #--------------------------------------------------------------------------

    def stop(self):  # pragma: no cover
        """Used for debugging."""
        create_app().exec_()
        self.gui.close()

    def next(self):
        s = self.supervisor
        s.select_actions.next()
        s.block()

    def next_best(self):
        s = self.supervisor
        s.select_actions.next_best()
        s.block()

    def label(self, name, value):
        s = self.supervisor
        s.actions.label(name, value)
        s.block()

    def merge(self):
        s = self.supervisor
        s.actions.merge()
        s.block()

    def split(self):
        s = self.supervisor
        s.actions.split()
        s.block()

    def undo(self):
        s = self.supervisor
        s.actions.undo()
        s.block()

    def redo(self):
        s = self.supervisor
        s.actions.redo()
        s.block()

    def move(self, w):
        s = self.supervisor
        getattr(s.actions, 'move_%s' % w)()
        s.block()

    def lasso(self, view, scale=1.):
        w, h = view.canvas.get_size()
        w *= scale
        h *= scale
        mouse_click(self.qtbot, view.canvas, (1, 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (w - 1, 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (w - 1, h - 1), modifiers=('Control',))
        mouse_click(self.qtbot, view.canvas, (1, h - 1), modifiers=('Control',))

    # Fixtures
    #--------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        Debouncer.delay = 1
        cls._qtbot = QtBot(create_app())
        cls._tempdir_ = tempfile.mkdtemp()
        cls._tempdir = Path(cls._tempdir_)
        cls._controller = cls.get_controller(cls._tempdir)
        cls._create_gui()

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
            cls._qtbot.stop()
        cls._close_gui()
        shutil.rmtree(cls._tempdir_)

    @classmethod
    def _create_gui(cls):
        cls._gui = cls._controller.create_gui(do_prompt_save=False)
        s = cls._controller.supervisor
        b = Barrier()
        connect(b('cluster_view'), event='ready', sender=s.cluster_view)
        connect(b('similarity_view'), event='ready', sender=s.similarity_view)
        cls._gui.show()
        # cls._qtbot.addWidget(cls._gui)
        cls._qtbot.waitForWindowShown(cls._gui)
        b.wait()

    @classmethod
    def _close_gui(cls):
        cls._gui.close()
        cls._gui.deleteLater()

        # NOTE: make sure all callback functions are unconnected at the end of the tests
        # to avoid side-effects and spurious dependencies between tests.
        reset()


class BaseControllerTests(MinimalControllerTests):

    # Common test methods
    #--------------------------------------------------------------------------

    def test_common_01(self):
        """Select one cluster."""
        self.next_best()
        self.assertEqual(len(self.selected), 1)

    def test_common_02(self):
        """Select one similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_03(self):
        """Select another similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_04(self):
        """Merge the selected clusters."""
        self.merge()
        self.assertEqual(len(self.selected), 1)

    def test_common_05(self):
        """Select a similar cluster."""
        self.next()
        self.assertEqual(len(self.selected), 2)

    def test_common_06(self):
        """Undo/redo the merge several times."""
        for _ in range(3):
            self.undo()
            self.assertEqual(len(self.selected), 2)

            self.redo()
            self.assertEqual(len(self.selected), 2)

    def test_common_07(self):
        """Move action."""
        self.move('similar_to_noise')
        self.assertEqual(len(self.selected), 2)

    def test_common_08(self):
        """Move action."""
        self.move('best_to_good')
        self.assertEqual(len(self.selected), 1)

    def test_common_09(self):
        """Label action."""
        self.next()

        @connect(sender=self.supervisor)
        def on_cluster(sender, up):
            cls = self.__class__
            cls._label_name, cls._label_value = 'new_label', up.metadata_value

        self.label('new_label', 3)

        unconnect(on_cluster)

    def test_common_10(self):
        self.supervisor.save()

    def test_common_11(self):
        s = self.controller.selection
        self.assertEqual(s.cluster_ids, self.selected)
        self.gui.view_actions.toggle_spike_reorder(True)
        self.gui.view_actions.switch_raw_data_filter()


class GlobalViewsTests(object):
    def test_global_filter_1(self):
        self.next()
        cv = self.supervisor.cluster_view
        emit('table_filter', cv, self.cluster_ids[::2])

    def test_global_sort_1(self):
        cv = self.supervisor.cluster_view
        emit('table_sort', cv, self.cluster_ids[::-1])


#------------------------------------------------------------------------------
# Mock test cases
#------------------------------------------------------------------------------

class MockControllerTests(MinimalControllerTests, GlobalViewsTests, unittest.TestCase):
    """Empty mock controller."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyController)

    def test_create_ipython_view(self):
        self.gui.create_and_add_view('IPythonView')

    def test_create_raster_view(self):
        view = self.gui.create_and_add_view('RasterView')
        mouse_click(self.qtbot, view.canvas, (10, 10), modifiers=('Control',))
        view.actions.next_color_scheme()


class MockControllerWTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with waveforms."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerW)

    @property
    def waveform_view(self):
        return self.gui.list_views(WaveformView)[0]

    def test_waveform_view(self):
        self.waveform_view.actions.toggle_mean_waveforms(True)
        self.waveform_view.actions.next_waveforms_type()
        self.waveform_view.actions.change_n_spikes_waveforms(200)

    def test_mean_amplitudes(self):
        self.next()
        self.assertTrue(self.controller.get_mean_spike_raw_amplitudes(self.selected[0]) >= 0)

    def test_waveform_select_channel(self):
        self.amplitude_view.amplitudes_type = 'raw'

        fv = self.waveform_view
        # Select channel in waveform view.
        w, h = fv.canvas.get_size()
        w, h = w / 2, h / 2
        x, y = w / 2, h / 2
        mouse_click(self.qtbot, fv.canvas, (x, y), button='Left', modifiers=('Control',))


class MockControllerFTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with features."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerF)

    @property
    def feature_view(self):
        return self.gui.list_views(FeatureView)[0]

    def test_feature_view_split(self):
        self.next()
        n = max(self.cluster_ids)
        self.lasso(self.feature_view, .1)
        self.split()
        # Split one cluster => Two new clusters should be selected after the split.
        self.assertEqual(self.selected[:2], [n + 1, n + 2])

    def test_feature_view_toggle_spike_reorder(self):
        self.gui.view_actions.toggle_spike_reorder(True)

    def test_select_feature(self):
        self.next()

        fv = self.feature_view
        # Select feature in feature view.
        w, h = fv.canvas.get_size()
        w, h = w / 4, h / 4
        x, y = w / 2, h / 2
        mouse_click(self.qtbot, fv.canvas, (x, y), button='Right', modifiers=('Alt',))


class MockControllerTTests(GlobalViewsTests, MinimalControllerTests, unittest.TestCase):
    """Mock controller with traces."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerT)

    @property
    def trace_view(self):
        return self.gui.list_views(TraceView)[0]

    def test_trace_view(self):
        self.trace_view.actions.go_to_next_spike()
        self.trace_view.actions.go_to_previous_spike()
        self.trace_view.actions.toggle_highlighted_spikes(True)
        mouse_click(self.qtbot, self.trace_view.canvas, (100, 100), modifiers=('Control',))
        mouse_click(self.qtbot, self.trace_view.canvas, (150, 100), modifiers=('Shift',))
        emit('select_time', self, 0)
        self.trace_view.actions.next_color_scheme()


class MockControllerTmpTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with templates."""

    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerTmp)

    @property
    def template_view(self):
        return self.gui.list_views(TemplateView)[0]

    def test_template_view_select(self):
        mouse_click(self.qtbot, self.template_view.canvas, (100, 100), modifiers=('Control',))
        mouse_click(self.qtbot, self.template_view.canvas, (150, 100), modifiers=('Shift',))

    def test_mean_amplitudes(self):
        self.next()
        self.assertTrue(self.controller.get_mean_spike_template_amplitudes(self.selected[0]) >= 0)

    def test_split_template_amplitude(self):
        self.next()
        self.amplitude_view.amplitudes_type = 'template'
        self.controller.get_amplitudes(self.selected[0], load_all=True)
        self.amplitude_view.plot()
        self.lasso(self.amplitude_view)
        self.split()


class MockControllerFullTests(MinimalControllerTests, unittest.TestCase):
    """Mock controller with all views."""
    @classmethod
    def get_controller(cls, tempdir):
        return _mock_controller(tempdir, MyControllerFull)

    def test_filter(self):
        rdf = self.controller.raw_data_filter

        @rdf.add_filter
        def diff(arr, axis=0):  # pragma: no cover
            out = np.zeros_like(arr)
            if axis == 0:
                out[1:, ...] = np.diff(arr, axis=axis)
            elif axis == 1:
                out[:, 1:, ...] = np.diff(arr, axis=axis)
            return out

        self.gui.view_actions.switch_raw_data_filter()
        self.gui.view_actions.switch_raw_data_filter()

        rdf.set('diff')
        assert rdf.current == 'diff'

    def test_y1_close_view(self):
        s = self.selected
        self.next_best()
        assert s != self.selected
        fv = self.gui.get_view(FeatureView)
        wv = self.gui.get_view(WaveformView)
        assert self.selected == wv.cluster_ids
        fv.dock.close()
        s = self.selected
        self.next_best()
        assert s != self.selected
        assert self.selected == wv.cluster_ids

    def test_z1_close_all_views(self):
        self.next()

        for view in self.gui.views:
            view.dock.close()
            self.qtbot.wait(200)

    def test_z2_open_all_views(self):
        for view_cls in self.controller.view_creator.keys():
            self.gui.create_and_add_view(view_cls)
            self.qtbot.wait(200)

    def test_z3_select(self):
        self.next()
        self.next()

    def test_z4_open_new_views(self):
        for view_cls in self.controller.view_creator.keys():
            self.gui.create_and_add_view(view_cls)
            self.qtbot.wait(200)

    def test_z5_select(self):
        self.next_best()
        self.next()
