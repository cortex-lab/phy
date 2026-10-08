"""Test Histogram view."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from phylib.utils import Bunch

from ..histogram import FiringRateView, HistogramView, ISIView, _compute_histogram
from . import _stop_and_close

# ------------------------------------------------------------------------------
# Test Histogram view
# ------------------------------------------------------------------------------


def test_compute_histogram_n_bins():
    histogram = _compute_histogram(
        np.array([0.5, 1.5, 2.5, 3.5]), x_min=0, x_max=4, n_bins=4, normalize=False
    )
    np.testing.assert_array_equal(histogram, np.ones(4))


def test_histogram_view_0(qtbot, gui):
    data = np.random.uniform(low=0, high=10, size=5000)
    # plot = .1 * np.random.uniform(low=0, high=.5, size=1000)
    v = HistogramView(
        cluster_stat=lambda cluster_id: Bunch(
            data=data,
            # plot=plot,
            text=f'this is:\ncluster {cluster_id}',
        )
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)
    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[2, 3, 5])

    v.set_n_bins(200)
    assert v.n_bins == 200

    v.set_x_min(-2)
    assert v.x_min >= -2
    v.set_x_min(10)  # should fail
    assert v.x_min >= -2

    v.set_x_max(5)
    assert v.x_max <= 5
    v.set_x_max(-10)  # should fail
    assert v.x_max <= 5

    bs = v.bin_size
    assert bs > 0
    v.set_bin_size(bs)
    assert v.bin_size == bs

    v.increase()
    v.decrease()

    # Use ms unit.
    v.bin_unit = 'ms'
    v.set_x_min(100)
    assert v.x_min == 0.1
    v.set_x_max(500)
    assert v.x_max == 0.5
    v.set_n_bins(400)
    assert v.bin_size == 1  # 1 ms
    v.set_bin_size(2)
    assert v.n_bins == 200

    v.increase()
    v.decrease()

    _stop_and_close(qtbot, v)


def test_firing_rate_view_ignores_global_x_max(qtbot, gui):
    gui.state.FiringRateView = Bunch(n_bins=200, x_max=12.0)
    v = FiringRateView(
        cluster_stat=lambda cluster_id: Bunch(
            data=np.array([1.0, 20.0]),
            x_min=0.0,
            x_max=30.0,
        )
    )

    v.attach(gui)
    assert v.x_max is None
    v.on_select(cluster_ids=[0])
    assert v.x_max == 30.0

    _stop_and_close(qtbot, v)


def test_firing_rate_view_displays_spikes_per_second(qtbot):
    v = FiringRateView(
        cluster_stat=lambda cluster_id: Bunch(
            data=np.arange(0.125, 2, 0.25),
            x_min=0.0,
            x_max=2.0,
        )
    )
    v.n_bins = 2
    v.cluster_ids = [0]

    bunch = v.get_clusters_data()[0]

    np.testing.assert_array_equal(bunch.histogram, np.array([4.0, 4.0]))
    _stop_and_close(qtbot, v)


def test_firing_rate_view_formats_recording_time_axis(qtbot):
    v = FiringRateView(
        cluster_stat=lambda cluster_id: Bunch(
            data=np.array([1.0, 3600.0]),
            x_min=0.0,
            x_max=7200.0,
        )
    )
    v.on_select(cluster_ids=[0])
    v._set_recording_time_format('h', 2)

    assert v.recording_time_unit == 'h'
    assert all(label.endswith(' h') for label in v.canvas.axes.locator.xtext)

    _stop_and_close(qtbot, v)


def test_histogram_view_settings(qtbot, gui, monkeypatch):
    v = ISIView(
        cluster_stat=lambda cluster_id: Bunch(
            data=np.array([0.001, 0.010]),
            x_min=0.0,
            x_max=0.050,
        )
    )
    v.attach(gui)
    v.on_select(cluster_ids=[0])
    monkeypatch.setattr(
        'phy.cluster.views.histogram.view_settings_dialog',
        lambda *args, **kwargs: {'bin_size': 2.0, 'x_min': 1.0, 'x_max': 41.0},
    )

    v.actions.get('View settings').trigger()

    assert v.x_min == 0.001
    assert v.x_max == 0.041
    assert v.n_bins == 20
    assert set(v.local_state_attrs) == {'n_bins', 'x_min', 'x_max'}
    _stop_and_close(qtbot, v)
