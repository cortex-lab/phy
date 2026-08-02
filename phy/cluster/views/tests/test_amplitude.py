"""Test amplitude view."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from phylib.io.mock import artificial_spike_samples
from phylib.utils import Bunch, connect

from phy.plot import NDC
from phy.plot.tests import mouse_click, mouse_drag
from phy.plot.transform import Range

from ..amplitude import AmplitudeView
from . import _stop_and_close

# ------------------------------------------------------------------------------
# Test amplitude view
# ------------------------------------------------------------------------------


def test_amplitude_view_0(qtbot, gui):
    v = AmplitudeView(
        amplitudes=lambda cluster_ids, load_all=False: None,
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    v.increase_marker_size()
    v.decrease_marker_size()
    v.reset_marker_size()

    _stop_and_close(qtbot, v)


def test_amplitude_view_1(qtbot, gui):
    x = np.zeros(1)
    v = AmplitudeView(
        amplitudes=lambda cluster_ids, load_all=False: [
            Bunch(amplitudes=x, spike_ids=[0], spike_times=[0])
        ],
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    v.show_time_range((0.499, 0.501))

    _stop_and_close(qtbot, v)


def test_amplitude_view_2(qtbot, gui):
    random_state = np.random.get_state()
    np.random.seed(0)
    rng = np.random.RandomState(0)

    n = 1000
    try:
        st1 = artificial_spike_samples(n) / 20000.0
        st2 = artificial_spike_samples(n) / 20000.0
        v = AmplitudeView(
            amplitudes={
                'amp1': lambda cluster_ids, load_all=False: [
                    Bunch(
                        amplitudes=15 + rng.randn(n),
                        spike_ids=np.arange(n),
                        spike_times=st1,
                    )
                    for c in cluster_ids
                ],
                'amp2': lambda cluster_ids, load_all=False: [
                    Bunch(
                        amplitudes=10 + rng.randn(n),
                        spike_ids=np.arange(n),
                        spike_times=st2,
                    )
                    for c in cluster_ids
                ],
            },
            duration=max(st1.max(), st2.max()),
        )
        with qtbot.waitExposed(v.canvas):
            v.show()
        v.attach(gui)

        v.on_select(cluster_ids=[])
        v.on_select(cluster_ids=[0])
        v.on_select(cluster_ids=[0, 2, 3])
        v.on_select(cluster_ids=[0, 2])

        v.next_amplitudes_type()
        v.previous_amplitudes_type()
        v.actions.change_amplitudes_type_to_amp2()

        v.set_state(v.state)

        w, h = v.canvas.get_size()

        _times = []

        @connect(sender=v)
        def on_select_time(sender, time):
            _times.append(time)

        mouse_click(qtbot, v.canvas, (w / 3, h / 2), modifiers=('Alt',))
        assert len(_times) == 1
        assert np.allclose(_times[0], 0.5, atol=0.01)

        # Split without selection.
        spike_ids = v.on_request_split()
        assert len(spike_ids) == 0

        a, b = 50, 1000
        mouse_click(qtbot, v.canvas, (a, a), modifiers=('Control',))
        mouse_click(qtbot, v.canvas, (a, b), modifiers=('Control',))
        mouse_click(qtbot, v.canvas, (b, b), modifiers=('Control',))
        mouse_click(qtbot, v.canvas, (b, a), modifiers=('Control',))

        # Split lassoed points.
        spike_ids = v.on_request_split()
        assert len(spike_ids) > 0

        _stop_and_close(qtbot, v)
    finally:
        np.random.set_state(random_state)


def test_amplitude_threshold_gesture_and_preview(qtbot, gui):
    background_color = (0.5, 0.5, 0.5, 0.5)

    def amplitudes(cluster_ids, load_all=False):
        out = []
        for cluster_id in cluster_ids:
            if cluster_id is None:
                out.append(Bunch(amplitudes=[0.5], spike_ids=[90], spike_times=[0.25]))
            else:
                out.append(
                    Bunch(amplitudes=[1.0, 3.0], spike_ids=[10, 11], spike_times=[0.4, 0.6])
                )
        return out

    v = AmplitudeView(amplitudes=amplitudes, duration=1.0)
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)
    v.on_select(cluster_ids=[0])

    selected_times = []

    @connect(sender=v)
    def on_select_time(sender, time):
        selected_times.append(time)

    w, h = v.canvas.get_size()
    v.canvas.panzoom.zoom = (1.2, 1.5)
    v.canvas.panzoom.pan = (0.1, -0.2)
    target = np.array([[0.5, 1.5]])
    ndc = Range(v.data_bounds, NDC).apply(target)[0]
    screen_ndc = (ndc + np.asarray(v.canvas.panzoom.pan)) * v.canvas.panzoom._zoom_aspect()
    pixel = ((screen_ndc[0] + 1) * w / 2, (1 - screen_ndc[1]) * h / 2)
    assert np.isclose(v._threshold_from_window_pos(pixel), target[0, 1])

    mouse_drag(
        qtbot,
        v.canvas,
        (w / 2, h * 0.75),
        (w / 2, h * 0.5),
        button='right',
        modifiers=('Alt',),
    )
    assert v.split_threshold is not None
    assert selected_times == []
    assert v.split_threshold_visual._hidden is False

    v.split_threshold = 2.0
    v._replot_displayed_amplitudes()
    colors = v.visual._acc.color
    assert np.allclose(colors[0], background_color)
    assert np.any(np.all(np.isclose(colors[1:], v.split_preview_color), axis=1))

    mouse_click(qtbot, v.canvas, (w / 2, h / 2), button='left', modifiers=('Alt',))
    assert len(selected_times) == 1

    mouse_click(qtbot, v.canvas, (w / 3, h / 3), button='left', modifiers=('Control',))
    assert v.canvas.lasso.count == 1
    assert v.split_threshold is None
    v.split_threshold = 2.0
    v._replot_displayed_amplitudes()
    mouse_click(qtbot, v.canvas, (w / 2, h / 2), button='right', modifiers=('Control',))
    assert v.split_threshold is None
    assert v.canvas.lasso.count == 0

    _stop_and_close(qtbot, v)


def test_amplitude_threshold_exact_split_is_strict_and_finite(qtbot):
    calls = []

    def amplitudes(cluster_ids, load_all=False):
        calls.append(load_all)
        if load_all:
            return [
                Bunch(
                    amplitudes=np.array([1.0, 2.0, np.nan, 3.0]),
                    spike_ids=np.array([10, 11, 12, 13]),
                    spike_times=np.arange(4.0),
                )
            ]
        return [
            Bunch(amplitudes=[0.0], spike_ids=[99], spike_times=[0.0]),
            Bunch(amplitudes=[1.0, 3.0], spike_ids=[10, 13], spike_times=[0.0, 3.0]),
        ]

    v = AmplitudeView(amplitudes=amplitudes, duration=4.0)
    v.on_select(cluster_ids=[0])
    v.split_threshold = 2.0
    spike_ids = v.on_request_split()

    assert calls.count(True) == 1
    assert spike_ids.dtype == np.int64
    assert np.array_equal(spike_ids, [10])
    assert v.split_threshold is None
    v.close()


def test_amplitude_threshold_rejects_empty_whole_and_invalid_activation(qtbot):
    exact_amplitudes = np.array([1.0, 2.0, 3.0])

    def amplitudes(cluster_ids, load_all=False):
        if load_all:
            return [
                Bunch(
                    amplitudes=exact_amplitudes,
                    spike_ids=np.array([0, 1, 2]),
                    spike_times=np.arange(3.0),
                )
            ]
        return [Bunch(amplitudes=[0.0], spike_ids=[9], spike_times=[0.0]) for _ in cluster_ids]

    v = AmplitudeView(amplitudes=amplitudes, split_is_eligible=lambda: False)
    v.on_select(cluster_ids=[0])
    assert not v._can_set_split_threshold()

    v.split_threshold = 0.0
    assert v.on_request_split().size == 0
    assert v.split_threshold == 0.0
    v.split_threshold = 4.0
    assert v.on_request_split().size == 0
    assert v.split_threshold == 4.0

    v.on_select(cluster_ids=[0, 1])
    assert v.split_threshold is None
    assert not v._can_set_split_threshold()
    v.close()
