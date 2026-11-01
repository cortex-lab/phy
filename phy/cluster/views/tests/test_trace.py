"""Test views."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
import pytest
from numpy.testing import assert_allclose as ac
from phylib.io.mock import artificial_spike_clusters, artificial_traces
from phylib.utils import Bunch, connect
from phylib.utils.geometry import linear_positions

from phy.plot.tests import mouse_click

from ..correlogram import CorrelogramView
from ..trace import (
    TraceImageView,
    TraceView,
    _iter_spike_waveforms,
    _next_short_isi,
    select_traces,
)
from . import _stop_and_close

# ------------------------------------------------------------------------------
# Test trace view
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'spike_times,time,period,expected',
    [
        (None, 0, 0.002, None),
        ([], 0, 0.002, None),
        ([1.0], 0, 0.002, None),
        ([0.0, 0.5, 1.0], 0, 0.002, None),
        ([0.25, 0.375, 0.75], 0, 0.125, None),
        ([0.25, 0.375, 0.75], 0, 0.25, 0.25),
        ([0.25, 0.26, 0.75, 0.76], 0.25, 0.02, 0.75),
        ([0.25, 0.26, 0.75, 0.76], 0.8, 0.02, 0.25),
        ([0.25, 0.25, 0.75], 0, 0.002, 0.25),
        ([0.25, 0.26], 0, 0, None),
    ],
)
def test_next_short_isi(spike_times, time, period, expected):
    times = None if spike_times is None else np.asarray(spike_times)
    assert _next_short_isi(times, time, period, block_size=2) == expected


def test_next_short_isi_block_boundary():
    times = np.array([0, 1, 2, 2.001, 4, 5])
    assert _next_short_isi(times, 0, 0.002, block_size=2) == 2
    assert _next_short_isi(times, 4, 0.002, block_size=2) == 2


def test_trace_short_isi_navigation(qtbot, gui):
    selection = {'times': np.array([0.3, 0.301, 0.6, 0.604])}
    view = TraceView(
        traces=lambda interval: Bunch(data=np.zeros((100, 2)), waveforms=[]),
        spike_times=lambda: selection['times'],
        n_channels=2,
        sample_rate=1000,
        duration=1,
    )
    correlogram = CorrelogramView(
        correlograms=lambda *args, **kwargs: np.zeros((1, 1, 51)),
        firing_rate=lambda *args, **kwargs: np.zeros((1, 1)),
        sample_rate=1000,
    )
    try:
        view.attach(gui)
        view.go_to(0.2)
        view.go_to_next_short_isi()
        ac(view.time, 0.2)  # No correlogram: do not guess a threshold.

        correlogram.attach(gui)
        correlogram.set_refractory_period(2)  # Milliseconds, stored in seconds.
        view.actions.get('go_to_next_short_isi').trigger()
        ac(view.time, 0.3)
        view.go_to_next_short_isi()
        ac(view.time, 0.3)  # Only one qualifying pair: wrap to it.

        correlogram.set_refractory_period(5)
        view.go_to_next_short_isi()
        ac(view.time, 0.6)  # Read the current threshold on every action.

        selection['times'] = np.array([0.4, 0.401])
        view.on_select(cluster_ids=[1])
        view.go_to_next_short_isi()
        ac(view.time, 0.4)  # A new selection must not reuse old candidates.

        for times in (None, np.array([]), np.array([0.5]), np.array([0.2, 0.8])):
            selection['times'] = times
            view.go_to_next_short_isi()
            ac(view.time, 0.4)

        selection['times'] = np.array([0.01, 0.011, 0.99, 0.991])
        view.on_select(cluster_ids=[2])
        view.go_to(0.5)
        view.go_to_next_short_isi()
        ac(view.time, 0.875)  # The target is near the end; the viewport is clipped.
        view.go_to_next_short_isi()
        ac(view.time, 0.125)  # Wrap rather than getting stuck at the last pair.
        view.go_to_next_short_isi()
        ac(view.time, 0.875)
    finally:
        _stop_and_close(qtbot, correlogram)
        _stop_and_close(qtbot, view)


def test_iter_spike_waveforms():
    nc = 5
    ns = 20
    sr = 2000.0
    ch = list(range(nc))
    duration = 1.0
    st = np.linspace(0.1, 0.9, ns)
    sc = artificial_spike_clusters(ns, nc)
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    m = Bunch(spike_times=st, spike_clusters=sc, sample_rate=sr)
    s = Bunch(cluster_meta={}, selected=[0])

    for w in _iter_spike_waveforms(
        interval=[0.0, 1.0],
        traces_interval=traces,
        model=m,
        supervisor=s,
        n_samples_waveforms=ns,
        show_all_spikes=True,
        get_best_channels=lambda cluster_id: (ch, np.ones(nc)),
    ):
        assert w


def test_trace_view_1(qtbot, tempdir, gui):
    nc = 5
    ns = 20
    sr = 2000.0
    duration = 1.0
    st = np.linspace(0.1, 0.9, ns)
    sc = artificial_spike_clusters(ns, nc)
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    def get_traces(interval):
        out = Bunch(
            data=select_traces(traces, interval, sample_rate=sr),
            color=(0.75, 0.75, 0.75, 1),
        )
        a, b = st.searchsorted(interval)
        out.waveforms = []
        k = 20
        for i in range(a, b):
            t = st[i]
            c = sc[i]
            s = int(round(t * sr))
            d = Bunch(
                data=traces[s - k : s + k, :],
                start_time=(s - k) / sr,
                channel_ids=np.arange(5),
                spike_id=i,
                spike_cluster=c,
                select_index=0,
            )
            out.waveforms.append(d)
        return out

    def get_spike_times():
        return st

    v = TraceView(
        traces=get_traces,
        spike_times=get_spike_times,
        n_channels=nc,
        sample_rate=sr,
        duration=duration,
        channel_positions=linear_positions(nc),
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.stacked.add_boxes(v.canvas)

    ac(v.stacked.box_size, (0.950, 0.165), atol=1e-3)
    v.set_interval((0.375, 0.625))
    assert v.time == 0.5
    qtbot.wait(1)

    v.go_to(0.25)
    assert v.time == 0.25
    qtbot.wait(1)

    v.go_to(-0.5)
    assert v.time == 0.125
    qtbot.wait(1)

    v.go_left()
    assert v.time == 0.125
    qtbot.wait(1)

    v.go_right()
    ac(v.time, 0.150)
    qtbot.wait(1)

    v.jump_left()
    qtbot.wait(1)

    v.jump_right()
    qtbot.wait(1)

    v.go_to_next_spike()
    qtbot.wait(1)

    v.go_to_previous_spike()
    qtbot.wait(1)

    # Change interval size.
    v.interval = (0.25, 0.75)
    ac(v.interval, (0.25, 0.75))
    qtbot.wait(1)

    v.widen()
    ac(v.interval, (0.1875, 0.8125))
    qtbot.wait(1)

    v.narrow()
    ac(v.interval, (0.25, 0.75))
    qtbot.wait(1)

    v.go_to_start()
    qtbot.wait(1)
    assert v.interval[0] == 0

    v.go_to_end()
    qtbot.wait(1)
    assert v.interval[1] == duration

    # Widen the max interval.
    v.set_interval((0, duration))
    v.widen()
    qtbot.wait(1)

    v.toggle_show_labels(True)
    v.go_right()

    # Check auto scaling.
    db = v.data_bounds
    v.toggle_auto_scale(False)
    v.narrow()
    qtbot.wait(1)
    # Check that ymin and ymax have not changed.
    assert v.data_bounds[1] == db[1]
    assert v.data_bounds[3] == db[3]

    v.toggle_auto_update(True)
    assert v.do_show_labels
    qtbot.wait(1)

    v.toggle_highlighted_spikes(True)
    qtbot.wait(50)

    # Change channel scaling.
    bs = v.stacked.box_size
    v.decrease()
    qtbot.wait(1)

    v.increase()
    ac(v.stacked.box_size, bs, atol=0.05)
    qtbot.wait(1)

    v.origin = 'bottom'
    v.switch_origin()
    assert v.origin == 'top'
    qtbot.wait(1)

    # Simulate spike selection.
    _clicked = []

    @connect(sender=v)
    def on_select_spike(sender, channel_id=None, spike_id=None, cluster_id=None, key=None):
        _clicked.append((channel_id, spike_id, cluster_id))

    mouse_click(qtbot, v.canvas, pos=(0.0, 0.0), button='Left', modifiers=('Control',))

    v.set_state(v.state)

    assert len(_clicked[0]) == 3

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_select_channel(sender, channel_id=None, button=None):
        _clicked.append((channel_id, button))

    mouse_click(qtbot, v.canvas, pos=(0.0, 0.0), button='Left', modifiers=('Shift',))
    mouse_click(qtbot, v.canvas, pos=(0.0, 0.0), button='Right', modifiers=('Shift',))

    assert _clicked == [(2, 'Left'), (2, 'Right')]

    _stop_and_close(qtbot, v)


# ------------------------------------------------------------------------------
# Test trace imageview
# ------------------------------------------------------------------------------


def test_trace_image_view_1(qtbot, tempdir, gui):
    nc = 350
    sr = 2000.0
    duration = 1.0
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    def get_traces(interval):
        return Bunch(
            data=select_traces(traces, interval, sample_rate=sr),
            color=(0.75, 0.75, 0.75, 1),
        )

    v = TraceImageView(
        traces=get_traces,
        n_channels=nc,
        sample_rate=sr,
        duration=duration,
        channel_positions=linear_positions(nc),
    )
    with qtbot.waitExposed(v.canvas):
        v.show()
    v.attach(gui)

    v.update_color()

    v.set_interval((0.375, 0.625))
    assert v.time == 0.5
    qtbot.wait(1)

    v.go_to(0.25)
    assert v.time == 0.25
    qtbot.wait(1)

    v.go_to(-0.5)
    assert v.time == 0.125
    qtbot.wait(1)

    v.go_left()
    assert v.time == 0.125
    qtbot.wait(1)

    v.go_right()
    ac(v.time, 0.150)
    qtbot.wait(1)

    v.jump_left()
    qtbot.wait(1)

    v.jump_right()
    qtbot.wait(1)

    # Change interval size.
    v.interval = (0.25, 0.75)
    ac(v.interval, (0.25, 0.75))
    qtbot.wait(1)

    v.widen()
    ac(v.interval, (0.1875, 0.8125))
    qtbot.wait(1)

    v.narrow()
    ac(v.interval, (0.25, 0.75))
    qtbot.wait(1)

    v.go_to_start()
    qtbot.wait(1)
    assert v.interval[0] == 0

    v.go_to_end()
    qtbot.wait(1)
    assert v.interval[1] == duration

    # Widen the max interval.
    v.set_interval((0, duration))
    v.widen()
    qtbot.wait(1)

    v.toggle_auto_update(True)
    assert v.do_show_labels
    qtbot.wait(1)

    # Change channel scaling.
    v.decrease()
    qtbot.wait(1)

    v.increase()
    qtbot.wait(1)

    v.origin = 'bottom'
    v.switch_origin()
    # assert v.origin == 'top'
    qtbot.wait(1)

    _stop_and_close(qtbot, v)
