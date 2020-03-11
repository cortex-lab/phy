# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac

from phylib.io.mock import artificial_traces, artificial_spike_clusters
from phylib.utils import Bunch, connect
from phylib.utils.geometry import linear_positions
from phy.plot.tests import mouse_click

from ..trace import TraceView, TraceImageView, select_traces, _iter_spike_waveforms
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test trace view
#------------------------------------------------------------------------------

def test_iter_spike_waveforms():
    nc = 5
    ns = 20
    sr = 2000.
    ch = list(range(nc))
    duration = 1.
    st = np.linspace(0.1, .9, ns)
    sc = artificial_spike_clusters(ns, nc)
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    m = Bunch(spike_times=st, spike_clusters=sc, sample_rate=sr)
    s = Bunch(cluster_meta={}, selected=[0])

    for w in _iter_spike_waveforms(
            interval=[0., 1.],
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
    sr = 2000.
    duration = 1.
    st = np.linspace(0.1, .9, ns)
    sc = artificial_spike_clusters(ns, nc)
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    def get_traces(interval):
        out = Bunch(
            data=select_traces(traces, interval, sample_rate=sr),
            color=(.75, .75, .75, 1),
        )
        a, b = st.searchsorted(interval)
        out.waveforms = []
        k = 20
        for i in range(a, b):
            t = st[i]
            c = sc[i]
            s = int(round(t * sr))
            d = Bunch(
                data=traces[s - k:s + k, :],
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
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.on_select(cluster_ids=[])
    v.on_select(cluster_ids=[0])
    v.on_select(cluster_ids=[0, 2, 3])
    v.on_select(cluster_ids=[0, 2])

    v.stacked.add_boxes(v.canvas)

    ac(v.stacked.box_size, (.950, .165), atol=1e-3)
    v.set_interval((.375, .625))
    assert v.time == .5
    qtbot.wait(1)

    v.go_to(.25)
    assert v.time == .25
    qtbot.wait(1)

    v.go_to(-.5)
    assert v.time == .125
    qtbot.wait(1)

    v.go_left()
    assert v.time == .125
    qtbot.wait(1)

    v.go_right()
    ac(v.time, .150)
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
    v.interval = (.25, .75)
    ac(v.interval, (.25, .75))
    qtbot.wait(1)

    v.widen()
    ac(v.interval, (.1875, .8125))
    qtbot.wait(1)

    v.narrow()
    ac(v.interval, (.25, .75))
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
    ac(v.stacked.box_size, bs, atol=.05)
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

    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left', modifiers=('Control',))

    v.set_state(v.state)

    assert len(_clicked[0]) == 3

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_select_channel(sender, channel_id=None, button=None):
        _clicked.append((channel_id, button))

    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left', modifiers=('Shift',))
    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Right', modifiers=('Shift',))

    assert _clicked == [(2, 'Left'), (2, 'Right')]

    _stop_and_close(qtbot, v)


#------------------------------------------------------------------------------
# Test trace imageview
#------------------------------------------------------------------------------

def test_trace_image_view_1(qtbot, tempdir, gui):
    nc = 350
    sr = 2000.
    duration = 1.
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)

    def get_traces(interval):
        return Bunch(data=select_traces(traces, interval, sample_rate=sr),
                     color=(.75, .75, .75, 1),
                     )

    v = TraceImageView(
        traces=get_traces,
        n_channels=nc,
        sample_rate=sr,
        duration=duration,
        channel_positions=linear_positions(nc),
    )
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.update_color()

    v.set_interval((.375, .625))
    assert v.time == .5
    qtbot.wait(1)

    v.go_to(.25)
    assert v.time == .25
    qtbot.wait(1)

    v.go_to(-.5)
    assert v.time == .125
    qtbot.wait(1)

    v.go_left()
    assert v.time == .125
    qtbot.wait(1)

    v.go_right()
    ac(v.time, .150)
    qtbot.wait(1)

    v.jump_left()
    qtbot.wait(1)

    v.jump_right()
    qtbot.wait(1)

    # Change interval size.
    v.interval = (.25, .75)
    ac(v.interval, (.25, .75))
    qtbot.wait(1)

    v.widen()
    ac(v.interval, (.1875, .8125))
    qtbot.wait(1)

    v.narrow()
    ac(v.interval, (.25, .75))
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
