# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac

from phy.gui import GUI
from phy.io.mock import (artificial_traces,
                         artificial_spike_clusters,
                         )
from phy.utils import Bunch
from phy.utils._color import ColorSelector

from ..trace import TraceView, select_traces


#------------------------------------------------------------------------------
# Test trace view
#------------------------------------------------------------------------------

def test_trace_view(tempdir, qtbot):
    nc = 5
    ns = 9
    sr = 1000.
    duration = 1.
    st = np.linspace(0.1, .9, ns)
    sc = artificial_spike_clusters(ns, nc)
    traces = 10 * artificial_traces(int(round(duration * sr)), nc)
    cs = ColorSelector()

    def get_traces(interval):
        out = Bunch(data=select_traces(traces, interval, sample_rate=sr),
                    color=(.75,) * 4,
                    )
        a, b = st.searchsorted(interval)
        out.waveforms = []
        k = 20
        for i in range(a, b):
            t = st[i]
            c = sc[i]
            s = int(round(t * sr))
            d = Bunch(data=traces[s - k:s + k, :],
                      start_time=t - k / sr,
                      color=cs.get(c),
                      channel_ids=np.arange(5),
                      cluster_id=c,
                      )
            out.waveforms.append(d)
        return out

    v = TraceView(traces=get_traces,
                  n_channels=nc,
                  sample_rate=sr,
                  duration=duration,
                  )
    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    # qtbot.waitForWindowShown(gui)

    v.on_select([])
    v.on_select([0])
    v.on_select([0, 2, 3])
    v.on_select([0, 2])

    # ac(v.stacked.box_size, (1., .08181), atol=1e-3)
    v.set_interval((.375, .625))
    assert v.time == .5

    v.go_to(.25)
    assert v.time == .25

    v.go_to(-.5)
    assert v.time == .125

    v.go_left()
    assert v.time == .125

    v.go_right()
    assert v.time == .175

    # Change interval size.
    v.interval = (.25, .75)
    ac(v.interval, (.25, .75))
    v.widen()
    ac(v.interval, (.125, .875))
    v.narrow()
    ac(v.interval, (.25, .75))

    # Widen the max interval.
    v.set_interval((0, duration))
    v.widen()

    v.toggle_show_labels()
    # v.toggle_show_labels()
    v.go_right()
    assert v.do_show_labels

    # Change channel scaling.
    bs = v.stacked.box_size
    v.increase()
    v.decrease()
    ac(v.stacked.box_size, bs, atol=1e-3)

    v.origin = 'upper'
    assert v.origin == 'upper'

    # qtbot.stop()
    gui.close()
