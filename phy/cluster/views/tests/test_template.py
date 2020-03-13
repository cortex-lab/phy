# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_waveforms
from phylib.utils import Bunch, connect

from phy.plot.tests import mouse_click
from ..template import TemplateView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test template view
#------------------------------------------------------------------------------

def test_template_view_0(qtbot, tempdir, gui):
    n_samples = 50
    n_clusters = 10
    channel_ids = np.arange(n_clusters + 2)

    def get_templates(cluster_ids):
        return {i: Bunch(
            template=artificial_waveforms(1, n_samples, 2)[0, ...],
            channel_ids=np.arange(i, i + 2),
        ) for i in cluster_ids}

    v = TemplateView(templates=get_templates, channel_ids=channel_ids)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.plot()

    _stop_and_close(qtbot, v)


def test_template_view_1(qtbot, tempdir, gui):
    n_samples = 50
    n_clusters = 10
    channel_ids = np.arange(n_clusters + 2)
    cluster_ids = np.arange(n_clusters)

    def get_templates(cluster_ids):
        return {i: Bunch(
            template=artificial_waveforms(1, n_samples, 2)[0, ...],
            channel_ids=np.arange(i, i + 2),
        ) for i in cluster_ids}

    class Supervisor(object):
        pass
    s = Supervisor()

    v = TemplateView(templates=get_templates, channel_ids=channel_ids, cluster_ids=cluster_ids)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.update_color()  # should call .plot() instead as update_color() is for subsequent updates

    v.on_select(cluster_ids=[], sender=s)
    v.on_select(cluster_ids=[0], sender=s)
    assert v.status

    v.update_cluster_sort(cluster_ids[::-1])

    # Simulate cluster selection.
    _clicked = []

    w, h = v.canvas.get_size()

    @connect(sender=v)
    def on_request_select(sender, cluster_ids):
        _clicked.append(cluster_ids)

    @connect(sender=v)
    def on_select_more(sender, cluster_ids):
        _clicked.append(cluster_ids)

    mouse_click(qtbot, v.canvas, pos=(0, 0.), button='Left', modifiers=('Control',))
    assert len(_clicked) == 1
    assert _clicked[0] in ([4], [5])

    mouse_click(qtbot, v.canvas, pos=(0, h / 2), button='Left', modifiers=('Control', 'Shift',))
    assert len(_clicked) == 2
    assert _clicked[1] == [9]

    cluster_ids = np.arange(2, n_clusters + 2)
    v.set_cluster_ids(cluster_ids)
    v.plot()

    v.update_color()

    v.increase()
    v.decrease()
    v.scaling = v.scaling

    _stop_and_close(qtbot, v)
