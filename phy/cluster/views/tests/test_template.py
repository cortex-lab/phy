# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_waveforms
from phylib.utils import Bunch, connect
from phylib.utils.color import ClusterColorSelector

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

    cluster_color_selector = ClusterColorSelector(cluster_ids=cluster_ids)

    class Supervisor(object):
        pass
    s = Supervisor()

    v = TemplateView(
        templates=get_templates, channel_ids=channel_ids, cluster_ids=cluster_ids,
        cluster_color_selector=cluster_color_selector)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.update_color()  # should call .plot() instead as update_color() is for subsequent updates

    v.on_select(cluster_ids=[], sender=s)
    v.on_select(cluster_ids=[0], sender=s)

    v.update_cluster_sort(cluster_ids[::-1])

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_cluster_click(sender, cluster_id=None, button=None, modifiers=None):
        _clicked.append((cluster_id, button))

    mouse_click(qtbot, v.canvas, pos=(10., 10.), button='Left', modifiers=('Control',))

    assert _clicked == [(9, 'Left')]

    cluster_ids = np.arange(2, n_clusters + 2)
    v.set_cluster_ids(cluster_ids)
    v.plot()

    cluster_color_selector.set_color_mapping(colormap='linear')
    v.update_color()

    v.increase()
    v.decrease()
    v.scaling = v.scaling

    _stop_and_close(qtbot, v)
