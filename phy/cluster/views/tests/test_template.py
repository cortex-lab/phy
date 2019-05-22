# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.mock import artificial_waveforms
from phylib.utils import Bunch, connect
from phylib.utils._color import ClusterColorSelector

from ..template import TemplateView
from phy.plot.tests import mouse_click, key_press, key_release


#------------------------------------------------------------------------------
# Test template view
#------------------------------------------------------------------------------

def test_template_view(qtbot, tempdir, gui):
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

    v = TemplateView(
        templates=get_templates, channel_ids=channel_ids, cluster_ids=cluster_ids,
        cluster_color_selector=cluster_color_selector)
    v.show()
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    v.plot()

    v.on_select([])
    v.on_select([0])

    # Simulate channel selection.
    _clicked = []

    @connect(sender=v)
    def on_cluster_click(sender, cluster_id=None, button=None, key=None):
        _clicked.append((cluster_id, button, key))

    key_press(qtbot, v.canvas, '2')
    mouse_click(qtbot, v.canvas, pos=(0., 0.), button='Left')
    key_release(qtbot, v.canvas, '2')

    assert _clicked == [(4, 'Left', 2)]

    # qtbot.stop()
    v.close()
