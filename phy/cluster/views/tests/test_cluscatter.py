# -*- coding: utf-8 -*-

"""Test views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.random import RandomState

from phylib.utils import Bunch, connect, emit

from phy.plot.tests import mouse_click
from ..cluscatter import ClusterScatterView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test cluster scatter view
#------------------------------------------------------------------------------

def test_cluster_scatter_view_1(qtbot, tempdir, gui):
    n_clusters = 1000
    cluster_ids = np.arange(n_clusters)[2::3]

    class Supervisor(object):
        pass
    s = Supervisor()

    def cluster_info(cluster_id):
        np.random.seed(cluster_id)
        return Bunch({
            'fet1': np.random.randn(),
            'fet2': np.random.randn(),
            'fet3': np.random.uniform(low=5, high=20)
        })

    bindings = Bunch({'x_axis': 'fet1', 'y_axis': 'fet2', 'size': 'fet3'})

    v = ClusterScatterView(cluster_info=cluster_info, cluster_ids=cluster_ids, bindings=bindings)
    v.add_color_scheme(
        lambda cluster_id: RandomState(cluster_id).rand(), name='depth',
        colormap='linear', cluster_ids=cluster_ids)
    v.show()
    v.plot()
    v.color_scheme = 'depth'
    qtbot.waitForWindowShown(v.canvas)
    v.attach(gui)

    @connect(sender=v)
    def on_request_select(sender, cluster_ids):
        v.on_select(sender, cluster_ids)

    @connect(sender=v)
    def on_select_more(sender, cluster_ids):
        v.on_select(sender, list(np.concatenate((v.cluster_ids, cluster_ids))))

    v.on_select(s, list(cluster_ids[1::50]))

    up = Bunch(all_cluster_ids=cluster_ids[3::20])
    emit('cluster', s, up)

    v.on_select(s, list(cluster_ids[3::40]))

    v.actions.change_x_axis_to_fet1()
    v.set_x_axis('fet1')
    v.set_y_axis('fet2')
    v.set_size('fet3')

    v.actions.get('Toggle log scale for x_axis').trigger()
    v.actions.get('Toggle log scale for y_axis').trigger()
    v.actions.get('Toggle log scale for size').trigger()

    v.increase_marker_size()
    assert v.status

    # Simulate cluster selection.
    _clicked = []

    w, h = v.canvas.get_size()

    @connect(sender=v)  # noqa
    def on_request_select(sender, cluster_ids):  # noqa
        _clicked.append(cluster_ids)

    @connect(sender=v)  # noqa
    def on_select_more(sender, cluster_ids):  # noqa
        _clicked.append(cluster_ids)

    mouse_click(qtbot, v.canvas, pos=(w / 2, h / 2), button='Left', modifiers=())
    assert len(_clicked) == 1

    mouse_click(
        qtbot, v.canvas, pos=(w / 2, h / 2), button='Left', modifiers=('Shift',))
    assert len(_clicked) == 2

    mouse_click(qtbot, v.canvas, pos=(w * .3, h * .3), button='Left', modifiers=('Control',))
    mouse_click(qtbot, v.canvas, pos=(w * .7, h * .3), button='Left', modifiers=('Control',))
    mouse_click(qtbot, v.canvas, pos=(w * .7, h * .7), button='Left', modifiers=('Control',))
    mouse_click(qtbot, v.canvas, pos=(w * .3, h * .7), button='Left', modifiers=('Control',))

    assert len(v.cluster_ids) >= 1

    _stop_and_close(qtbot, v)
