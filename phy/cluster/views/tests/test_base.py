# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import emit
from phy.utils.color import selected_cluster_color, colormaps
from ..base import ManualClusteringView
from . import _stop_and_close


#------------------------------------------------------------------------------
# Test manual clustering view
#------------------------------------------------------------------------------

class MyView(ManualClusteringView):
    def plot(self, **kwargs):
        for i in range(len(self.cluster_ids)):
            self.canvas.scatter(pos=.25 * np.random.randn(100, 2), color=selected_cluster_color(i))


def test_manual_clustering_view_1(qtbot, tempdir):
    v = MyView()
    v.canvas.show()
    # qtbot.addWidget(v.canvas)
    v.on_select(cluster_ids=[0, 1])

    v.set_state({'auto_update': False})
    assert v.auto_update is False

    qtbot.wait(10)

    path = v.screenshot(dir=tempdir)
    qtbot.wait(10)

    assert str(path).startswith(str(tempdir))
    assert path.exists()

    v.set_dock_status("hello world")

    _stop_and_close(qtbot, v)


def test_manual_clustering_view_2(qtbot, gui):
    v = MyView()
    v.canvas.show()
    # qtbot.addWidget(v.canvas)
    v.attach(gui)

    class Supervisor(object):
        pass

    emit('select', Supervisor(), cluster_ids=[0, 1])

    v.add_color_scheme(
        lambda cid: cid, name='myscheme', colormap=colormaps.rainbow, cluster_ids=[0, 1])
    v.next_color_scheme()
    v.previous_color_scheme()
    assert v.get_cluster_colors([0, 1]).shape == (2, 4)

    qtbot.wait(200)
    # qtbot.stop()
    v.canvas.close()
    v.actions.close()
    qtbot.wait(100)
