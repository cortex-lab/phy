# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.utils import emit
from phylib.utils._color import selected_cluster_color
from ..base import ManualClusteringView


#------------------------------------------------------------------------------
# Test manual clustering view
#------------------------------------------------------------------------------

class MyView(ManualClusteringView):
    def on_select(self, cluster_ids, **kwargs):
        for i in range(len(cluster_ids)):
            self.canvas.scatter(pos=.25 * np.random.randn(100, 2), color=selected_cluster_color(i))


def test_manual_clustering_view_1(qtbot):
    v = MyView()
    v.canvas.show()
    # qtbot.addWidget(v.canvas)
    v.on_select(cluster_ids=[0, 1])

    v.set_state({'auto_update': False})
    assert v.auto_update is False

    qtbot.wait(1)
    # qtbot.stop()
    v.canvas.close()


def test_manual_clustering_view_2(qtbot, gui):
    v = MyView()
    v.canvas.show()
    # qtbot.addWidget(v.canvas)
    v.attach(gui)

    class Supervisor(object):
        pass

    emit('select', Supervisor(), cluster_ids=[0, 1])

    qtbot.wait(200)
    # qtbot.stop()
    v.canvas.close()
    v.dock_widget.close()
    qtbot.wait(100)
