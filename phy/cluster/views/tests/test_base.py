# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.utils._color import _colormap
from ..base import ManualClusteringView


#------------------------------------------------------------------------------
# Test manual clustering view
#------------------------------------------------------------------------------

class MyView(ManualClusteringView):
    def on_select(self, cluster_ids, **kwargs):
        for i in range(len(cluster_ids)):
            self.canvas.scatter(pos=.25 * np.random.randn(100, 2), color=_colormap(i, 1))


def test_manual_clustering_view_1(qtbot):
    v = MyView()
    v.canvas.show()
    qtbot.addWidget(v.canvas)
    v.on_select(cluster_ids=[0, 1])

    # qtbot.stop()
    v.canvas.close()
