# -*- coding: utf-8 -*-

"""Test scatter view."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phy.utils import Bunch
from phy.gui import GUI
from ..base import ManualClusteringViewMatplotlib, zoom_fun


#------------------------------------------------------------------------------
# Test matplotlib view
#------------------------------------------------------------------------------

class MyMatplotlibView(ManualClusteringViewMatplotlib):
    def on_select(self, cluster_ids, **kwargs):
        axes = self.subplots(1, len(cluster_ids))
        for ax in axes.flat:
            ax.plot(np.random.randn(100))


def test_mpl_view(qtbot, tempdir):

    v = MyMatplotlibView()

    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select(cluster_ids=[0])
    zoom_fun(v.axes[0, 0], Bunch(xdata=10, ydata=10, button='down'))

    v.on_select(cluster_ids=[0, 1])
    zoom_fun(v.axes[0, 1], Bunch(xdata=10, ydata=10, button='down'))

    # qtbot.stop()
    gui.close()
