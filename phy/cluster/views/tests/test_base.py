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
        ax = self.subplots(1, 1)[0, 0]
        ax.plot(np.random.randn(100))


def test_mpl_view(qtbot, tempdir):

    v = MyMatplotlibView()

    gui = GUI(config_dir=tempdir)
    gui.show()
    v.attach(gui)
    qtbot.addWidget(gui)

    v.on_select(cluster_ids=[])

    zoom_fun(v.axes[0, 0], Bunch(xdata=10, ydata=10, button='down'))

    # qtbot.stop()
    gui.close()
