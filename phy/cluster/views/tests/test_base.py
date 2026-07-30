"""Test scatter view."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from phylib.utils import emit

from phy.utils.color import colormaps, selected_cluster_color

from ..base import BaseColorView, ManualClusteringView
from . import _stop_and_close

# ------------------------------------------------------------------------------
# Test manual clustering view
# ------------------------------------------------------------------------------


class MyView(BaseColorView, ManualClusteringView):
    def plot(self, **kwargs):
        for i in range(len(self.cluster_ids)):
            self.canvas.scatter(
                pos=0.25 * np.random.randn(100, 2), color=selected_cluster_color(i)
            )

    @property
    def status(self):
        return 'hello'


class DeferredView(ManualClusteringView):
    defer_hidden_updates = True
    max_n_clusters = 2

    def __init__(self):
        super().__init__()
        self.updates = []

    def plot(self, **kwargs):
        self.updates.append((list(self.cluster_ids), kwargs))


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

    _stop_and_close(qtbot, v)


def test_manual_clustering_view_2(qtbot, gui):
    v = MyView()
    v.canvas.show()
    v.add_color_scheme(
        lambda cid: cid, name='myscheme', colormap=colormaps.rainbow, cluster_ids=[0, 1]
    )
    v.attach(gui)

    class Supervisor:
        pass

    emit('select', Supervisor(), cluster_ids=[0, 1])

    v.actions.get('Change color scheme to myscheme').trigger()
    v.next_color_scheme()
    v.previous_color_scheme()
    assert v.get_cluster_colors([0, 1]).shape == (2, 4)

    qtbot.wait(200)
    # qtbot.stop()
    v.canvas.close()
    v.actions.close()
    qtbot.wait(100)


def test_manual_clustering_view_selection_is_limited(qtbot, gui):
    v = MyView()
    v.max_n_clusters = 2
    v.canvas.show()
    v.attach(gui)

    class Supervisor:
        pass

    emit('select', Supervisor(), cluster_ids=[3, 2, 1])

    assert v.cluster_ids == [3, 2]

    _stop_and_close(qtbot, v)


def test_manual_clustering_view_defers_latest_selection_while_hidden(qtbot, gui):
    v = DeferredView()
    v.attach(gui)

    class Supervisor:
        pass

    emit('select', Supervisor(), cluster_ids=[1], marker='visible')
    assert v.updates == [([1], {'marker': 'visible'})]

    v.dock.hide()
    qtbot.waitUntil(lambda: not v._dock_visible)
    emit('select', Supervisor(), cluster_ids=[2], marker='superseded')
    emit('select', Supervisor(), cluster_ids=[3, 4, 5], marker='latest')

    # Public state follows the limited selection, but no hidden plot occurs and
    # only the latest payload remains retained.
    assert v.cluster_ids == [3, 4]
    assert v.updates == [([1], {'marker': 'visible'})]
    assert v._pending_selection == ([3, 4], {'marker': 'latest'})

    v.dock.show()
    qtbot.waitUntil(lambda: len(v.updates) == 2)
    assert v.updates[-1] == ([3, 4], {'marker': 'latest'})
    assert v._pending_selection is None

    _stop_and_close(qtbot, v)


def test_manual_clustering_view_defers_inactive_tab(qtbot, gui):
    hidden = DeferredView()
    visible = DeferredView()
    hidden.attach(gui)
    visible.attach(gui)
    gui.tabifyDockWidget(hidden.dock, visible.dock)
    visible.dock.raise_()
    qtbot.waitUntil(lambda: not hidden._dock_visible and visible._dock_visible)

    class Supervisor:
        pass

    emit('select', Supervisor(), cluster_ids=[7])
    assert hidden.updates == []
    assert visible.updates == [([7], {})]

    hidden.dock.raise_()
    qtbot.waitUntil(lambda: len(hidden.updates) == 1)
    assert hidden.updates == [([7], {})]

    _stop_and_close(qtbot, hidden)
    _stop_and_close(qtbot, visible)
