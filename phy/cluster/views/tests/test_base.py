"""Test scatter view."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import numpy as np
from phylib.utils import emit

from phy.utils.color import colormaps, selected_cluster_color

from ..base import BaseColorView, ManualClusteringView, SplitSelectionMixin
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


class SplitSelectionView(SplitSelectionMixin, ManualClusteringView):
    def __init__(self):
        super().__init__()
        self.canvas.enable_lasso()


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
        selection_color_order = (0, 2, 1)

    emit('select', Supervisor(), cluster_ids=[0, 1])
    assert v.cluster_color_index(0, 0) == 0
    assert v.cluster_color_index(1, 1) == 2

    v.actions.get('Change color scheme to myscheme').trigger()
    v.next_color_scheme()
    v.previous_color_scheme()
    assert v.get_cluster_colors([0, 1]).shape == (2, 4)

    qtbot.wait(200)
    # qtbot.stop()
    v.canvas.close()
    v.actions.close()
    qtbot.wait(100)


def test_manual_clustering_view_menu_utility_footer(qtbot, gui):
    v = MyView()
    v.attach(gui)
    v.add_color_scheme(lambda cid: cid, name='myscheme')

    v.dock._menu.aboutToShow.emit()
    actions = v.dock._menu.actions()
    assert actions[-3:] == [
        v.actions.get('toggle_auto_update'),
        v.actions.get('screenshot'),
        v.actions.get('close'),
    ]
    assert actions[-4].isSeparator()
    assert not any(
        action.isSeparator() and next_action.isSeparator()
        for action, next_action in zip(actions, actions[1:])
    )

    _stop_and_close(qtbot, v)


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


def test_split_selection_is_exclusive_and_disconnects_on_close(qtbot, gui):
    first = SplitSelectionView()
    second = SplitSelectionView()
    first.attach(gui)
    second.attach(gui)

    first.canvas.lasso.add((0, 0))
    emit('lasso_updated', first.canvas, first.canvas.lasso.polygon)
    assert first.canvas.lasso.count == 1

    second.canvas.lasso.add((0, 0))
    emit('lasso_updated', second.canvas, second.canvas.lasso.polygon)
    assert first.canvas.lasso.count == 0
    assert second.canvas.lasso.count == 1

    first.canvas.lasso.add((0, 0))
    emit('lasso_updated', first.canvas, first.canvas.lasso.polygon)
    assert first.canvas.lasso.count == 1
    assert second.canvas.lasso.count == 0

    first.close()
    calls = []
    first.clear_split_selection = lambda: calls.append(True)
    second.activate_split_selection()
    assert not calls

    _stop_and_close(qtbot, second)
