"""Regression tests for temporary Merge View resource cleanup."""

from phylib.utils.event import _EVENT

from phy.gui.tests.conftest import gui  # noqa: F401

from .test_supervisor import _select, supervisor  # noqa: F401


def test_supervisor_merge_mode_releases_temporary_event_callbacks(supervisor):  # noqa: F811
    _select(supervisor, [30], [20])

    def callbacks_for(callback):
        return [entry for entry in _EVENT._callbacks if entry[2] == callback]

    def retained_by_event_callback(obj):
        retained = []
        for entry in _EVENT._callbacks:
            callback = entry[2]
            cells = getattr(callback, '__closure__', ()) or ()
            if any(cell.cell_contents is obj for cell in cells):
                retained.append(entry)
        return retained

    assert callbacks_for(supervisor._on_cluster_drop) == []
    assert callbacks_for(supervisor._remove_merge_candidate_on_right_click) == []

    for _ in range(2):
        supervisor.toggle_merge_mode()
        merge_view = supervisor.merge_view

        assert len(callbacks_for(supervisor._on_cluster_drop)) == 2
        assert len(callbacks_for(supervisor._remove_merge_candidate_on_right_click)) == 1

        supervisor.toggle_merge_mode()

        assert callbacks_for(supervisor._on_cluster_drop) == []
        assert callbacks_for(supervisor._remove_merge_candidate_on_right_click) == []
        assert all(
            sender not in (merge_view, merge_view.dock) for _, sender, _, _ in _EVENT._callbacks
        )
        assert retained_by_event_callback(merge_view) == []
        assert retained_by_event_callback(merge_view.dock) == []

    close_callback = supervisor._merge_close_callback
    assert close_callback is not None
    assert callbacks_for(close_callback)

    supervisor._save_gui_state(supervisor.gui)

    assert callbacks_for(close_callback) == []
    assert supervisor._merge_close_callback is None
    assert supervisor.gui is None
