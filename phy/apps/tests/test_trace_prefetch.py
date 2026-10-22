"""Navigation policy and shared hint ownership, independent of recording backends."""

import numpy as np
from phylib.utils import Bunch, connect, emit, unconnect
from pytest import mark

from phy.apps._utils import _TraceNavigationPrefetch
from phy.cluster.views.trace import TraceImageView, TraceView
from phy.gui import GUI


class Reader:
    n_samples = 10000

    def __init__(self):
        self.calls = []

    def prefetch(self, sample):
        self.calls.append(('hint', sample))

    def cancel_prefetch(self):
        self.calls.append(('cancel',))


class View:
    sample_rate = 1000.0
    background_prefetch = False


def navigate(reader, view, interval=(1.0, 1.5), direction=1, serial=1, phase='after'):
    emit('trace_navigation', view, interval, direction, 'before', serial)
    reader.calls.append(('read', interval))
    emit('trace_navigation', view, interval, direction, phase, serial)


def test_shared_navigation_ownership():
    reader = Reader()
    adapter = _TraceNavigationPrefetch(reader)
    first, second = View(), View()
    adapter.register(first)
    adapter.register(second)
    try:
        navigate(reader, first)
        assert reader.calls == [('read', (1.0, 1.5))]
        first.background_prefetch = second.background_prefetch = True
        reader.calls.clear()
        navigate(reader, first)
        assert reader.calls == [('read', (1.0, 1.5)), ('hint', 1999)]
        reader.calls.clear()
        navigate(reader, first, (1.1, 1.6))
        assert reader.calls == [('read', (1.1, 1.6)), ('hint', 2099)]
        reader.calls.clear()
        navigate(reader, first, (0.9, 1.4), direction=-1)
        assert reader.calls == [('cancel',), ('read', (0.9, 1.4)), ('hint', 400)]
        reader.calls.clear()
        navigate(reader, second)
        assert reader.calls == [('cancel',), ('read', (1.0, 1.5)), ('hint', 1999)]
        reader.calls.clear()
        emit('trace_prefetch_changed', first, False)
        adapter.remove(first)
        assert reader.calls == []
        navigate(reader, second, direction=None)
        assert reader.calls == [('cancel',), ('read', (1.0, 1.5))]
        reader.calls.clear()
        navigate(reader, second)
        second.background_prefetch = False
        emit('trace_prefetch_changed', second, False)
        assert reader.calls[-1] == ('cancel',)
    finally:
        adapter.close()


def test_hint_bounds_failures_and_disabled_owner_transfer():
    reader = Reader()
    adapter = _TraceNavigationPrefetch(reader)
    first, second = View(), View()
    first.background_prefetch = True
    adapter.register(first)
    adapter.register(second)
    try:
        navigate(reader, first, (0.1, 0.35), direction=-1)
        assert reader.calls[-1] == ('hint', 0)
        reader.calls.clear()
        navigate(reader, first, (9.5, 10.0))
        assert not any(call[0] == 'hint' for call in reader.calls)
        reader.calls.clear()
        navigate(reader, first)
        navigate(reader, second)
        assert reader.calls[-2:] == [('cancel',), ('read', (1.0, 1.5))]
        reader.calls.clear()
        navigate(reader, first, phase='failed')
        assert not any(call[0] == 'hint' for call in reader.calls)
        assert adapter._owner is None
    finally:
        adapter.close()


@mark.parametrize('view_type', [TraceView, TraceImageView])
def test_trace_navigation_intent_and_lifecycle(qtbot, tmp_path, view_type):
    reader = Reader()
    adapter = _TraceNavigationPrefetch(reader)
    events = []

    def get_traces(interval):
        events.append(('read',))
        return Bunch(data=np.ones((max(1, round((interval[1] - interval[0]) * 1000)), 2)))

    gui = GUI(config_dir=tmp_path)
    view = view_type(
        traces=get_traces,
        sample_rate=1000.0,
        duration=10.0,
        n_channels=2,
        spike_times=lambda: np.array([1.0, 2.0]),
    )
    adapter.register(view)
    view.attach(gui)
    gui.show()
    qtbot.wait(1)

    @connect(sender=view)
    def on_trace_navigation(sender, interval, direction, phase, serial):
        events.append((phase, direction, interval))

    try:
        action = view.actions.get('Background prefetch')
        assert not action.isChecked()
        action.trigger()
        assert view.background_prefetch
        assert 'background_prefetch' not in view.state_attrs
        assert 'background_prefetch' not in view.local_state_attrs
        view.set_interval((1.0, 1.5))
        events.clear()
        reader.calls.clear()
        view.go_right()
        assert events[0][0:2] == ('before', 1)
        assert events[-1][0:2] == ('after', 1)
        assert ('read',) in events[1:-1]
        assert reader.calls[-1] == ('hint', 2049)
        reader.calls.clear()
        events.clear()
        view.set_interval()
        view.on_select(cluster_ids=[0])
        view.plot()
        assert not reader.calls
        assert all(event[0] == 'read' for event in events)
        events.clear()
        view.go_to(2.0)
        assert events[0][0:2] == ('before', None)
        assert reader.calls == [('cancel',)]
        view.go_to_start()
        events.clear()
        view.go_left()
        assert not any(event[0] in ('before', 'after') for event in events)
        view.go_right()
        assert adapter._owner is view
        view.dock.hide()
        assert adapter._owner is None
        reader.calls.clear()
        view.go_right()
        assert not reader.calls

        # A vetoed GUI close leaves callbacks registered and navigation usable.
        @connect(sender=gui)
        def on_close(sender):
            return False

        gui.close()
        assert view in adapter._views
        unconnect(on_close)
        gui.close()
        assert not adapter._views
        assert not adapter._guis
        # Recreating a GUI does not reuse opt-in permission.
        second = view_type(traces=get_traces, sample_rate=1000.0, duration=10.0, n_channels=2)
        second_gui = GUI(config_dir=tmp_path)
        adapter.register(second)
        second.attach(second_gui)
        assert not second.background_prefetch
        second_gui.close()
        second.canvas.close()
    finally:
        unconnect(on_trace_navigation)
        gui.close()
        view.canvas.close()
        adapter.close()


def test_old_reader_and_unregister_late_completion():
    adapter = _TraceNavigationPrefetch(object())
    view = View()
    adapter.register(view)
    assert not view._prefetch_available
    adapter.close()
    reader = Reader()
    adapter = _TraceNavigationPrefetch(reader)
    view.background_prefetch = True
    adapter.register(view)
    emit('trace_navigation', view, (1.0, 1.5), 1, 'before', 1)
    adapter.remove(view)
    emit('trace_navigation', view, (1.0, 1.5), 1, 'after', 1)
    assert reader.calls == [('cancel',)]
