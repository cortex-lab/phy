"""Shared application resource helpers."""


def _close_trace_reader(traces):
    """Close a phylib trace reader, including readers without a public close method."""
    if traces is None:
        return

    close = getattr(traces, 'close', None)
    if callable(close):
        close()
        return

    arrays = list(getattr(traces, '_mmaps', ()))
    arr = getattr(traces, 'arr', None)
    if arr is not None:
        arrays.append(arr)
    for arr in arrays:
        mmap = getattr(arr, '_mmap', None)
        if mmap is not None and not mmap.closed:
            mmap.close()

    reader = getattr(traces, 'reader', None)
    close = getattr(reader, 'close', None)
    if callable(close):
        close()


class _TraceNavigationPrefetch:
    """Translate explicit view navigation into one shared reader's finite lookahead hint."""

    def __init__(self, reader):
        self.reader = reader
        self.available = all(
            callable(getattr(reader, method, None)) for method in ('prefetch', 'cancel_prefetch')
        )
        self._views = {}
        self._guis = {}
        self._owner = None
        self._direction = None
        self._navigation = None

    def _release(self, view):
        if self._owner is view:
            self.reader.cancel_prefetch()
            self._owner = self._direction = None
        if self._navigation is not None and self._navigation[0] is view:
            self._navigation = None

    def register(self, view):
        from phylib.utils import connect

        view._prefetch_available = self.available
        if not self.available:
            return
        self._views[view] = {'gui': None, 'visibility': None, 'callbacks': []}

        @connect(sender=view)
        def on_trace_navigation(sender, interval, direction, phase, serial):
            self._navigate(view, interval, direction, phase, serial)

        @connect(sender=view)
        def on_trace_prefetch_changed(sender, enabled):
            if not enabled:
                self._release(view)

        @connect(sender=view)
        def on_close_view(sender, gui):
            self.remove(view)

        @connect(sender=view)
        def on_view_attached(sender, gui):
            entry = self._views.get(view)
            if entry is None:
                return
            entry['gui'] = gui

            def visibility(visible):
                if not visible:
                    self._release(view)

            entry['visibility'] = visibility
            view.dock.visibilityChanged.connect(visibility)
            if gui not in self._guis:

                @connect(sender=gui)
                def on_closed(sender):
                    for client, info in list(self._views.items()):
                        if info['gui'] is gui:
                            self.remove(client)

                self._guis[gui] = on_closed

        self._views[view]['callbacks'] = [
            on_trace_navigation,
            on_trace_prefetch_changed,
            on_close_view,
            on_view_attached,
        ]

    def _navigate(self, view, interval, direction, phase, serial):
        if view not in self._views:
            return
        entry = self._views[view]
        permitted = view.background_prefetch and (entry['gui'] is None or view.dock.isVisible())
        token = (view, serial)
        if phase == 'before':
            if self._owner is not None and (
                self._owner is not view
                or direction != self._direction
                or direction is None
                or not permitted
            ):
                self._release(self._owner)
            self._navigation = token
            if direction is not None and permitted:
                self._owner, self._direction = view, direction
        elif token == self._navigation:
            self._navigation = None
            if phase == 'failed':
                self._release(view)
            elif self._owner is view and direction is not None and permitted:
                sr = view.sample_rate
                start, stop = (int(round(t * sr)) for t in interval)
                total = self.reader.n_samples
                if (direction > 0 and stop >= total) or (direction < 0 and start <= 0):
                    self._release(view)
                    return
                horizon = max(1, round(0.5 * sr))
                sample = stop - 1 + horizon if direction > 0 else start - horizon
                self.reader.prefetch(min(total - 1, max(0, sample)))

    def remove(self, view):
        from phylib.utils import unconnect

        entry = self._views.pop(view, None)
        if entry is None:
            return
        self._release(view)
        view.background_prefetch = False
        for callback in entry['callbacks']:
            unconnect(callback)
        if entry['visibility'] is not None:
            view.dock.visibilityChanged.disconnect(entry['visibility'])
        gui = entry['gui']
        if gui in self._guis and not any(info['gui'] is gui for info in self._views.values()):
            unconnect(self._guis.pop(gui))

    def close(self):
        """Detach all clients without closing their shared recording."""
        for view in list(self._views):
            self.remove(view)
