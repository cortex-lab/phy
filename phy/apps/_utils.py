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
