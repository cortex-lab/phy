# -*- coding: utf-8 -*-
from __future__ import print_function

"""Simple event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
import string
import re
from functools import partial

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Event system
#------------------------------------------------------------------------------

class EventEmitter(object):
    """Singleton class that emits events and accepts registered callbacks.

    Example
    -------

    ```python
    class MyClass(EventEmitter):
        def f(self):
            self.emit('my_event', 1, key=2)

    o = MyClass()

    # The following function will be called when `o.f()` is called.
    @o.connect
    def on_my_event(arg, key=None):
        print(arg, key)

    ```

    """

    def __init__(self):
        self._reset()
        self._is_silent = False

    def _reset(self):
        """Remove all registered callbacks."""
        self._callbacks = []

    def _get_on_name(self, func):
        """Return `eventname` when the function name is `on_<eventname>()`."""
        r = re.match("^on_(.+)$", func.__name__)
        if r:
            event = r.group(1)
        else:
            raise ValueError("The function name should be "
                             "`on_<eventname>`().")
        return event

    @contextmanager
    def silent(self):
        """Prevent all callbacks to be called if events are raised
        in the context manager.
        """
        self._is_silent = not(self._is_silent)
        yield
        self._is_silent = not(self._is_silent)

    def connect(self, func=None, event=None, sender=None, **kwargs):
        """Register a callback function to a given event.

        To register a callback function to the `spam` event, where `obj` is
        an instance of a class deriving from `EventEmitter`:

        ```python
        @obj.connect(sender=sender)
        def on_spam(sender, arg1, arg2):
            pass
        ```

        This is called when `obj.emit('spam', sender, arg1, arg2)` is called.

        Several callback functions can be registered for a given event.

        The registration order is conserved and may matter in applications.

        """
        if func is None:
            return partial(self.connect, sender=sender)

        # Get the event name from the function.
        if event is None:
            event = self._get_on_name(func)

        # We register the callback function.
        self._callbacks.append((event, sender, func, kwargs))

        return func

    def unconnect(self, *funcs):
        """Unconnect specified callback functions."""
        self._callbacks = [(event, sender, f, kwargs)
                           for (event, sender, f, kwargs) in self._callbacks if f not in funcs]

    def emit(self, event, sender, *args, **kwargs):
        """Call all callback functions registered with an event.

        Any positional and keyword arguments can be passed here, and they will
        be forwarded to the callback functions.

        Return the list of callback return results.

        """
        if self._is_silent:
            return
        logger.log(10, "Emit event %s.%s", sender.__class__.__name__, event)
        # Call the last callback if this is a single event.
        single = kwargs.pop('single', None)
        res = []
        # Put `last=True` callbacks at the end.
        callbacks = [c for c in self._callbacks if not c[-1].get('last', None)]
        callbacks += [c for c in self._callbacks if c[-1].get('last', None)]
        for e, s, f, k in callbacks:
            if e == event and (s is None or s == sender):
                f_name = getattr(f, '__qualname__', getattr(f, '__name__', str(f)))
                s_name = s.__class__.__name__
                logger.debug("Event callback %s (%s).", f_name, s_name)
                res.append(f(sender, *args, **kwargs))
                if single:
                    return res[-1]
        return res


#------------------------------------------------------------------------------
# Progress reporter
#------------------------------------------------------------------------------

class PartialFormatter(string.Formatter):
    """Prevent KeyError when a format parameter is absent."""
    def get_field(self, field_name, args, kwargs):
        try:
            return super(PartialFormatter, self).get_field(field_name,
                                                           args,
                                                           kwargs)
        except (KeyError, AttributeError):
            return None, field_name

    def format_field(self, value, spec):
        if value is None:
            return '?'
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            return '?'


def _default_on_progress(sender, message, value, value_max, end='\r', **kwargs):
    if value_max == 0:  # pragma: no cover
        return
    if value <= value_max:
        progress = 100 * value / float(value_max)
        fmt = PartialFormatter()
        kwargs['value'] = value
        kwargs['value_max'] = value_max
        print(fmt.format(message, progress=progress, **kwargs), end=end)


def _default_on_complete(message, end='\n', **kwargs):
    # Override the initializing message and clear the terminal
    # line.
    fmt = PartialFormatter()
    print(fmt.format(message + '\033[K', **kwargs), end=end)


class ProgressReporter(object):
    """A class that reports progress done.

    Example
    -------

    ```python
    pr = ProgressReporter()
    pr.set_progress_message("Progress: {progress}%...")
    pr.set_complete_message("Completed!")
    pr.value_max = 10

    for i in range(10):
        pr.value += 1  # or pr.increment()
    ```

    You can also add custom keyword arguments in `pr.increment()`: these
    will be replaced in the message string.

    Emits
    -----

    * `progress(value, value_max)`
    * `complete()`

    """
    def __init__(self):
        super(ProgressReporter, self).__init__()
        self._value = 0
        self._value_max = 0
        self._has_completed = False

    def set_progress_message(self, message, line_break=False):
        """Set a progress message.

        The string needs to contain `{progress}`.

        """

        end = '\r' if not line_break else None

        @connect(sender=self)
        def on_progress(sender, value, value_max, **kwargs):
            kwargs['end'] = None if value == value_max else end
            _default_on_progress(sender, message, value, value_max, **kwargs)

    def set_complete_message(self, message):
        """Set a complete message."""

        @connect(sender=self)
        def on_complete(sender, **kwargs):
            _default_on_complete(message, **kwargs)

    def _set_value(self, value, **kwargs):
        if value < self._value_max:
            self._has_completed = False
        self._value = value
        emit('progress', self, self._value, self._value_max, **kwargs)
        if not self._has_completed and self._value >= self._value_max:
            emit('complete', self, **kwargs)
            self._has_completed = True

    def increment(self, **kwargs):
        """Equivalent to `self.value += 1`.

        Custom keywoard arguments can also be passed to be processed in the
        progress message format string.

        """
        self._set_value(self._value + 1, **kwargs)

    def reset(self, value_max=None):
        """Reset the value to 0 and the value max to a given value."""
        self._value = 0
        if value_max is not None:
            self._value_max = value_max

    @property
    def value(self):
        """Current value (integer)."""
        return self._value

    @value.setter
    def value(self, value):
        self._set_value(value)

    @property
    def value_max(self):
        """Maximum value (integer)."""
        return self._value_max

    @value_max.setter
    def value_max(self, value_max):
        if value_max > self._value_max:
            self._has_completed = False
        self._value_max = value_max

    def is_complete(self):
        """Return whether the task has completed."""
        return self._value >= self._value_max

    def set_complete(self, **kwargs):
        """Set the task as complete."""
        self._set_value(self.value_max, **kwargs)

    @property
    def progress(self):
        """Return the current progress as a float value in `[0, 1]`."""
        return self._value / float(self._value_max)


#------------------------------------------------------------------------------
# Global event system
#------------------------------------------------------------------------------

_EVENT = EventEmitter()

emit = _EVENT.emit
connect = _EVENT.connect
unconnect = _EVENT.unconnect
