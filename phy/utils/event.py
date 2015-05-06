# -*- coding: utf-8 -*-
from __future__ import print_function

"""Simple event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import re
from collections import defaultdict
from functools import partial
from inspect import getargspec


#------------------------------------------------------------------------------
# Event system
#------------------------------------------------------------------------------

class EventEmitter(object):
    """Class that emits events and accepts registered callbacks."""

    def __init__(self):
        self._callbacks = defaultdict(list)

    def _get_on_name(self, func):
        """Return 'eventname' when the function name is `on_<eventname>()`."""
        r = re.match("^on_(.+)$", func.__name__)
        if r:
            event = r.group(1)
        else:
            raise ValueError("The function name should be "
                             "`on_<eventname>`().")
        return event

    def _create_emitter(self, event):
        """Create a method that emits an event of the same name."""
        if not hasattr(self, event):
            setattr(self, event,
                    lambda *args, **kwargs: self.emit(event, *args, **kwargs))

    def connect(self, func=None, event=None, set_method=False):
        """Decorator for a function reacting to an event being raised."""
        if func is None:
            return partial(self.connect, set_method=set_method)

        # Get the event name from the function.
        if event is None:
            event = self._get_on_name(func)

        # We register the callback function.
        self._callbacks[event].append(func)

        # A new method self.event() emitting the event is created.
        if set_method:
            self._create_emitter(event)

        return func

    def unconnect(self, *funcs):
        """Unconnect callback functions."""
        for func in funcs:
            for callbacks in self._callbacks.values():
                if func in callbacks:
                    callbacks.remove(func)

    def emit(self, event, *args, **kwargs):
        """Call all callback functions registered for that event."""
        for callback in self._callbacks.get(event, []):
            # Only keep the kwargs that are part of the callback's arg spec.
            kwargs = {n: v for n, v in kwargs.items()
                      if n in getargspec(callback).args}
            callback(*args, **kwargs)


#------------------------------------------------------------------------------
# Progress reporter
#------------------------------------------------------------------------------

class ProgressReporter(EventEmitter):
    """A class that reports progress done."""
    def __init__(self, progress_message=None, complete_message=None):
        super(ProgressReporter, self).__init__()
        self._value = 0
        self._value_max = 0
        self._has_completed = False

        if progress_message is not None:
            @self.connect
            def on_progress(value, value_max):
                if value_max == 0:
                    return
                if value < value_max:
                    progress = 100 * value / float(value_max)
                    print(progress_message.format(progress=progress), end='\r')

        if complete_message is not None:
            @self.connect
            def on_complete():
                # Override the initializing message and clear the terminal
                # line.
                print(complete_message + '\033[K', end='\n')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value < self._value_max:
            self._has_completed = False
        self._value = value
        self.emit('progress', self._value, self._value_max)
        if not self._has_completed and self._value >= self._value_max:
            self.emit('complete')
            self._has_completed = True

    @property
    def value_max(self):
        return self._value_max

    @value_max.setter
    def value_max(self, value_max):
        if value_max > self._value_max:
            self._has_completed = False
        self._value_max = value_max

    def is_complete(self):
        return self._value >= self._value_max

    def set_complete(self):
        self.value = self.value_max

    @property
    def progress(self):
        return self._value / float(self._value_max)
