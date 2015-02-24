# -*- coding: utf-8 -*-

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

    def connect(self, func=None):
        """Decorator for a function reacting to an event being raised."""
        if func is None:
            return self.connect

        # Get the event name from the function.
        event = self._get_on_name(func)

        # We register the callback function.
        self._callbacks[event].append(func)

        # self.event() should emit the event.
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
        for callback in self._callbacks[event]:
            # Only keep the kwargs that are part of the callback's arg spec.
            kwargs = {n: v for n, v in kwargs.items()
                      if n in getargspec(callback).args}
            callback(*args, **kwargs)
