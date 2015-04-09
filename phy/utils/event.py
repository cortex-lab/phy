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
    """A class that reports total progress done with multiple jobs."""
    def __init__(self):
        super(ProgressReporter, self).__init__()
        # A mapping {channel: [value, max_value]}.
        self._channels = {}

    def _value(self, channel):
        return self._channels[channel][0]

    def _max_value(self, channel):
        return self._channels[channel][1]

    def _set_value(self, channel, index, value):
        if channel not in self._channels:
            self._channels[channel] = [0, 0]
        # old_value = self._value(channel)
        max_value = self._max_value(channel)
        if index == 0:
            value = min(value, max_value)
        # if ((index == 0 and value > max_value) or
        #    (index == 1 and old_value > value)):
        #     raise ValueError("The current value {0} ".format(value) +
        #                      "needs to be less "
        #                      "than the maximum value {0}.".format(max_value))
        # else:
        self._channels[channel][index] = value

    def increment(self, *channels, **kwargs):
        """Increment the values of one or multiple channels."""
        increment = kwargs.get('increment', 1)
        self.set(**{channel: (self._value(channel) + increment)
                 for channel in channels})

    def set(self, **values):
        """Set the current values of one or several channels."""
        for channel, value in values.items():
            self._set_value(channel, 0, value)
        current, total = self.current(), self.total()
        self.emit('report', current, total)
        if current == total:
            self.emit('complete')

    def set_max(self, **max_values):
        """Set the maximum values of one or several channels."""
        for channel, max_value in max_values.items():
            self._set_value(channel, 1, max_value)

    def is_complete(self):
        return self.current() == self.total()

    def current(self):
        """Return the total current value."""
        return sum(v[0] for k, v in self._channels.items())

    def total(self):
        """Return the total of the maximum values."""
        return sum(v[1] for k, v in self._channels.items())
