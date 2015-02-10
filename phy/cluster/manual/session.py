# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import re
from collections import defaultdict
from functools import wraps, partial
from inspect import getargspec

from ...utils._misc import _fun_arg_count
from ...ext.six import string_types


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

def _get_on_name(func):
    """Return 'eventname' when the function name is `on_<eventname>()`."""
    r = re.match("^on_(.+)$", func.__name__)
    if r:
        event = r.group(1)
    else:
        raise ValueError("The function name should be "
                         "`on_<eventname>`().")
    return event


class Session(object):
    """Provide actions, views, and an event system for creating an interactive
    session."""
    def __init__(self):
        self._callbacks = defaultdict(list)
        self._actions = []

    def connect(self, func=None):
        """Decorator for a function reacting to an event being raised."""
        if func is None:
            return self.connect

        # Get the event name from the function.
        event = _get_on_name(func)

        # We register the callback function.
        self._callbacks[event].append(func)

        return func

    def unconnect(self, *funcs):
        """Unconnect callback functions."""
        for func in funcs:
            for callbacks in self._callbacks.values():
                if func in callbacks:
                    callbacks.remove(func)

    def action(self, func=None, title=None):
        """Decorator for a callback function of an action.

        The 'title' argument is used as a title for the GUI button.

        """
        if func is None:
            return partial(self.action, title=title)

        # HACK: handle the case where the first argument is the title.
        if isinstance(func, string_types):
            return partial(self.action, title=func)

        # Register the action.
        self._actions.append({'func': func, 'title': title})

        # Set the action function as a Session method.
        setattr(self, func.__name__, func)

        return func

    def emit(self, event, *args, **kwargs):
        """Call all callback functions registered for that event."""
        for callback in self._callbacks[event]:
            # Only keep the kwargs that are part of the callback's arg spec.
            kwargs = {n: v for n, v in kwargs.items()
                      if n in getargspec(callback).args}
            callback(*args, **kwargs)
