# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import re
from collections import defaultdict
from functools import wraps, partial

from ...utils._misc import _fun_arg_count

# Template for a decorator that accepts both @decorator and @decorator().
# def my_decorator(func=None, arg=None):
#     if func is None:
#         return partial(my_decorator, arg=arg)
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         # TODO
#         return func(*args, **kwargs)
#     return wrapper


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

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # We register the callback function.
        self._callbacks[event].append(wrapper)

        return wrapper

    def action(self, func=None, name=None, event=None):
        """Decorator for a callback function of an action.

        It automatically raises an event named 'event'. If None, the name of
        the event is the name of the function.

        The 'name' argument is used as a title for the GUI button.

        """
        if func is None:
            return partial(self.action, name=name, event=event)

        # By default, the event name is the function name.
        if event is None:
            event = func.__name__

        # Register the action.
        self._actions.append({'func': func, 'name': name, 'event': event})

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the action.
            out = func(*args, **kwargs)
            # Raise an event with the output function as an argument.
            self.emit(event, out)
            return out

        # Set the action function as a Session method.
        setattr(self, func.__name__, wrapper)

        return wrapper

    def emit(self, event, data=None):
        """Call all callback functions registered for that event."""
        for callback in self._callbacks[event]:
            n_args = _fun_arg_count(callback)
            if n_args == 0:
                callback()
            elif n_args == 1:
                callback(data)
