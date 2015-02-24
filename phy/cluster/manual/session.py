# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial

from ...ext.six import string_types
from ...utils.event import EventEmitter


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

class Session(EventEmitter):
    """Provide actions, views, and an event system for creating an interactive
    session."""
    def __init__(self):
        super(Session, self).__init__()
        self._actions = []

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
