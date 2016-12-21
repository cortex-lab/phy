# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from vispy.util.event import Event

from phy.gui import Actions
from phy.gui.qt import AsyncCaller, busy_cursor
from phy.plot import View
from phy.utils import Bunch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

class StatusEvent(Event):
    def __init__(self, type, message=None):
        super(StatusEvent, self).__init__(type)
        self.message = message


class ManualClusteringView(View):
    """Base class for clustering views.

    The views take their data with functions `cluster_ids: spike_ids, data`.

    """
    default_shortcuts = {
    }
    _callback_delay = 10

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Message to show in the status bar.
        self.status = None

        # Attached GUI.
        self.gui = None

        # Keep track of the selected clusters and spikes.
        self.cluster_ids = None

        super(ManualClusteringView, self).__init__(**kwargs)
        self.panzoom._default_zoom = .9
        self.panzoom.reset()
        self.events.add(status=StatusEvent)

    def on_select(self, cluster_ids=None, **kwargs):
        cluster_ids = (cluster_ids if cluster_ids is not None
                       else self.cluster_ids)
        self.cluster_ids = list(cluster_ids) if cluster_ids is not None else []
        self.cluster_ids = [int(c) for c in self.cluster_ids]

    def attach(self, gui, name=None):
        """Attach the view to the GUI."""

        # Disable keyboard pan so that we can use arrows as global shortcuts
        # in the GUI.
        self.panzoom.enable_keyboard_pan = False

        gui.add_view(self)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

        # Call on_select() asynchronously after a delay, and set a busy
        # cursor.
        self.async_caller = AsyncCaller(delay=self._callback_delay)

        @gui.connect_
        def on_select(cluster_ids, **kwargs):
            # Call this function after a delay unless there is another
            # cluster selection in the meantime.
            @self.async_caller.set
            def update_view():
                with busy_cursor():
                    self.on_select(cluster_ids, **kwargs)

        self.actions = Actions(gui,
                               name=name or self.__class__.__name__,
                               menu=self.__class__.__name__,
                               default_shortcuts=self.shortcuts)

        # Update the GUI status message when the `self.set_status()` method
        # is called, i.e. when the `status` event is raised by the VisPy
        # view.
        @self.connect
        def on_status(e):
            gui.status_message = e.message

        # Save the view state in the GUI state.
        @gui.connect_
        def on_close():
            gui.state.update_view_state(self, self.state)
            # NOTE: create_gui() already saves the state, but the event
            # is registered *before* we add all views.
            gui.state.save()

        self.show()

    @property
    def state(self):
        """View state.

        This Bunch will be automatically persisted in the GUI state when the
        GUI is closed.

        To be overriden.

        """
        return Bunch()

    def set_state(self, state):
        """Set the view state.

        The passed object is the persisted `self.state` bunch.

        May be overriden.

        """
        for k, v in state.items():
            setattr(self, k, v)

    def set_status(self, message=None):
        message = message or self.status
        if not message:
            return
        self.status = message
        self.events.status(message=message)

    def on_mouse_move(self, e):  # pragma: no cover
        self.set_status()
