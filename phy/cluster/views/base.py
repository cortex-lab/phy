# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from phy.gui import Actions
from phy.gui.qt import AsyncCaller, busy_cursor
from phy.plot import PlotCanvas
from phy.utils import Bunch, connect, unconnect

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

class ManualClusteringView(object):
    """Base class for clustering views.

    The views take their data with functions `cluster_ids: spike_ids, data`.

    """
    default_shortcuts = {
    }
    _callback_delay = 1
    auto_update = True  # automatically update the view when the cluster selection changes

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Message to show in the status bar.
        self.status = None

        # Attached GUI.
        self.gui = None

        self.canvas = PlotCanvas()

    def on_select(self, cluster_ids=None, **kwargs):
        # To override.
        pass

    def attach(self, gui):
        """Attach the view to the GUI."""

        gui.add_view(self)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self, gui))

        # Call on_select() asynchronously after a delay, and set a busy
        # cursor.
        self.async_caller = AsyncCaller(delay=self._callback_delay)

        @connect
        def on_select(sender, cluster_ids, **kwargs):
            if not self.auto_update:
                return
            if sender.__class__.__name__ != 'Supervisor':
                return
            assert isinstance(cluster_ids, list)
            if not cluster_ids:
                return

            # Call this function after a delay unless there is another
            # cluster selection in the meantime.
            @self.async_caller.set
            def update_view():
                with busy_cursor():
                    logger.log(5, "Selecting %s in %s.", cluster_ids, self)
                    self.on_select(cluster_ids=cluster_ids, **kwargs)

        self.actions = Actions(
            gui, name=gui.view_name(self),
            menu=self.__class__.__name__, default_shortcuts=self.shortcuts)

        # Freeze and unfreeze the view when selecting clusters.
        self.actions.add(self.toggle_auto_update, checkable=True, checked=self.auto_update)
        self.actions.separator()

        # Update the GUI status message when the `self.set_status()` method
        # is called, i.e. when the `status` event is raised by the view.
        @connect(sender=self)  # pragma: no cover
        def on_status(sender=None, e=None):
            gui.status_message = e.message

        # Save the view state in the GUI state.
        @connect(sender=gui)
        def on_close(sender=None):
            unconnect(on_select)
            gui.state.update_view_state(self, self.state, gui)
            self.canvas.close()

        self.canvas.show()

    def toggle_auto_update(self, checked):
        """Auto update means the view is updated automatically
        when the cluster selection changes."""
        self.auto_update = checked

    @property
    def state(self):
        """View state.

        This Bunch will be automatically persisted in the GUI state when the
        GUI is closed.

        To be overriden.

        """
        return Bunch(auto_update=self.auto_update)

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

    def show(self):
        return self.canvas.show()

    def close(self):
        self.canvas.close()
