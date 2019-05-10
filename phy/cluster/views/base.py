# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import gc
import logging

from phy.gui import Actions
from phy.gui.qt import AsyncCaller
from phy.plot import PlotCanvas
from phylib.utils import Bunch, connect, unconnect, emit

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
    _callback_delay = 100
    auto_update = True  # automatically update the view when the cluster selection changes
    _default_position = None

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        self._is_busy = False

        # Message to show in the status bar.
        self.status = None

        # Attached GUI.
        self.gui = None

        self.canvas = PlotCanvas()

        # Attach the Qt events to this class, so that derived class
        # can override on_mouse_click() and so on.
        self.canvas.attach_events(self)

    def on_select(self, cluster_ids=None, **kwargs):
        # To override.
        pass

    def attach(self, gui):
        """Attach the view to the GUI."""

        gui.add_view(self, position=self._default_position)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self, gui))

        # Call on_select() asynchronously after a delay, and set a busy
        # cursor.
        self.async_caller = AsyncCaller(delay=self._callback_delay)
        self.async_caller2 = AsyncCaller(delay=self._callback_delay)

        @connect
        def on_select(sender, cluster_ids, **kwargs):
            if not self.auto_update:
                return
            if sender.__class__.__name__ != 'Supervisor':
                return
            assert isinstance(cluster_ids, list)
            if not cluster_ids:
                return

            # Immediately set is_busy to True.
            emit('is_busy', self, True)
            # Set the view as busy.
            @self.async_caller.set
            def update_view():
                logger.log(5, "Selecting %s in %s.", cluster_ids, self)
                self.on_select(cluster_ids=cluster_ids, **kwargs)
                @self.async_caller2.set
                def finished():
                    logger.log(5, "Done selecting %s in %s.", cluster_ids, self)
                    emit('is_busy', self, False)
                    gc.collect()

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
            gc.collect()

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
        gc.collect()
