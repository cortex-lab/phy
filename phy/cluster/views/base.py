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
    auto_update = True  # automatically update the view when the cluster selection changes
    _default_position = None

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        self._is_busy = False

        # Message to show in the status bar.
        self.status = None

        # List of attributes to save in the GUI view state.
        self.state_attrs = ('auto_update',)

        # List of attributes to save in the local GUI state as well.
        self.local_state_attrs = ()

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

        # Add shortcuts only for the first view of any given type.
        shortcuts = self.shortcuts if not gui.list_views(self.__class__) else None

        gui.add_view(self, position=self._default_position)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

        self.actions = Actions(
            gui, name=self.name, menu=self.name, default_shortcuts=shortcuts)

        # Freeze and unfreeze the view when selecting clusters.
        self.actions.add(self.toggle_auto_update, checkable=True, checked=self.auto_update)
        self.actions.separator()

        emit('view_actions_created', self)

        # Call on_select() asynchronously after a delay, and set a busy
        # cursor.
        self.async_caller = AsyncCaller(delay=1)
        self.async_caller2 = AsyncCaller(delay=10)

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

        # Update the GUI status message when the `self.set_status()` method
        # is called, i.e. when the `status` event is raised by the view.
        @connect(sender=self)  # pragma: no cover
        def on_status(sender=None, e=None):
            gui.status_message = e.message

        # Save the view state in the GUI state.
        @connect(sender=gui)
        def on_close_view(sender, view):
            if view != self:
                return
            logger.debug("Close view %s.", view)
            gui.remove_menu(self.name)
            unconnect(on_select)
            gui.state.update_view_state(self, self.state)
            self.canvas.close()
            gc.collect()

        @connect(sender=gui)
        def on_close(sender):
            gui.state.update_view_state(self, self.state)

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
        return Bunch({key: getattr(self, key, None) for key in self.state_attrs})

    def set_state(self, state):
        """Set the view state.

        The passed object is the persisted `self.state` bunch.

        May be overriden.

        """
        logger.debug("Set state for %s.", getattr(self, 'name', self.__class__.__name__))
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
