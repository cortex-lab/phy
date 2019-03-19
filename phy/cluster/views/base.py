# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from vispy.util.event import Event

from phy.gui import Actions
from phy.gui.qt import AsyncCaller, busy_cursor
from phy.plot import BaseCanvas
from phy.utils import Bunch, connect, unconnect

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

class StatusEvent(Event):
    def __init__(self, type, message=None):
        super(StatusEvent, self).__init__(type)
        self.message = message


class BaseManualClusteringView(object):
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

        # Keep track of the selected clusters and spikes.
        #self.cluster_ids = None

        super(BaseManualClusteringView, self).__init__(**kwargs)

    def on_select(self, cluster_ids=None, **kwargs):
        # To override.
        pass

    def attach(self, gui, name=None):
        """Attach the view to the GUI."""

        gui.add_view(self)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

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
                    logger.debug("Selecting %s in %s.", cluster_ids, self)
                    self.on_select(cluster_ids=cluster_ids, **kwargs)

        self.actions = Actions(
            gui, name=name or self.__class__.__name__,
            menu=self.__class__.__name__, default_shortcuts=self.shortcuts)

        # Freeze and unfreeze the view when selecting clusters.
        self.actions.add(self.toggle_auto_update, checkable=True, checked=self.auto_update)
        self.actions.separator()

        # Update the GUI status message when the `self.set_status()` method
        # is called, i.e. when the `status` event is raised by the VisPy
        # view.
        @connect(sender=self)  # pragma: no cover
        def on_status(sender=None, e=None):
            gui.status_message = e.message

        # Save the view state in the GUI state.
        @connect(sender=gui)
        def on_close(sender=None):
            unconnect(on_select)
            gui.state.update_view_state(self, self.state)
            self.close()

        self.show()

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


class ManualClusteringView(BaseManualClusteringView, BaseCanvas):
    def __init__(self, *args, **kwargs):
        super(ManualClusteringView, self).__init__(*args, **kwargs)
        self.panzoom._default_zoom = .9
        self.panzoom.reset()
        self.events.add(status=StatusEvent)

    def attach(self, *args, **kwargs):

        # Disable keyboard pan so that we can use arrows as global shortcuts
        # in the GUI.
        self.panzoom.enable_keyboard_pan = False

        super(ManualClusteringView, self).attach(*args, **kwargs)

    def set_status(self, message=None):
        super(ManualClusteringView, self).set_status(message=message)
        self.events.status(message=message)

    def on_mouse_move(self, e):  # pragma: no cover
        self.set_status()


# -----------------------------------------------------------------------------
# Matplotlib view
# -----------------------------------------------------------------------------

def zoom_fun(ax, event):
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    if xdata is None or ydata is None:
        return
    x_left = xdata - cur_xlim[0]
    x_right = cur_xlim[1] - xdata
    y_top = ydata - cur_ylim[0]
    y_bottom = cur_ylim[1] - ydata
    k = 1.3
    scale_factor = {'up': 1. / k, 'down': k}.get(event.button, 1.)
    ax.set_xlim([xdata - x_left * scale_factor,
                 xdata + x_right * scale_factor])
    ax.set_ylim([ydata - y_top * scale_factor,
                 ydata + y_bottom * scale_factor])


class ManualClusteringViewMatplotlib(BaseManualClusteringView):
    def __init__(self, *args, **kwargs):
        super(ManualClusteringViewMatplotlib, self).__init__(*args, **kwargs)
        plt.style.use('dark_background')
        self.figure = plt.figure()

    def subplots(self, nrows=1, ncols=1, **kwargs):
        self.axes = self.figure.subplots(nrows, ncols, squeeze=False, **kwargs)
        for ax in self.axes.flat:
            self.config_ax(ax)
        return self.axes

    def config_ax(self, ax):
        xaxis = ax.get_xaxis()
        yaxis = ax.get_yaxis()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        xaxis.set_ticks_position('bottom')
        xaxis.set_tick_params(direction='out')

        yaxis.set_ticks_position('left')
        yaxis.set_tick_params(direction='out')

        def on_zoom(event):  # pragma: no cover
            zoom_fun(ax, event)
            self.show()

        self.canvas.mpl_connect('scroll_event', on_zoom)

    @property
    def canvas(self):
        return self.figure.canvas

    def attach(self, gui, **kwargs):
        super(ManualClusteringViewMatplotlib, self).attach(gui)
        self.nav = NavigationToolbar(self.canvas, gui, coordinates=False)
        self.nav.pan()

    def show(self):
        self.canvas.draw()

    def close(self):
        self.canvas.close()
