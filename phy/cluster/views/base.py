# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from datetime import datetime
import gc
import logging
from pathlib import Path
import sys
import traceback

import numpy as np

from phylib.utils import Bunch, connect, unconnect, emit
from phylib.utils._misc import phy_config_dir
from phylib.utils.geometry import range_transform
from phy.gui import Actions
from phy.gui.qt import AsyncCaller, screenshot, thread_pool, Worker
from phy.plot import PlotCanvas, NDC, extend_bounds

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

_ENABLE_THREADING = True


def _get_bunch_bounds(bunch):
    """Return the data bounds of a bunch."""
    if 'data_bounds' in bunch and bunch.data_bounds is not None:
        return bunch.data_bounds
    xmin, ymin = bunch.pos.min(axis=0)
    xmax, ymax = bunch.pos.max(axis=0)
    return (xmin, ymin, xmax, ymax)


class ManualClusteringView(object):
    """Base class for clustering views.

    Typical property objects:

    - `self.canvas`: a `PlotCanvas` instance by default (can also be a `PlotCanvasMpl` instance).
    - `self.default_shortcuts`: a dictionary with the default keyboard shortcuts for the view
    - `self.shortcuts`: a dictionary with the actual keyboard shortcuts for the view (can be passed
      to the view's constructor).
    - `self.state_attrs`: a tuple with all attributes that should be automatically saved in the
      view's global GUI state.
    - `self.local_state_attrs`: like above, but for the local GUI state (dataset-dependent).

    """
    default_shortcuts = {}
    default_snippets = {}
    auto_update = True  # automatically update the view when the cluster selection changes
    _default_position = None
    plot_canvas_class = PlotCanvas

    def __init__(self, shortcuts=None, **kwargs):

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Message to show in the status bar.
        self.status = None

        # List of attributes to save in the GUI view state.
        self.state_attrs = ('auto_update',)

        # List of attributes to save in the local GUI state as well.
        self.local_state_attrs = ()

        # Attached GUI.
        self.gui = None

        self.canvas = self.plot_canvas_class()

        # Attach the Qt events to this class, so that derived class
        # can override on_mouse_click() and so on.
        self.canvas.attach_events(self)

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _get_data_bounds(self, bunchs):
        """Compute the data bounds."""
        # Return the extended data_bounds if they
        return extend_bounds([_get_bunch_bounds(bunch) for bunch in bunchs])

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids.

        To override.

        """
        return

    def _plot_cluster(self, bunch):
        """Plot one cluster.

        To override.

        """
        pass

    def _update_axes(self):
        """Update the axes."""
        self.canvas.axes.reset_data_bounds(self.data_bounds)

    def plot(self, **kwargs):  # pragma: no cover
        """Update the view with the current cluster selection."""
        bunchs = self.get_clusters_data()
        self.data_bounds = self._get_data_bounds(bunchs)
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self._update_axes()
        self.canvas.update()

    # -------------------------------------------------------------------------
    # Main public methods
    # -------------------------------------------------------------------------

    def on_select(self, cluster_ids=None, **kwargs):
        """Callback functions when clusters are selected. To be overriden."""
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        self.plot(**kwargs)

    def attach(self, gui):
        """Attach the view to the GUI.

        Perform the following:

        - Add the view to the GUI.
        - Update the view's attribute from the GUI state
        - Add the default view actions (auto_update, screenshot)
        - Bind the on_select() method to the select event raised by the supervisor.
          This runs on a background thread not to block the GUI thread.

        """

        # Add shortcuts only for the first view of any given type.
        shortcuts = self.shortcuts if not gui.list_views(self.__class__) else None

        gui.add_view(self, position=self._default_position)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

        self.actions = Actions(
            gui, name=self.name, menu='&View', submenu=self.name,
            default_shortcuts=shortcuts, default_snippets=self.default_snippets)

        # Freeze and unfreeze the view when selecting clusters.
        self.actions.add(
            self.toggle_auto_update, checkable=True, checked=self.auto_update, show_shortcut=False)
        self.actions.add(self.screenshot, show_shortcut=False)
        self.actions.separator()

        emit('view_actions_created', self)

        @connect
        def on_select(sender, cluster_ids, **kwargs):
            # Decide whether the view should react to the select event or not.
            if not self.auto_update:
                return
            if sender.__class__.__name__ != 'Supervisor':
                return
            assert isinstance(cluster_ids, list)
            if not cluster_ids:
                return

            # The view update occurs in a thread in order not to block the main GUI thread.
            # A complication is that OpenGL updates should only occur in the main GUI thread,
            # whereas the computation of the data buffers to upload to the GPU should happen
            # in a thread. Finally, the select events are throttled (more precisely, debounced)
            # to avoid clogging the GUI when many clusters are successively selected, but this
            # is implemented at the level of the table widget, not here.

            # This function executes in the Qt thread pool.
            def _worker():  # pragma: no cover
                try:
                    # All errors happening in the view updates are collected here.
                    self.on_select(cluster_ids=cluster_ids, **kwargs)
                except Exception:
                    logger.debug(''.join(traceback.format_exception(*sys.exc_info())))

            # We launch this function in the thread pool.
            worker = Worker(_worker)

            # Once the worker has finished in the thread, the finished signal is raised,
            # and the callback function below runs on the main GUI thread.
            # All OpenGL updates triggered in the worker (background thread) where recorded
            # instead of being immediately executed (which would have caused errors because
            # OpenGL updates should not be executed from a background thread).
            # Once these updates have been collected in the right order, we execute all of
            # them here, in the main GUI thread.
            @worker.signals.finished.connect
            def finished():
                # When the task has finished in the thread pool, we recover all program
                # updates of the view, and we execute them on the GPU.
                if isinstance(self.canvas, PlotCanvas):
                    self.canvas.set_lazy(False)
                    # We go through all collected OpenGL updates.
                    for program, name, data in self.canvas.iter_update_queue():
                        # We update data buffers in OpenGL programs.
                        program[name] = data
                # Finally, we update the canvas.
                self.canvas.update()
                emit('is_busy', self, False)

            # Start the task on the thread pool, and let the OpenGL canvas know that we're
            # starting to record all OpenGL calls instead of executing them immediately.
            # This is what we call the "lazy" mode.
            emit('is_busy', self, True)
            if _ENABLE_THREADING:
                # This is only for OpenGL views.
                self.canvas.set_lazy(True)
                thread_pool().start(worker)
            else:
                # This is for OpenGL views, without threading.
                worker.run()

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
            logger.debug("Close view %s.", self.name)
            gui.remove_menu(self.name)
            unconnect(on_select)
            gui.state.update_view_state(self, self.state)
            self.canvas.close()
            gc.collect(0)

        @connect(sender=gui)
        def on_close(sender):
            gui.state.update_view_state(self, self.state)

        # HACK: Fix bug on macOS where docked OpenGL widgets were not displayed at startup.
        self._set_floating = AsyncCaller(delay=1)
        @self._set_floating.set
        def _set_floating():
            self.dock_widget.setFloating(False)

    # -------------------------------------------------------------------------
    # Misc public methods
    # -------------------------------------------------------------------------

    def toggle_auto_update(self, checked):
        """When on, the view is automatically updated when the cluster selection changes."""
        self.auto_update = checked

    def screenshot(self, dir=None):
        """Save a PNG screenshot of the view into a given directory. By default, the screenshots
        are saved in `~/.phy/screenshots/`."""
        date = datetime.now().strftime('%Y%m%d%H%M%S')
        name = 'phy_screenshot_%s_%s.png' % (date, self.__class__.__name__)
        path = (Path(dir) if dir else phy_config_dir() / 'screenshots') / name
        path.parent.mkdir(exist_ok=True, parents=True)
        screenshot(self.canvas, path)
        return path

    @property
    def state(self):
        """View state, a Bunch instance automatically persisted in the GUI state when the
        GUI is closed. To be overriden."""
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
        """Set the status bar message in the GUI."""
        message = message or self.status
        if not message:
            return
        self.status = message

    def show(self):
        """Show the underlying canvas."""
        return self.canvas.show()

    def close(self):
        """Close the underlying canvas."""
        self.canvas.close()
        gc.collect()


# -----------------------------------------------------------------------------
# Mixins for manual clustering views
# -----------------------------------------------------------------------------

class ScalingMixin(object):
    """Implement increase, decrease actions, as well as control+wheel shortcut."""
    _scaling_param_increment = 1.1
    _scaling_param_min = .01

    def attach(self, gui):
        super(ScalingMixin, self).attach(gui)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.separator()

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        if e.modifiers == ('Control',):
            if e.delta > 0:
                self.increase()
            else:
                self.decrease()

    def _get_scaling_value(self):  # pragma: no cover
        """Return the scaling parameter. May be overriden."""
        return self.scaling

    def _set_scaling_value(self, value):  # pragma: no cover
        """Set the scaling parameter. May be overriden."""
        self.scaling = value

    def increase(self):
        """Increase the scaling parameter."""
        value = self._get_scaling_value()
        self._set_scaling_value(value * self._scaling_param_increment)

    def decrease(self):
        """Decrease the scaling parameter."""
        value = self._get_scaling_value()
        self._set_scaling_value(max(
            self._scaling_param_min, value / self._scaling_param_increment))


class MarkerSizeMixin(ScalingMixin):
    _marker_size = 5.

    def __init__(self, *args, **kwargs):
        super(MarkerSizeMixin, self).__init__(*args, **kwargs)
        self.state_attrs += ('marker_size',)
        self.local_state_attrs += ('marker_size',)

    # Marker size
    # -------------------------------------------------------------------------

    def _get_scaling_value(self):
        return self.marker_size

    def _set_scaling_value(self, value):
        self.marker_size = value

    @property
    def marker_size(self):
        """Size of the spike markers, in pixels."""
        return self._marker_size

    @marker_size.setter
    def marker_size(self, val):
        assert val > 0
        self._marker_size = val
        self.visual.set_marker_size(val)
        self.canvas.update()


class LassoMixin(object):
    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or not len(self.cluster_ids)):  # pragma: no cover
            return np.array([], dtype=np.int64)

        # Get all points from all clusters.
        pos = []
        spike_ids = []

        # each item is a Bunch with attribute `pos` et `spike_ids`
        bunchs = self.get_clusters_data(load_all=True)
        if bunchs is None:
            return
        for bunch in bunchs:
            # Skip background points.
            # NOTE: we need to make sure that the bunch has a cluster_id key.
            if bunch.cluster_id is None:
                continue
            assert bunch.cluster_id >= 0
            # Load all spikes.
            points = np.c_[bunch.pos]
            pos.append(points)
            spike_ids.append(bunch.spike_ids)
        pos = np.vstack(pos)
        pos = range_transform(self.data_bounds, NDC, pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])

    def attach(self, gui):
        super(LassoMixin, self).attach(gui)
        connect(self.on_request_split)
