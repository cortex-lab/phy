# -*- coding: utf-8 -*-

"""Manual clustering views."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import gc
import logging

import numpy as np

from phylib.utils import Bunch, connect, unconnect, emit
from phylib.utils.geometry import range_transform
from phy.cluster._utils import RotatingProperty
from phy.gui import Actions
from phy.gui.qt import AsyncCaller, screenshot, screenshot_default_path, thread_pool, Worker
from phy.plot import PlotCanvas, NDC, extend_bounds
from phy.utils.color import ClusterColorSelector

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Manual clustering view
# -----------------------------------------------------------------------------

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

    Events raised:

    - `view_attached(view, gui)`: this is the event to connect to if you write a plugin that
      needs to modify a view.
    - `is_busy(view)`
    - `toggle_auto_update(view)`

    """
    default_shortcuts = {}
    default_snippets = {}
    auto_update = True  # automatically update the view when the cluster selection changes
    _default_position = None
    plot_canvas_class = PlotCanvas
    ex_status = ''  # the GUI can update this to
    max_n_clusters = 0  # By default, show all clusters.

    def __init__(self, shortcuts=None, **kwargs):
        self._lock = None
        self._closed = False
        self.cluster_ids = ()

        # Load default shortcuts, and override with any user shortcuts.
        self.shortcuts = self.default_shortcuts.copy()
        self.shortcuts.update(shortcuts or {})

        # Whether to enable threading. Disabled in tests.
        self._enable_threading = kwargs.get('enable_threading', True)

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

    def _plot_cluster(self, bunch):
        """Plot one cluster.

        To override.

        """
        pass

    def _update_axes(self):
        """Update the axes."""
        self.canvas.axes.reset_data_bounds(self.data_bounds)

    def get_clusters_data(self, load_all=None):
        """Return a list of Bunch instances, with attributes pos and spike_ids.

        To override.

        """
        return

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
        """Callback function when clusters are selected. May be overriden."""
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        self.plot(**kwargs)

    def on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs):
        # Decide whether the view should react to the select event or not.
        if not self.auto_update or self._closed:
            return
        # Only the Supervisor and some specific views can trigger a proper select event.
        if sender.__class__.__name__ in ('ClusterView', 'SimilarityView'):
            return
        assert isinstance(cluster_ids, list)
        if not cluster_ids:
            return
        # Maximum number of clusters that can be displayed in the view, for performance reasons.
        if self.max_n_clusters and len(cluster_ids) > self.max_n_clusters:
            return

        # The lock is used so that two different background threads do not access the same
        # view simultaneously, which can lead to conflicts, errors in the plotting code,
        # and QTimer thread exceptions that lead to frozen OpenGL views.
        if self._lock:
            return
        self._lock = True

        # The view update occurs in a thread in order not to block the main GUI thread.
        # A complication is that OpenGL updates should only occur in the main GUI thread,
        # whereas the computation of the data buffers to upload to the GPU should happen
        # in a thread. Finally, the select events are throttled (more precisely, debounced)
        # to avoid clogging the GUI when many clusters are successively selected, but this
        # is implemented at the level of the table widget, not here.

        # This function executes in the Qt thread pool.
        def _worker():  # pragma: no cover
            self.on_select(cluster_ids=cluster_ids, **kwargs)

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
            # HACK: work-around for https://github.com/cortex-lab/phy/issues/1016
            try:
                self
            except NameError as e:  # pragma: no cover
                logger.warning(str(e))
                return
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
            self._lock = None
            self.update_status()

        # Start the task on the thread pool, and let the OpenGL canvas know that we're
        # starting to record all OpenGL calls instead of executing them immediately.
        # This is what we call the "lazy" mode.
        emit('is_busy', self, True)

        # HACK: disable threading mechanism for now
        # if getattr(gui, '_enable_threading', True):
        if 0:   # pragma: no cover
            # This is only for OpenGL views.
            self.canvas.set_lazy(True)
            thread_pool().start(worker)
        else:
            # This is for OpenGL views, without threading.
            worker.run()
            self._lock = None

    def on_cluster(self, up):
        """Callback function when a clustering action occurs. May be overriden.

        Note: this method is called *before* on_select() so as to give a chance to the view
        to update itself before the selection of the new clusters.

        This method is mostly only useful to views that show all clusters and not just the
        selected clusters (template view, raster view).

        """

    def attach(self, gui):
        """Attach the view to the GUI.

        Perform the following:

        - Add the view to the GUI.
        - Update the view's attribute from the GUI state
        - Add the default view actions (auto_update, screenshot)
        - Bind the on_select() method to the select event raised by the supervisor.

        """

        # Add shortcuts only for the first view of any given type.
        shortcuts = self.shortcuts if not gui.list_views(self.__class__) else None

        gui.add_view(self, position=self._default_position)
        self.gui = gui

        # Set the view state.
        self.set_state(gui.state.get_view_state(self))

        self.actions = Actions(
            gui, name=self.name, view=self,
            default_shortcuts=shortcuts, default_snippets=self.default_snippets)

        # Freeze and unfreeze the view when selecting clusters.
        self.actions.add(
            self.toggle_auto_update, checkable=True, checked=self.auto_update, show_shortcut=False)
        self.actions.add(self.screenshot, show_shortcut=False)
        self.actions.add(self.close, show_shortcut=False)
        self.actions.separator()

        on_select = partial(self.on_select_threaded, gui=gui)
        connect(on_select, event='select')

        # Save the view state in the GUI state.
        @connect
        def on_close_view(view_, gui):
            if view_ != self:
                return
            logger.debug("Close view %s.", self.name)
            self._closed = True
            gui.remove_menu(self.name)
            unconnect(on_select)
            gui.state.update_view_state(self, self.state)
            self.canvas.close()
            gc.collect(0)

        @connect(sender=gui)
        def on_close(sender):
            gui.state.update_view_state(self, self.state)

        # HACK: Fix bug on macOS where docked OpenGL widgets were not displayed at startup.
        self._set_floating = AsyncCaller(delay=5)

        @self._set_floating.set
        def _set_floating():
            self.dock.setFloating(False)

        emit('view_attached', self, gui)

    @property
    def status(self):
        """To be overriden."""
        return ''

    def update_status(self):
        if hasattr(self, 'dock'):
            self.dock.set_status('%s %s' % (self.status, self.ex_status))

    # -------------------------------------------------------------------------
    # Misc public methods
    # -------------------------------------------------------------------------

    def toggle_auto_update(self, checked):
        """When on, the view is automatically updated when the cluster selection changes."""
        logger.debug("%s auto update for %s.", 'Enable' if checked else 'Disable', self.name)
        self.auto_update = checked
        emit('toggle_auto_update', self, checked)

    def screenshot(self, dir=None):
        """Save a PNG screenshot of the view into a given directory. By default, the screenshots
        are saved in `~/.phy/screenshots/`."""
        path = screenshot_default_path(self, dir=dir)
        return screenshot(self.canvas, path=path)

    @property
    def state(self):
        """View state, a Bunch instance automatically persisted in the GUI state when the
        GUI is closed. To be overriden."""
        attrs = set(self.state_attrs + self.local_state_attrs)
        return Bunch({key: getattr(self, key, None) for key in attrs})

    def set_state(self, state):
        """Set the view state.

        The passed object is the persisted `self.state` bunch.

        May be overriden.

        """
        logger.debug("Set state for %s.", getattr(self, 'name', self.__class__.__name__))
        for k, v in state.items():
            setattr(self, k, v)

    def show(self):
        """Show the underlying canvas."""
        return self.canvas.show()

    def close(self):
        """Close the view."""
        if hasattr(self, 'dock'):
            return self.dock.close()
        self.canvas.close()
        self._closed = True
        unconnect(self)
        gc.collect(0)


# -----------------------------------------------------------------------------
# Mixins for manual clustering views
# -----------------------------------------------------------------------------

class BaseWheelMixin(object):
    def on_mouse_wheel(self, e):
        pass


class BaseGlobalView(object):
    """A view that shows all clusters instead of the selected clusters.

    This view shows the clusters in the same order as in the cluster view. It reacts to sorting
    and filtering events.

    The `get_cluster_data()` method (to be overriden) must return a list of Bunch instances
    with each cluster's data, but also the attributes `cluster_rel`, `cluster_idx`, `cluster_id`.

    """

    # All cluster ids, in the order they are shown in the cluster view.
    all_cluster_ids = ()

    # Like all_cluster_ids, but sorted by increasing it. Internal data is stored in this order.
    sorted_cluster_ids = ()

    # For every cluster (sorted by increasing cluster id), its index in all_cluster_ids.
    cluster_idxs = ()

    def _iter_clusters(self):
        """Iterate through all clusters in their natural order (increasing cluster id).

        Yield a tuple (cluster_rel, cluster_idx, cluster_id).

        cluster_rel : int
            Range from 0 to n_clusters - 1.
        cluster_idx : int
            The position of the current cluster in `self.cluster_ids`
        cluster_id : int
            The cluster id.

        """
        for i in range(len(self.all_cluster_ids)):
            yield i, self.cluster_idxs[i], self.sorted_cluster_ids[i]

    def set_cluster_ids(self, cluster_ids):
        pass

    def set_spike_clusters(self, spike_clusters):
        pass

    def update_cluster_sort(self, cluster_ids):
        pass

    def on_select(self, sender=None, cluster_ids=(), **kwargs):
        # Decide whether the view should react to the select event or not.
        if not self.auto_update:
            return
        # Only the Supervisor and some specific views can trigger a proper select event.
        if sender.__class__.__name__ in ('ClusterView', 'SimilarityView'):
            return
        assert isinstance(cluster_ids, list)
        if not cluster_ids:
            return
        self.cluster_ids = cluster_ids  # selected clusters


class BaseColorView(BaseWheelMixin):
    """Provide facilities to add and select color schemes in the view.
    """

    def __init__(self, *args, **kwargs):
        super(BaseColorView, self).__init__(*args, **kwargs)
        self.state_attrs += ('color_scheme',)

        # Color schemes.
        self.color_schemes = RotatingProperty()
        self.add_color_scheme(fun=0, name='blank', colormap='blank', categorical=True)

    def add_color_scheme(
            self, fun=None, name=None, cluster_ids=None,
            colormap=None, categorical=None, logarithmic=None):
        """Add a color scheme to the view. Can be used as follows:

        ```python
        @connect
        def on_view_attached(gui, view):
            view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
        ```

        """
        if fun is None:
            return partial(
                self.add_color_scheme, name=name, cluster_ids=cluster_ids,
                colormap=colormap, categorical=categorical, logarithmic=logarithmic)
        field = name or fun.__name__
        cs = ClusterColorSelector(
            fun, cluster_ids=cluster_ids,
            colormap=colormap, categorical=categorical, logarithmic=logarithmic)
        self.color_schemes.add(field, cs)

    def get_cluster_colors(self, cluster_ids, alpha=1.0):
        """Return the cluster colors depending on the currently-selected color scheme."""
        cs = self.color_schemes.get()
        if cs is None:  # pragma: no cover
            raise RuntimeError("Make sure that at least a color scheme is added.")
        return cs.get_colors(cluster_ids, alpha=alpha)

    def _neighbor_color_scheme(self, dir=+1):
        name = self.color_schemes._neighbor(dir=dir)
        logger.debug("Switch to `%s` color scheme in %s.", name, self.__class__.__name__)
        self.update_color()
        self.update_select_color()
        self.update_status()

    def next_color_scheme(self):
        """Switch to the next color scheme."""
        self._neighbor_color_scheme(+1)

    def previous_color_scheme(self):
        """Switch to the previous color scheme."""
        self._neighbor_color_scheme(-1)

    def update_color(self):
        """Update the cluster colors depending on the current color scheme. To be overriden."""
        pass

    def update_select_color(self):
        """Update the cluster colors after the cluster selection changes."""
        pass

    @property
    def color_scheme(self):
        """Current color scheme."""
        return self.color_schemes.current

    @color_scheme.setter
    def color_scheme(self, color_scheme):
        """Change the current color scheme."""
        logger.debug("Set color scheme to %s.", color_scheme)
        self.color_schemes.set(color_scheme)
        self.update_color()
        self.update_status()

    def attach(self, gui):
        super(BaseColorView, self).attach(gui)
        # Set the current color scheme to the GUI state color scheme (automatically set
        # in self.color_scheme).
        self.color_schemes.set(self.color_scheme)

        # Color scheme actions.
        def _make_color_scheme_action(cs):
            def callback():
                self.color_scheme = cs
            return callback

        for cs in self.color_schemes.keys():
            name = 'Change color scheme to %s' % cs
            self.actions.add(
                _make_color_scheme_action(cs), show_shortcut=False,
                name=name, view_submenu='Change color scheme')

        self.actions.add(self.next_color_scheme)
        self.actions.add(self.previous_color_scheme)
        self.actions.separator()

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        super(BaseColorView, self).on_mouse_wheel(e)
        if e.modifiers == ('Shift',):
            if e.delta > 0:
                self.next_color_scheme()
            elif e.delta < 0:
                self.previous_color_scheme()


class ScalingMixin(BaseWheelMixin):
    """Provide features to change the scaling.

    Implement increase, decrease, reset actions, as well as control+wheel shortcut."""
    _scaling_param_increment = 1.1
    _scaling_param_min = 1e-3
    _scaling_param_max = 1e3
    _scaling_default = 1.0
    _scaling_modifiers = ('Control',)

    def attach(self, gui):
        super(ScalingMixin, self).attach(gui)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.add(self.reset_scaling)
        self.actions.separator()

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        super(ScalingMixin, self).on_mouse_wheel(e)
        if e.modifiers == self._scaling_modifiers:
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
        self._set_scaling_value(min(
            self._scaling_param_max, value * self._scaling_param_increment))

    def decrease(self):
        """Decrease the scaling parameter."""
        value = self._get_scaling_value()
        self._set_scaling_value(max(
            self._scaling_param_min, value / self._scaling_param_increment))

    def reset_scaling(self):
        """Reset the scaling to the default value."""
        self._set_scaling_value(self._scaling_default)


class MarkerSizeMixin(BaseWheelMixin):
    _marker_size = 5.
    _default_marker_size = 5.
    _marker_size_min = 1e-2
    _marker_size_max = 1e2
    _marker_size_increment = 1.1

    def __init__(self, *args, **kwargs):
        super(MarkerSizeMixin, self).__init__(*args, **kwargs)
        self.state_attrs += ('marker_size',)
        self.local_state_attrs += ()

    # Marker size
    # -------------------------------------------------------------------------

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

    def attach(self, gui):
        super(MarkerSizeMixin, self).attach(gui)
        self.actions.add(self.increase_marker_size)
        self.actions.add(self.decrease_marker_size)
        self.actions.add(self.reset_marker_size)
        self.actions.separator()

    def increase_marker_size(self):
        """Increase the scaling parameter."""
        self.marker_size = min(
            self._marker_size_max, self.marker_size * self._marker_size_increment)

    def decrease_marker_size(self):
        """Decrease the scaling parameter."""
        self.marker_size = max(
            self._marker_size_min, self.marker_size / self._marker_size_increment)

    def reset_marker_size(self):
        """Reset the scaling to the default value."""
        self.marker_size = self._default_marker_size

    def on_mouse_wheel(self, e):  # pragma: no cover
        """Change the scaling with the wheel."""
        super(MarkerSizeMixin, self).on_mouse_wheel(e)
        if e.modifiers == ('Alt',):
            if e.delta > 0:
                self.increase_marker_size()
            else:
                self.decrease_marker_size()


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
            if 'cluster_id' in bunch and bunch.cluster_id is None:
                continue
            # Load all spikes.
            points = np.c_[bunch.pos]
            pos.append(points)
            spike_ids.append(bunch.spike_ids)
        if not pos:  # pragma: no cover
            logger.warning("Empty lasso.")
            return np.array([])
        pos = np.vstack(pos)
        pos = range_transform([self.data_bounds], [NDC], pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])

    def attach(self, gui):
        super(LassoMixin, self).attach(gui)
        connect(self.on_request_split)
