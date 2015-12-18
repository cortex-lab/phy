# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging
import os.path as op

from six import string_types

from .qt import (QApplication, QWidget, QDockWidget, QStatusBar, QMainWindow,
                 Qt, QSize, QMetaObject)
from .actions import Actions, Snippets
from phy.utils.event import EventEmitter
from phy.utils import (load_master_config, Bunch, _bunchify,
                       _load_json, _save_json,
                       _ensure_dir_exists, phy_user_dir,)
from phy.utils.plugin import get_plugin, IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GUI main window
# -----------------------------------------------------------------------------

def _try_get_vispy_canvas(view):
    # Get the Qt widget from a VisPy canvas.
    try:
        from vispy.app import Canvas
        if isinstance(view, Canvas):
            view = view.native
    except ImportError:  # pragma: no cover
        pass
    return view


def _try_get_matplotlib_canvas(view):
    # Get the Qt widget from a matplotlib figure.
    try:
        from matplotlib.pyplot import Figure
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
        if isinstance(view, Figure):
            view = FigureCanvasQTAgg(view)
    except ImportError:  # pragma: no cover
        pass
    return view


class DockWidget(QDockWidget):
    """A QDockWidget that can emit events."""
    def __init__(self, *args, **kwargs):
        super(DockWidget, self).__init__(*args, **kwargs)
        self._event = EventEmitter()

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        self.emit('close_widget')
        super(DockWidget, self).closeEvent(e)


def _create_dock_widget(widget, name, closable=True, floatable=True):
    # Create the gui widget.
    dock_widget = DockWidget()
    dock_widget.setObjectName(name)
    dock_widget.setWindowTitle(name)
    dock_widget.setWidget(widget)

    # Set gui widget options.
    options = QDockWidget.DockWidgetMovable
    if closable:
        options = options | QDockWidget.DockWidgetClosable
    if floatable:
        options = options | QDockWidget.DockWidgetFloatable

    dock_widget.setFeatures(options)
    dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea |
                                Qt.RightDockWidgetArea |
                                Qt.TopDockWidgetArea |
                                Qt.BottomDockWidgetArea
                                )

    return dock_widget


def _get_dock_position(position):
    return {'left': Qt.LeftDockWidgetArea,
            'right': Qt.RightDockWidgetArea,
            'top': Qt.TopDockWidgetArea,
            'bottom': Qt.BottomDockWidgetArea,
            }[position or 'right']


class GUI(QMainWindow):
    """A Qt main window holding docking Qt or VisPy widgets.

    `GUI` derives from `QMainWindow`.

    Events
    ------

    close
    show
    add_view
    close_view

    Note
    ----

    Use `connect_()`, not `connect()`, because of a name conflict with Qt.

    """
    def __init__(self,
                 position=None,
                 size=None,
                 name=None,
                 subtitle=None,
                 ):
        # HACK to ensure that closeEvent is called only twice (seems like a
        # Qt bug).
        self._closed = False
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super(GUI, self).__init__()
        QMetaObject.connectSlotsByName(self)
        self.setDockOptions(QMainWindow.AllowTabbedDocks |
                            QMainWindow.AllowNestedDocks |
                            QMainWindow.AnimatedDocks
                            )

        self._set_name(name, subtitle)
        self._set_pos_size(position, size)

        # Mapping {name: menuBar}.
        self._menus = {}

        # We can derive from EventEmitter because of a conflict with connect.
        self._event = EventEmitter()

        # Status bar.
        self._lock_status = False
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # List of attached Actions instances.
        self.actions = []

        # Default actions.
        self._set_default_actions()

        # Create and attach snippets.
        self.snippets = Snippets(self)

    def _set_name(self, name, subtitle):
        if name is None:
            name = self.__class__.__name__
        title = name if not subtitle else name + ' - ' + subtitle
        self.setWindowTitle(title)
        self.setObjectName(name)
        # Set the name in the GUI.
        self.name = name

    def _set_pos_size(self, position, size):
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(QSize(size[0], size[1]))

    def _set_default_actions(self):
        self.default_actions = Actions(self, name='Default', menu='&File')

        @self.default_actions.add(shortcut=('HelpContents', 'h'))
        def show_all_shortcuts():
            for actions in self.actions:
                actions.show_shortcuts()

        @self.default_actions.add(shortcut='ctrl+q')
        def exit():
            self.close()

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def unconnect_(self, *args, **kwargs):
        self._event.unconnect(*args, **kwargs)

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        if self._closed:
            return
        self._closed = True
        res = self.emit('close')
        # Discard the close event if False is returned by one of the callback
        # functions.
        if False in res:  # pragma: no cover
            e.ignore()
            return
        super(GUI, self).closeEvent(e)

    def show(self):
        """Show the window."""
        self.emit('show')
        super(GUI, self).show()

    # Views
    # -------------------------------------------------------------------------

    def _get_view_name(self, view):
        """The view name is the class name followed by 1, 2, or n."""
        name = view.__class__.__name__
        views = self.list_views(name)
        n = len(views) + 1
        return '{:s}{:d}'.format(name, n)

    def add_view(self,
                 view,
                 name=None,
                 position=None,
                 closable=False,
                 floatable=True,
                 floating=None):
        """Add a widget to the main window."""

        name = name or self._get_view_name(view)
        # Set the name in the view.
        view.__name__ = name
        widget = _try_get_vispy_canvas(view)
        widget = _try_get_matplotlib_canvas(widget)

        dock_widget = _create_dock_widget(widget, name,
                                          closable=closable,
                                          floatable=floatable,
                                          )
        self.addDockWidget(_get_dock_position(position), dock_widget)
        if floating is not None:
            dock_widget.setFloating(floating)
        dock_widget.view = view

        # Emit the close_view event when the dock widget is closed.
        @dock_widget.connect_
        def on_close_widget():
            self.emit('close_view', view)

        dock_widget.show()
        self.emit('add_view', view)
        logger.log(5, "Add %s to GUI.", name)
        return dock_widget

    def list_views(self, name='', is_visible=True):
        """List all views which name start with a given string."""
        children = self.findChildren(QWidget)
        return [child.view for child in children
                if isinstance(child, QDockWidget) and
                child.view.__name__.startswith(name) and
                (child.isVisible() if is_visible else True) and
                child.width() >= 10 and
                child.height() >= 10
                ]

    def view_count(self):
        """Return the number of opened views."""
        views = self.list_views()
        counts = defaultdict(lambda: 0)
        for view in views:
            counts[view.__name__] += 1
        return dict(counts)

    # Menu bar
    # -------------------------------------------------------------------------

    def get_menu(self, name):
        """Return or create a menu."""
        if name not in self._menus:
            self._menus[name] = self.menuBar().addMenu(name)
        return self._menus[name]

    # Status bar
    # -------------------------------------------------------------------------

    @property
    def status_message(self):
        """The message in the status bar."""
        return str(self._status_bar.currentMessage())

    @status_message.setter
    def status_message(self, value):
        if self._lock_status:
            return
        self._status_bar.showMessage(str(value))

    def lock_status(self):
        self._lock_status = True

    def unlock_status(self):
        self._lock_status = False

    # State
    # -------------------------------------------------------------------------

    def save_geometry_state(self):
        """Return picklable geometry and state of the window and docks.

        This function can be called in `on_close()`.

        """
        return {
            'geometry': self.saveGeometry(),
            'state': self.saveState(),
        }

    def restore_geometry_state(self, gs):
        """Restore the position of the main window and the docks.

        The gui widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        if not gs:
            return
        if gs.get('geometry', None):
            self.restoreGeometry((gs['geometry']))
        if gs.get('state', None):
            self.restoreState((gs['state']))


# -----------------------------------------------------------------------------
# GUI state, creator, plugins
# -----------------------------------------------------------------------------

class GUIState(Bunch):
    """Represent the state of the GUI: positions of the views and
    all parameters associated to the GUI and views.

    This is automatically loaded from the configuration directory.

    """
    def __init__(self, name='GUI', config_dir=None, **kwargs):
        super(GUIState, self).__init__(**kwargs)
        self.name = name
        self.config_dir = config_dir or phy_user_dir()
        _ensure_dir_exists(op.join(self.config_dir, self.name))
        self.load()

    def get_view_params(self, view_name, *names):
        # TODO: how to choose view index
        return [self.get(view_name + '1', Bunch()).get(name, None)
                for name in names]

    def set_view_params(self, view, **kwargs):
        view_name = view if isinstance(view, string_types) else view.__name__
        if view_name not in self:
            self[view_name] = Bunch()
        self[view_name].update(kwargs)

    @property
    def path(self):
        return op.join(self.config_dir, self.name, 'state.json')

    def load(self):
        """Load the state from the JSON file in the config dir."""
        if not op.exists(self.path):
            logger.debug("The GUI state file `%s` doesn't exist.", self.path)
            # TODO: create the default state.
            return
        assert op.exists(self.path)
        logger.debug("Load the GUI state from `%s`.", self.path)
        self.update(_bunchify(_load_json(self.path)))

    def save(self):
        """Save the state to the JSON file in the config dir."""
        logger.debug("Save the GUI state to `%s`.", self.path)
        _save_json(self.path, {k: v for k, v in self.items()
                               if k not in ('config_dir', 'name')})


class SaveGeometryStatePlugin(IPlugin):
    def attach_to_gui(self, gui, state=None, model=None):

        @gui.connect_
        def on_close():
            gs = gui.save_geometry_state()
            state['geometry_state'] = gs

        @gui.connect_
        def on_show():
            gs = state.get('geometry_state', None)
            gui.restore_geometry_state(gs)


def create_gui(name=None, subtitle=None, model=None,
               plugins=None, config_dir=None):
    """Create a GUI with a model and a list of plugins.

    By default, the list of plugins is taken from the `c.TheGUI.plugins`
    parameter, where `TheGUI` is the name of the GUI class.

    """
    gui = GUI(name=name, subtitle=subtitle)
    name = gui.name
    plugins = plugins or []

    # Load the state.
    state = GUIState(gui.name, config_dir=config_dir)
    gui.state = state

    # If no plugins are specified, load the master config and
    # get the list of user plugins to attach to the GUI.
    plugins_conf = load_master_config()[name].plugins
    plugins_conf = plugins_conf if isinstance(plugins_conf, list) else []
    plugins.extend(plugins_conf)

    # Attach the plugins to the GUI.
    for plugin in plugins:
        logger.debug("Attach plugin `%s` to %s.", plugin, name)
        get_plugin(plugin)().attach_to_gui(gui, state=state, model=model)

    # Save the state to disk.
    @gui.connect_
    def on_close():
        state.save()

    return gui
