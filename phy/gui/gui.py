# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

from .qt import (QApplication, QWidget, QDockWidget, QStatusBar, QMainWindow,
                 Qt, QSize, QMetaObject)
from .actions import Actions, _show_shortcuts, Snippets
from phy.utils.event import EventEmitter
from phy.utils import load_master_config, Bunch, _load_json, _save_json
from phy.utils.plugin import get_plugin

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

        self._set_name(name)
        self._set_pos_size(position, size)

        # Mapping {name: menuBar}.
        self._menus = {}

        # We can derive from EventEmitter because of a conflict with connect.
        self._event = EventEmitter()

        # Status bar.
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # List of attached Actions instances.
        self.actions = []

        # Default actions.
        self._set_default_actions()

        # Create and attach snippets.
        self.snippets = Snippets(self)

    def _set_name(self, name):
        if name is None:
            name = self.__class__.__name__
        self.setWindowTitle(name)
        self.setObjectName(name)
        # Set the name in the GUI.
        self.__name__ = name

    def _set_pos_size(self, position, size):
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(QSize(size[0], size[1]))

    def _set_default_actions(self):
        self.default_actions = Actions(self)

        @self.default_actions.add(shortcut='ctrl+q', menu='&File')
        def exit():
            self.close()

        @self.default_actions.add(shortcut=('HelpContents', 'h'),
                                  menu='&Help')
        def show_shortcuts():
            shortcuts = self.default_actions.shortcuts
            for actions in self.actions:
                shortcuts.update(actions.shortcuts)
            _show_shortcuts(shortcuts, self.__name__)

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
    def name(self):
        return str(self.windowTitle())

    @property
    def status_message(self):
        """The message in the status bar."""
        return str(self._status_bar.currentMessage())

    @status_message.setter
    def status_message(self, value):
        self._status_bar.showMessage(str(value))

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

class DefaultBunch(defaultdict):
    def __init__(self, *args, **kwargs):
        super(DefaultBunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __missing__(self, item)
        pass


class DefaultDictBunch(defaultdict):
    def __init__(self, **kwargs):
        super(DefaultDictBunch, self).__init__(DefaultBunch, **kwargs)
        self.__dict__ = self


class GUIState(Bunch):
    def __init__(self, geometry_state=None, plugins=None, **kwargs):
        super(GUIState, self).__init__(geomety_state=geometry_state,
                                       plugins=plugins or [],
                                       **kwargs)

    def to_json(self, filename):
        _save_json(filename, self)

    def from_json(self, filename):
        self.update(_load_json(filename))


def create_gui(name=None, model=None, state=None):
    """Create a GUI with a model and a GUI state.

    By default, the list of plugins is taken from the `c.TheGUI.plugins`
    parameter, where `TheGUI` is the name of the GUI class.

    """
    gui = GUI(name=name)
    state = state or GUIState()
    plugins = state.plugins
    # GUI name.
    name = gui.__name__

    # If no plugins are specified, load the master config and
    # get the list of user plugins to attach to the GUI.
    config = load_master_config()
    plugins_conf = config[name].plugins
    plugins_conf = plugins_conf if isinstance(plugins_conf, list) else []
    plugins.extend(plugins_conf)

    # Attach the plugins to the GUI.
    for plugin in plugins:
        logger.debug("Attach plugin `%s` to %s.", plugin, name)
        get_plugin(plugin)().attach_to_gui(gui, state=state, model=model)

    return gui
