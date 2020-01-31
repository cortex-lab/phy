# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
from functools import partial
import logging

from .qt import (
    QApplication, QWidget, QDockWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QCheckBox,
    QMenu, QToolBar, QStatusBar, QMainWindow, QMessageBox, Qt, QPoint, QSize, _load_font,
    _wait, prompt, show_box, screenshot as make_screenshot)
from .state import GUIState, _gui_state_path, _get_default_state_path
from .actions import Actions, Snippets
from phylib.utils import emit, connect

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GUI utils
# -----------------------------------------------------------------------------

def _try_get_matplotlib_canvas(view):
    """Get the Qt widget from a matplotlib figure."""
    try:
        from matplotlib.pyplot import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        if isinstance(view, Figure):
            view = FigureCanvasQTAgg(view)
        # Case where the view has a .figure property which is a matplotlib figure.
        elif isinstance(getattr(view, 'figure', None), Figure):
            view = FigureCanvasQTAgg(view.figure)
        elif isinstance(getattr(getattr(view, 'canvas', None), 'figure', None), Figure):
            view = FigureCanvasQTAgg(view.canvas.figure)
    except ImportError as e:  # pragma: no cover
        logger.warning("Import error: %s", e)
    return view


def _try_get_opengl_canvas(view):
    """Convert from QOpenGLWindow to QOpenGLWidget."""
    from phy.plot.base import BaseCanvas
    if isinstance(view, BaseCanvas):
        return QWidget.createWindowContainer(view)
    elif isinstance(getattr(view, 'canvas', None), BaseCanvas):
        return QWidget.createWindowContainer(view.canvas)
    return view


def _widget_position(widget):  # pragma: no cover
    return widget.parentWidget().mapToGlobal(widget.geometry().topLeft())


# -----------------------------------------------------------------------------
# Dock widget
# -----------------------------------------------------------------------------

DOCK_TITLE_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: #232426;
        color: white;
    }

    QPushButton {
        padding: 4px;
        margin: 0 1px;
    }

    QCheckBox {
        padding: 2px 4px;
        margin: 0 1px;
    }

    QLabel {
        padding: 3px;
    }

    QPushButton:hover, QCheckBox:hover {
        background: #323438;
    }

    QPushButton:pressed {
        background: #53575e;
    }

    QPushButton:checked {
        background: #6c717a;
    }
'''


DOCK_STATUS_STYLESHEET = '''
    * {
        padding: 0;
        margin: 0;
        border: 0;
        background: black;
        color: white;
    }

    QLabel {
        padding: 3px;
    }
'''


class DockWidget(QDockWidget):
    """A dock widget with a custom title bar.

    The title bar has a status text at the middle, and a group of buttons on the right.
    By default, the buttons on the right are screenshot and close. New buttons can be added
    in this group, from right to left.

    """

    confirm_before_close_view = False
    max_status_length = 64

    def __init__(self, *args, widget=None, **kwargs):
        super(DockWidget, self).__init__(*args, **kwargs)
        # Load the font awesome font.
        self._font = _load_font('fa-solid-900.ttf')
        self._dock_widgets = {}
        self._widget = widget

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        emit('close_dock_widget', self)
        super(DockWidget, self).closeEvent(e)

    def add_button(
            self, callback=None, text=None, icon=None, checkable=False,
            checked=False, event=None, name=None):
        """Add a button to the dock title bar, to the right.

        Parameters
        ----------

        callback : function
            Callback function when the button is clicked.
        text : str
            Text of the button.
        icon : str
            Fontawesome icon of the button specified as a unicode string with 4 hexadecimal
            characters.
        checkable : boolean
            Whether the button is checkable.
        checked : boolean
            Whether the checkable button is initially checked.
        event : str
            Name of the event that is externally raised when the status of the button is changed.
            This is used to synchronize the button's checked status when the value changes
            via another mean than clicking on the button.
        name : str
            Name of the button.

        """
        if callback is None:
            return partial(
                self.add_button, text=text, icon=icon, name=name,
                checkable=checkable, checked=checked, event=event)

        name = name or getattr(callback, '__name__', None) or text
        assert name
        button = QPushButton(chr(int(icon, 16)) if icon else text)
        if self._font:
            button.setFont(self._font)
        button.setCheckable(checkable)
        if checkable:
            button.setChecked(checked)
        button.setToolTip(name)

        if callback:
            @button.clicked.connect
            def on_clicked(state):
                return callback(state)

        # Change the state of the button when this event is called.
        if event:
            @connect(event=event, sender=self.view)
            def on_state_changed(sender, checked):
                button.setChecked(checked)

        assert name not in self._dock_widgets
        self._dock_widgets[name] = button
        self._buttons_layout.addWidget(button, 1)

        return button

    def add_checkbox(self, callback=None, text=None, checked=False, name=None):
        """Add a checkbox to the dock title bar, to the right.

        Parameters
        ----------

        callback : function
            Callback function when the checkbox is clicked.
        text : str
            Text of the checkbox.
        checked : boolean
            Whether the checkbox is initially checked.
        name : str
            Name of the button.

        """
        if callback is None:
            return partial(self.add_checkbox, text=text, checked=checked, name=name)

        name = name or getattr(callback, '__name__', None) or text
        assert name
        checkbox = QCheckBox(text)
        checkbox.setLayoutDirection(2)
        checkbox.setToolTip(name)
        if checked:
            checkbox.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        if callback:
            @checkbox.stateChanged.connect
            def on_state_changed(state):
                return callback(state == Qt.Checked)

        assert name not in self._dock_widgets
        self._dock_widgets[name] = checkbox
        self._buttons_layout.addWidget(checkbox, 1)

        return checkbox

    def get_widget(self, name):
        """Get a dock title bar widget by its name."""
        return self._dock_widgets[name]

    @property
    def status(self):
        """Current status text of the title bar."""
        return self._status.text()

    def set_status(self, text):
        """Set the status text of the widget."""
        n = self.max_status_length
        if len(text) >= n:
            text = text[:n // 2] + ' ... ' + text[-n // 2:]
        self._status.setText(text)

    def _default_buttons(self):
        """Create the default buttons on the right."""

        # Only show the close button if the dock widget is closable.
        if int(self.features()) % 2 == 1:
            # Close button.
            @self.add_button(name='close', text='✕')
            def on_close(e):  # pragma: no cover
                if not self.confirm_before_close_view or show_box(
                    prompt(
                        "Close %s?" % self.windowTitle(),
                        buttons=['yes', 'no'], title='Close?')) == 'yes':
                    self.close()

        # Screenshot button.
        @self.add_button(name='screenshot', icon='f030')
        def on_screenshot(e):  # pragma: no cover
            if hasattr(self.view, 'screenshot'):
                self.view.screenshot()
            else:
                make_screenshot(self.view)

        # View menu button.
        @self.add_button(name='view_menu', icon='f0c9')
        def on_view_menu(e):  # pragma: no cover
            # Display the view menu.
            button = self._dock_widgets['view_menu']
            x = _widget_position(button).x()
            y = _widget_position(self._widget).y()
            self._menu.exec(QPoint(x, y))

    def _create_menu(self):
        """Create the contextual menu for this view."""
        self._menu = QMenu("%s menu" % self.objectName(), self)

    def _create_title_bar(self):
        """Create the title bar."""
        self._title_bar = QWidget(self)

        self._layout = QHBoxLayout(self._title_bar)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._title_bar.setStyleSheet(DOCK_TITLE_STYLESHEET)

        # Left part of the bar.
        # ---------------------

        # Widget name.
        label = QLabel(self.windowTitle())
        self._layout.addWidget(label)

        # Space.
        # ------
        self._layout.addStretch(1)

        # Buttons on the right.
        # ---------------------

        self._buttons = QWidget(self._title_bar)
        self._buttons_layout = QHBoxLayout(self._buttons)
        self._buttons_layout.setDirection(1)
        self._buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._buttons_layout.setSpacing(1)
        self._buttons.setLayout(self._buttons_layout)

        # Add the default buttons.
        self._default_buttons()

        # Layout margin.
        self._layout.addWidget(self._buttons)
        self._title_bar.setLayout(self._layout)
        self.setTitleBarWidget(self._title_bar)

    def _create_status_bar(self):
        # Dock has requested widget and status bar.
        widget_container = QWidget(self)
        widget_layout = QVBoxLayout(widget_container)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.setSpacing(0)

        widget_layout.addWidget(self._widget, 100)

        # Widget status text.
        self._status = QLabel('')
        self._status.setMaximumHeight(30)
        self._status.setStyleSheet(DOCK_STATUS_STYLESHEET)
        widget_layout.addWidget(self._status, 1)

        widget_container.setLayout(widget_layout)
        self.setWidget(widget_container)


def _create_dock_widget(widget, name, closable=True, floatable=True):
    """Create a dock widget wrapping any Qt widget."""
    dock = DockWidget(widget=widget)
    dock.setObjectName(name)
    dock.setWindowTitle(name)

    # Set gui widget options.
    options = QDockWidget.DockWidgetMovable
    if closable:
        options = options | QDockWidget.DockWidgetClosable
    if floatable:
        options = options | QDockWidget.DockWidgetFloatable

    dock.setFeatures(options)
    dock.setAllowedAreas(
        Qt.LeftDockWidgetArea |
        Qt.RightDockWidgetArea |
        Qt.TopDockWidgetArea |
        Qt.BottomDockWidgetArea
    )

    dock._create_menu()
    dock._create_title_bar()
    dock._create_status_bar()

    return dock


def _get_dock_position(position):
    return {'left': Qt.LeftDockWidgetArea,
            'right': Qt.RightDockWidgetArea,
            'top': Qt.TopDockWidgetArea,
            'bottom': Qt.BottomDockWidgetArea,
            }[position or 'right']


def _prompt_save():  # pragma: no cover
    """Show a prompt asking the user whether he wants to save or not.

    Output is 'save', 'cancel', or 'close'

    """
    b = prompt(
        "Do you want to save your changes before quitting?",
        buttons=['save', 'cancel', 'close'], title='Save')
    return show_box(b)


def _remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# -----------------------------------------------------------------------------
# GUI main window
# -----------------------------------------------------------------------------

class GUI(QMainWindow):
    """A Qt main window containing docking widgets. This class derives from `QMainWindow`.

    Constructor
    -----------

    position : 2-tuple
        Coordinates of the GUI window on the screen, in pixels.
    size : 2-tuple
        Requested size of the GUI window, in pixels.
    name : str
        Name of the GUI window, set in the title bar.
    subtitle : str
        Subtitle of the GUI window, set in the title bar after the name.
    view_creator : dict
        Map view classnames to functions that take no arguments and return a new view instance
        of that class.
    view_count : dict
        Map view classnames to integers specifying the number of views to create for every
        view class.
    default_views : list-like
        List of view names to create by default (overriden by `view_count` if not empty).
    config_dir : str or Path
        User configuration directory used to load/save the GUI state
    enable_threading : boolean
        Whether to enable threading in views or not (used in `ManualClusteringView`).

    Events
    ------

    close(gui)
    show(gui)
    close_view(view, gui)

    """

    default_shortcuts = {
        'enable_snippet_mode': ':',
        'save': 'ctrl+s',
        'about': '?',
        'show_all_shortcuts': 'h',
        'exit': 'ctrl+q',
    }
    default_snippets = {}
    has_save_action = True

    def __init__(
            self, position=None, size=None, name=None, subtitle=None, view_creator=None,
            view_count=None, default_views=None, config_dir=None, enable_threading=True, **kwargs):
        # HACK to ensure that closeEvent is called only twice (seems like a
        # Qt bug).
        self._enable_threading = enable_threading
        self._closed = False
        if not QApplication.instance():  # pragma: no cover
            raise RuntimeError("A Qt application must be created.")
        super(GUI, self).__init__()
        self.setDockOptions(
            QMainWindow.AllowTabbedDocks | QMainWindow.AllowNestedDocks)
        self.setAnimated(False)

        logger.debug("Creating GUI.")

        self._set_name(name, str(subtitle or ''))
        position = position or (200, 200)
        size = size or (800, 600)
        self._set_pos_size(position, size)

        # Registered functions.
        self._registered = {}

        # List of attached Actions instances.
        self.actions = []

        # Mapping {name: menuBar}.
        self._menus = {}
        ds = self.default_shortcuts
        self.file_actions = Actions(self, name='File', menu='&File', default_shortcuts=ds)
        self.view_actions = Actions(self, name='View', menu='&View', default_shortcuts=ds)
        self.help_actions = Actions(self, name='Help', menu='&Help', default_shortcuts=ds)

        # Views,
        self._views = []
        self._view_class_indices = defaultdict(int)  # Dictionary {view_name: next_usable_index}

        # Create the GUI state.
        state_path = _gui_state_path(self.name, config_dir=config_dir)
        default_state_path = kwargs.pop('default_state_path', _get_default_state_path(self))
        self.state = GUIState(state_path, default_state_path=default_state_path, **kwargs)

        # View creator: dictionary {view_class: function_that_adds_view}
        self.default_views = default_views or ()
        self.view_creator = view_creator or {}
        # View count: take the requested one, or the GUI state one.
        self._requested_view_count = (
            view_count if view_count is not None else self.state.get('view_count', {}))
        # If there is still no view count, use a default one.
        self._requested_view_count = self._requested_view_count or {
            view_name: 1 for view_name in default_views or ()}

        # Status bar.
        self._lock_status = False
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Toolbar.
        self._toolbar = QToolBar('Toolbar', self)
        self._toolbar.setObjectName('Toolbar')
        self._toolbar.setIconSize(QSize(24, 24))
        self._toolbar.hide()
        self.addToolBar(self._toolbar)

        # Create and attach snippets.
        self.snippets = Snippets(self)

        @connect(sender=self)
        def on_show(sender):
            logger.debug("Load the geometry state.")
            gs = self.state.get('geometry_state', None)
            self.restore_geometry_state(gs)

    def _set_name(self, name, subtitle):
        """Set the GUI name."""
        if name is None:
            name = self.__class__.__name__
        title = name if not subtitle else name + ' - ' + subtitle
        self.setWindowTitle(title)
        self.setObjectName(name)
        # Set the name in the GUI.
        self.name = name

    def _set_pos_size(self, position, size):
        """Set the position and size of the GUI."""
        if position is not None:
            self.move(position[0], position[1])
        if size is not None:
            self.resize(QSize(size[0], size[1]))

    def set_default_actions(self):
        """Create the default actions (file, views, help...)."""

        # File menu.
        if self.has_save_action:
            @self.file_actions.add(icon='f0c7', toolbar=True)
            def save():
                emit('request_save', self)

        @self.file_actions.add
        def exit():
            """Close the GUI."""
            self.close()

        # Add "Add view" action.
        for view_name in sorted(self.view_creator.keys()):
            self.view_actions.add(
                partial(self.create_and_add_view, view_name),
                name='Add %s' % view_name,
                docstring="Add %s" % view_name,
                show_shortcut=False)
        self.view_actions.separator()

        # Help menu.
        @self.help_actions.add(shortcut=('HelpContents', 'h'))
        def show_all_shortcuts():
            """Show the shortcuts of all actions."""
            for actions in self.actions:
                actions.show_shortcuts()

        @self.help_actions.add(shortcut='?')
        def about():  # pragma: no cover
            """Display an about dialog."""
            from phy import __version_git__
            msg = "phy {} v{}".format(self.name, __version_git__)
            try:
                from phylib import __version__
                msg += "\nphylib v{}".format(__version__)
            except ImportError:
                pass
            QMessageBox.about(self, "About", msg)

    # Events
    # -------------------------------------------------------------------------

    def closeEvent(self, e):
        """Qt slot when the window is closed."""
        if self._closed:
            return
        _wait(250)
        res = emit('close', self)
        # Discard the close event if False is returned by one of the callback
        # functions.
        if False in res:  # pragma: no cover
            e.ignore()
            return
        super(GUI, self).closeEvent(e)
        self._closed = True

        # Save the state to disk when closing the GUI.
        logger.debug("Save the geometry state.")
        gs = self.save_geometry_state()
        self.state['geometry_state'] = gs
        self.state['view_count'] = self.view_count
        self.state.save()

    def show(self):
        """Show the window."""
        emit('show', self)
        super(GUI, self).show()

    # Views
    # -------------------------------------------------------------------------

    @property
    def views(self):
        """Return the list of views in the GUI."""
        # NOTE: need to do a copy because the list will be modified when iterating through
        # views for closing them.
        return self._views.copy()

    @property
    def view_count(self):
        """Return the number of views of every type, as a dictionary mapping view class names
        to an integer."""
        vc = defaultdict(int)
        for v in self.views:
            vc[v.__class__.__name__] += 1
        return dict(vc)

    def list_views(self, *classes):
        """Return the list of views which are instances of one or several classes."""
        s = set(classes)
        return [
            view for view in self._views
            if s.intersection({view.__class__, view.__class__.__name__})]

    def get_view(self, cls, index=0):
        """Return a view from a given class. If there are multiple views of the same class,
        specify the view index (0 by default)."""
        views = self.list_views(cls)
        if index <= len(views) - 1:
            return views[index]

    def _set_view_name(self, view):
        """Set a unique name for a view: view class name, followed by the view index."""
        assert view not in self._views
        # Get all views of the same class.
        cls = view.__class__
        basename = cls.__name__
        views = self.list_views(view.__class__)
        if not views:
            # If the view is the first of its class, just use the base name.
            name = basename
        else:
            # index is the next usable index for the view's class.
            index = self._view_class_indices.get(cls, 0)
            assert index >= 1
            name = '%s (%d)' % (basename, index)
        view.name = name
        return name

    def create_and_add_view(self, view_name):
        """Create a view and add it to the GUI."""
        assert isinstance(view_name, str)
        fn = self.view_creator.get(view_name, None)
        if fn is None:
            return
        # Create the view with the view creation function.
        view = fn()
        if view is None:  # pragma: no cover
            logger.warning("Could not create view %s.", view_name)
            return
        # Attach the view to the GUI if it has an attach(gui) method,
        # otherwise add the view.
        if hasattr(view, 'attach'):
            view.attach(self)
        else:
            self.add_view(view)
        return view

    def create_views(self):
        """Create and add as many views as specified in view_count."""
        self.view_actions.separator()
        # Keep the order of self.default_views.
        view_names = [vn for vn in self.default_views if vn in self._requested_view_count]
        # We add the views in the requested view count, but not in the default views.
        view_names.extend([
            vn for vn in self._requested_view_count.keys() if vn not in self.default_views])
        # Remove duplicates in view names.
        view_names = _remove_duplicates(view_names)
        # We add the view in the order they appear in the default views.
        for view_name in view_names:
            n_views = self._requested_view_count[view_name]
            if n_views <= 0:
                continue
            assert n_views >= 1
            for i in range(n_views):
                self.create_and_add_view(view_name)

    def add_view(self, view, position=None, closable=True, floatable=True, floating=None):
        """Add a dock widget to the main window.

        Parameters
        ----------

        view : View
        position : str
            Relative position where to add the view (left, right, top, bottom).
        closable : boolean
            Whether the view can be closed by the user.
        floatable : boolean
            Whether the view can be detached from the main GUI.
        floating : boolean
            Whether the view should be added in floating mode or not.

        """

        logger.debug("Add view %s to GUI.", view.__class__.__name__)

        name = self._set_view_name(view)
        self._views.append(view)
        self._view_class_indices[view.__class__] += 1

        # Get the Qt canvas for matplotlib/OpenGL views.
        widget = _try_get_matplotlib_canvas(view)
        widget = _try_get_opengl_canvas(widget)

        dock = _create_dock_widget(widget, name, closable=closable, floatable=floatable)
        self.addDockWidget(_get_dock_position(position), dock, Qt.Horizontal)
        if floating is not None:
            dock.setFloating(floating)
        dock.view = view
        view.dock = dock

        # Emit the close_view event when the dock widget is closed.
        @connect(sender=dock)
        def on_close_dock_widget(sender):
            self._views.remove(view)
            emit('close_view', view, self)

        dock.show()
        logger.log(5, "Add %s to GUI.", name)
        return dock

    # Menu bar
    # -------------------------------------------------------------------------

    def get_menu(self, name, insert_before=None):
        """Get or create a menu."""
        if name not in self._menus:
            menu = QMenu(name)
            if not insert_before:
                self.menuBar().addMenu(menu)
            else:
                self.menuBar().insertMenu(self.get_menu(insert_before).menuAction(), menu)
            self._menus[name] = menu
        return self._menus[name]

    def get_submenu(self, menu, name):
        """Get or create a submenu."""
        if name not in self._menus:
            self._menus[name] = self.get_menu(menu).addMenu(name)
        return self._menus[name]

    def remove_menu(self, name):
        """Remove a menu."""
        if name in self._menus:
            menu = self._menus[name]
            menu.clear()
            menu.setVisible(False)
            self.menuBar().removeAction(menu.menuAction())

    def dialog(self, message):
        """Show a message in a dialog box."""
        box = QMessageBox(self)
        box.setText(message)
        return box

    # Status bar
    # -------------------------------------------------------------------------

    @property
    def status_message(self):
        """The message in the status bar, can be set by the user."""
        return str(self._status_bar.currentMessage())

    @status_message.setter
    def status_message(self, value):
        if self._lock_status:
            return
        self._status_bar.showMessage(str(value))

    def lock_status(self):
        """Lock the status bar."""
        self._lock_status = True

    def unlock_status(self):
        """Unlock the status bar."""
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

        The GUI widgets need to be recreated first.

        This function can be called in `on_show()`.

        """
        if not gs:
            return
        if gs.get('geometry', None):
            self.restoreGeometry((gs['geometry']))
        if gs.get('state', None):
            self.restoreState((gs['state']))
