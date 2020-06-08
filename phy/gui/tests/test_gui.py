# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import raises

from ..qt import Qt, QApplication, QWidget, QMessageBox
from ..gui import (GUI, Actions,
                   _try_get_matplotlib_canvas,
                   _try_get_opengl_canvas,
                   )
from phy.plot import BaseCanvas
from phylib.utils import connect, unconnect, emit

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

def _create_canvas():
    """Create a GL view."""
    c = BaseCanvas()
    return c


#------------------------------------------------------------------------------
# Test views
#------------------------------------------------------------------------------

def test_matplotlib_view():
    from matplotlib.pyplot import Figure
    assert isinstance(_try_get_matplotlib_canvas(Figure()), QWidget)
    assert isinstance(_try_get_opengl_canvas(Figure()), Figure)

    class MyFigure(object):
        figure = Figure()
    assert isinstance(_try_get_matplotlib_canvas(MyFigure()), QWidget)

    class Canvas(object):
        figure = Figure()

    class MyFigure(object):
        canvas = Canvas()
    assert isinstance(_try_get_matplotlib_canvas(MyFigure()), QWidget)


def test_opengl_view():
    from phy.plot.base import BaseCanvas
    assert isinstance(_try_get_opengl_canvas(BaseCanvas()), QWidget)
    assert isinstance(_try_get_matplotlib_canvas(BaseCanvas()), BaseCanvas)

    class MyFigure(object):
        canvas = BaseCanvas()
    assert isinstance(_try_get_opengl_canvas(MyFigure()), QWidget)


#------------------------------------------------------------------------------
# Test GUI
#------------------------------------------------------------------------------

def test_gui_noapp(tempdir):
    if not QApplication.instance():
        with raises(RuntimeError):  # pragma: no cover
            GUI(config_dir=tempdir)


def test_gui_1(tempdir, qtbot):

    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    gui.set_default_actions()
    qtbot.addWidget(gui)

    assert gui.name == 'GUI'

    # Increase coverage.
    @connect(sender=gui)
    def on_show():
        pass
    unconnect(on_show)
    qtbot.keyPress(gui, Qt.Key_Control)
    qtbot.keyRelease(gui, Qt.Key_Control)

    assert isinstance(gui.dialog("Hello"), QMessageBox)

    dock_view = gui.add_view(_create_canvas(), floating=True, closable=True)
    gui.add_view(_create_canvas())
    dock_view.setFloating(False)
    gui.show()

    assert gui.get_view(BaseCanvas)
    assert len(gui.list_views(BaseCanvas)) == 2

    # Check that the close_widget event is fired when the gui widget is
    # closed.
    _close = []

    @connect(sender=dock_view)
    def on_close_dock_widget(sender):
        _close.append(0)

    @connect(sender=dock_view.view)
    def on_close_view(view_, gui):
        _close.append(1)

    dock_view.close()
    assert _close == [1, 0]

    gui.close()

    assert gui.state.geometry_state['geometry']
    assert gui.state.geometry_state['state']

    gui.help_actions.show_all_shortcuts()
    gui.file_actions.save()
    gui.file_actions.exit()


def test_gui_creator(tempdir, qtbot):
    class MyCanvas(BaseCanvas):
        def __init__(self, *args, **kwargs):
            super(MyCanvas, self).__init__(*args, **kwargs)
            self.actions = Actions(gui, menu='MyCanvas', name='actions')

        def attach(self, gui):
            gui.add_view(self)

    class UnusedClass(BaseCanvas):
        pass

    def _create_my_canvas():
        return MyCanvas()

    # View creator.
    vc = {'BaseCanvas': _create_canvas, 'MyCanvas': _create_my_canvas}

    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir, view_creator=vc)
    gui.set_default_actions()
    qtbot.addWidget(gui)

    # Automatically create the views with the view counts.
    gui._requested_view_count = {'BaseCanvas': 1, 'MyCanvas': 2, 'UnusedClass': 0}
    gui.create_views()
    gui.show()
    qtbot.waitForWindowShown(gui)

    assert gui.view_count == {'BaseCanvas': 1, 'MyCanvas': 2}
    assert len(gui.list_views(BaseCanvas)) == 1

    # Two MyCanvas views.
    views = gui.list_views('MyCanvas')
    assert len(views) == 2

    add_action = gui.view_actions.get('Add MyCanvas')

    # Close the first dock widget.
    views[0].dock.toggleViewAction().activate(0)
    gui.remove_menu('&File')

    # One remaining MyCanvas view.
    views = gui.list_views(MyCanvas)
    assert len(views) == 1
    assert views[0].name == 'MyCanvas (1)'
    assert gui.view_count == {'BaseCanvas': 1, 'MyCanvas': 1}

    # Add a new MyCanvas.
    add_action.activate(0)
    views = gui.list_views(MyCanvas)
    assert len(views) == 2
    assert views[0].name == 'MyCanvas (1)'
    assert views[1].name == 'MyCanvas (2)'
    assert gui.view_count == {'BaseCanvas': 1, 'MyCanvas': 2}

    # qtbot.stop()
    gui.close()


def test_gui_dock_widget_1(qtbot, gui):
    gui.show()

    v = _create_canvas()
    gui.add_view(v)

    def callback(checked):
        pass

    # Add 2 buttons.
    v.dock.add_button(name='b1', text='hello world', callback=callback)

    @v.dock.add_button(
        name='b2', checkable=True, checked=True, icon='f15c', event='button_clicked')
    def callback_1(checked):
        pass

    # Add a checkbox.
    @v.dock.add_checkbox(name='c1', text='checkbox', checked=True)
    def callback_2(checked):
        pass

    # Make sure the second button reacts to events.
    b2 = v.dock.get_widget('b2')
    assert b2.isChecked()
    emit('button_clicked', v, False)
    assert not b2.isChecked()

    # Set and check the title bar status text.
    v.dock.set_status("this is a status")
    assert v.dock.status == 'this is a status'

    # Set and check the title bar status text.
    v.dock.set_status("---very long---" + "------" * 10)
    assert len(v.dock.status) <= v.dock.max_status_length + 5

    b2.click()
    v.dock.get_widget('b1').click()
    v.dock.get_widget('c1').click()


def test_gui_view_action(qtbot, gui):
    gui.show()

    v = _create_canvas()
    gui.add_view(v)

    actions = gui.view_actions

    @actions.add(view=v)
    def my_action_1():
        pass

    @actions.add(view=v, view_submenu='view submenu')
    def my_action_2():
        pass

    actions.my_action_1()
    actions.my_action_2()


def test_gui_menu(qtbot, gui):
    gui.get_menu('&File')
    gui.get_submenu('&File', 'Submenu')

    @gui.file_actions.add(menu='Submenu')
    def my_action():
        pass
    gui.get_menu('&Edit', '&File')


def test_gui_status_message(gui):
    assert gui.status_message == ''
    gui.status_message = ':hello world!'
    assert gui.status_message == ':hello world!'

    gui.lock_status()
    gui.status_message = ''
    assert gui.status_message == ':hello world!'
    gui.unlock_status()
    gui.status_message = ''
    assert gui.status_message == ''


def test_gui_geometry_state(tempdir, qtbot):
    _gs = []
    gui = GUI(size=(800, 600), config_dir=tempdir)
    gui.set_default_actions()
    qtbot.addWidget(gui)

    @connect(sender=gui)
    def on_close(sender):
        _gs.append(gui.save_geometry_state())

    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())

    assert len(gui.list_views(BaseCanvas)) == 3

    gui.close()

    # Recreate the GUI with the saved state.
    gui = GUI(config_dir=tempdir)
    gui.set_default_actions()

    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())
    gui.add_view(_create_canvas())

    @connect(sender=gui)
    def on_show(sender):
        gui.restore_geometry_state(_gs[0])

    assert gui.restore_geometry_state(None) is None

    assert len(gui.list_views(BaseCanvas)) == 3

    gui.show()
    gui.close()
