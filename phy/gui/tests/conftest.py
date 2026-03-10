"""Test gui."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

from pytest import fixture

from . import show_and_wait
from ..actions import Actions, Snippets
from ..gui import GUI

# ------------------------------------------------------------------------------
# Utilities and fixtures
# ------------------------------------------------------------------------------


@fixture
def gui(tempdir, qtbot):
    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    gui.set_default_actions()
    qtbot.addWidget(gui)
    show_and_wait(qtbot, gui)
    yield gui
    gui.close()
    del gui


@fixture
def actions(gui):
    yield Actions(gui, name='actions')


@fixture
def snippets(gui):
    yield Snippets(gui)
