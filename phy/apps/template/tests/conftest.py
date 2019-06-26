# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import fixture
from phy.gui.qt import Debouncer
from phy.gui.widgets import Barrier

from phylib.io.tests.conftest import *  # noqa
from phylib.utils import connect, reset
from ..gui import TemplateController

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------


def _controller(qtbot, tempdir, model, default_views=None):
    Debouncer.delay = 1

    controller = TemplateController(
        dir_path=model.dir_path, config_dir=tempdir / 'config', model=model,
        clear_cache=True, enable_threading=False)
    gui = controller.create_gui(default_views=default_views, do_prompt_save=False)

    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=controller.supervisor.cluster_view)
    connect(b('similarity_view'), event='ready', sender=controller.supervisor.similarity_view)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitForWindowShown(gui)
    b.wait()

    yield controller, gui
    gui.close()

    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()


@fixture
def template_controller(qtbot, tempdir, template_model):
    yield from _controller(qtbot, tempdir, template_model)


@fixture
def template_controller_full(qtbot, tempdir, template_model_full):
    yield from _controller(qtbot, tempdir, template_model_full)


@fixture
def template_controller_empty_gui(qtbot, tempdir, template_model_full):
    yield from _controller(qtbot, tempdir, template_model_full, default_views=())
