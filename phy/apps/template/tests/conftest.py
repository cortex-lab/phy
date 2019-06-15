# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import fixture


from phylib.io.tests.conftest import *  # noqa
from phylib.utils.event import reset
from ..gui import TemplateController

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _template_controller(tempdir, model):
    import phy.apps.template.gui
    from phy.cluster.views import base
    from phy.gui.qt import Debouncer

    # Disable threading in the tests for better coverage.
    base._ENABLE_THREADING = False
    delay = Debouncer.delay
    Debouncer.delay = 1
    # HACK: mock _prompt_save to avoid GUI block in test when closing
    prompt = phy.apps.template.gui._prompt_save
    phy.apps.template.gui._prompt_save = lambda: None

    plugins = []

    c = TemplateController(model=model, config_dir=tempdir, plugins=plugins)
    yield c

    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()

    phy.apps.template.gui._prompt_save = prompt
    base._ENABLE_THREADING = True
    Debouncer.delay = delay


@fixture
def template_controller(tempdir, template_model):
    yield from _template_controller(tempdir, template_model)


@fixture
def template_controller_full(tempdir, template_model_full):
    yield from _template_controller(tempdir, template_model_full)
