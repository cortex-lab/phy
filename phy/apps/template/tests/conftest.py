# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import fixture


from phylib.io.tests.conftest import *  # noqa
from phylib.utils.event import reset
from phy.apps import _copy_gui_state
from ..gui import TemplateController

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def template_controller(tempdir, template_model):
    _copy_gui_state('TemplateGUI', 'template', config_dir=tempdir)
    plugins = []  # ['PrecachePlugin', 'SavePrompt', 'BackupPlugin']
    c = TemplateController(model=template_model,
                           config_dir=tempdir,
                           plugins=plugins)

    yield c

    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()


@fixture
def template_controller_clean(tempdir, template_model_clean):
    return template_controller(tempdir, template_model_clean)
