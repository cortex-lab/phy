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

@fixture
def template_controller(tempdir, template_model):
    plugins = []  # ['PrecachePlugin', 'SavePrompt', 'BackupPlugin']
    c = TemplateController(model=template_model,
                           config_dir=tempdir,
                           plugins=plugins)

    yield c

    # NOTE: make sure all callback functions are unconnected at the end of the tests
    # to avoid side-effects and spurious dependencies between tests.
    reset()
