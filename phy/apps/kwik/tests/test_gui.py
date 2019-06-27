# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import shutil
import unittest

from phylib.io.datasets import download_test_file

from phy.apps.tests.test_base import BaseControllerTests
from ..gui import KwikController

logger = logging.getLogger(__name__)


def _kwik_controller(tempdir):
    # Download the dataset.
    paths = list(map(
        download_test_file,
        ('kwik/hybrid_10sec.kwik', 'kwik/hybrid_10sec.kwx', 'kwik/hybrid_10sec.dat')))
    # Copy the dataset to a temporary directory.
    for path in paths:
        shutil.copy(path, tempdir / path.name)
    kwik_path = tempdir / paths[0].name
    return KwikController(kwik_path, channel_group=0)


class KwikControllerTests(BaseControllerTests, unittest.TestCase):
    """Kwik controller."""
    @classmethod
    def get_controller(cls, tempdir):
        return _kwik_controller(tempdir)
