# -*- coding: utf-8 -*-

"""Testing the Kwik GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import shutil
import unittest

from phylib.io.datasets import download_test_file
from phylib.utils.testing import captured_output

from phy.apps.tests.test_base import BaseControllerTests
from phy.plot.tests import key_press
from ..gui import KwikController, kwik_describe
from phy.cluster.views import WaveformView

logger = logging.getLogger(__name__)


def _kwik_controller(tempdir, kwik_only=False):
    paths = ['kwik/hybrid_10sec.kwik']
    if not kwik_only:
        paths += ['kwik/hybrid_10sec.kwx', 'kwik/hybrid_10sec.dat']
    for path in paths:
        loc_path = Path(download_test_file(path))
        # Copy the dataset to a temporary directory.
        shutil.copy(loc_path, tempdir / loc_path.name)
    kwik_path = tempdir / 'hybrid_10sec.kwik'
    return KwikController(
        kwik_path, channel_group=0, config_dir=tempdir / 'config',
        clear_cache=True, enable_threading=False)


def test_kwik_describe(qtbot, tempdir):
    temp_path = download_test_file('kwik/hybrid_10sec.kwik')
    kwik_path = tempdir / temp_path.name
    shutil.copy(temp_path, kwik_path)
    with captured_output() as (stdout, stderr):
        kwik_describe(str(kwik_path))
    assert stdout.getvalue()


class KwikControllerTests(BaseControllerTests, unittest.TestCase):
    """Kwik controller."""

    def key(self, key, modifiers=(), delay=550):
        key_press(self.qtbot, self.gui, key, delay=delay, modifiers=modifiers)

    @classmethod
    def get_controller(cls, tempdir):
        return _kwik_controller(tempdir)

    @property
    def waveform_view(self):
        views = self.gui.list_views(WaveformView)
        return views[0] if views else None

    def test_kwik_snippets(self):
        self.key('Down')
        self.key('Space')
        self.key('G')
        self.key('Space')
        self.key('G', modifiers=('Alt',))
        self.key('Z')
        self.key('N', modifiers=('Alt',))
        self.key('Space')
        # Recluster.
        self.key('Colon', delay=10)
        for char in 'RECLUSTER':
            self.key(char, delay=10)
        self.key('Enter')

    def test_kwik_waveform_view(self):
        if not self.waveform_view:
            return
        self.next()
        for _ in range(3):
            self.waveform_view.next_waveforms_type()
            self.qtbot.wait(250)

    def test_kwik_amplitude_view(self):
        if not self.amplitude_view:
            return
        self.next()
        for _ in range(3):
            self.amplitude_view.next_amplitudes_type()
            self.qtbot.wait(250)


class KwikControllerKwikOnlyTests(BaseControllerTests, unittest.TestCase):
    """Kwik controller with just the kwik file."""

    @classmethod
    def get_controller(cls, tempdir):
        return _kwik_controller(tempdir, kwik_only=True)
