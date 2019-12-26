# -*- coding: utf-8 -*-

"""Testing the Template GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import re
import unittest

import numpy as np

from phylib.io.model import load_model, get_template_params
from phylib.io.tests.conftest import _make_dataset
from phylib.utils.testing import captured_output

from phy.apps.tests.test_base import MinimalControllerTests, BaseControllerTests, GlobalViewsTests
from ..gui import (
    template_describe, TemplateController, TemplateFeatureView)

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _template_controller(tempdir, dir_path, **kwargs):
    kwargs.update(get_template_params(dir_path / 'params.py'))
    return TemplateController(
        config_dir=tempdir / 'config', plugin_dirs=[plugins_dir()],
        clear_cache=kwargs.pop('clear_cache', True),
        clear_state=True, enable_threading=False, **kwargs)


def test_template_describe(qtbot, tempdir):
    model = load_model(_make_dataset(tempdir, param='dense', has_spike_attributes=False))
    with captured_output() as (stdout, stderr):
        template_describe(model.dir_path / 'params.py')
    assert '314' in stdout.getvalue()


class TemplateControllerTests(GlobalViewsTests, BaseControllerTests):
    """Base template controller tests."""
    @classmethod
    def _create_dataset(cls, tempdir):  # pragma: no cover
        """To be overriden in child classes."""
        raise NotImplementedError()

    @classmethod
    def get_controller(cls, tempdir):
        cls._dataset = cls._create_dataset(tempdir)
        return _template_controller(tempdir, cls._dataset.parent)

    @property
    def template_feature_view(self):
        return self.gui.list_views(TemplateFeatureView)[0]

    def test_template_feature_view_split(self):
        self.next()
        self.next()
        self.assertEqual(len(self.selected), 2)
        # Need 2 selected clusters to split in the template feature view.

        n = max(self.cluster_ids)
        self.lasso(self.template_feature_view)
        self.split()

        # Split one cluster => Two new clusters should be selected after the split.
        self.assertTrue((n + 1) in self.selected)

    def test_template_split_init(self):
        self.supervisor.actions.split_init()

    def test_spike_attribute_views(self):
        """Open all available spike attribute views."""
        view_names = [
            name for name in self.controller.view_creator.keys() if name.startswith('Spike')]
        for name in view_names:
            self.gui.create_and_add_view(name)
            self.qtbot.wait(250)


class TemplateControllerDenseTests(TemplateControllerTests, unittest.TestCase):
    """Template controller with a dense dataset."""

    @classmethod
    def _create_dataset(cls, tempdir):
        return _make_dataset(tempdir, param='dense', has_spike_attributes=False)

    def test_amplitude_view(self):
        """Change the amplitude type in the amplitude view."""
        self.next()
        for _ in range(4):
            self.amplitude_view.next_amplitudes_type()
        self.amplitude_view.previous_amplitudes_type()

    def test_z1_close_reopen(self):
        cluster_ids = self.cluster_ids
        spike_clusters = self.supervisor.clustering.spike_clusters

        # Save.
        self.supervisor.save()

        # Close the GUI.
        self.__class__._close_gui()

        # Recreate the controller on the model.
        self.__class__._controller = _template_controller(
            self.__class__._tempdir, self.__class__._dataset.parent,
            clear_cache=False)
        self.__class__._create_gui()

        # Check that the data has been saved.
        self.assertTrue(np.all(self.cluster_ids == cluster_ids))
        self.assertTrue(np.all(self.supervisor.clustering.spike_clusters == spike_clusters))

        # Check the label.
        for cl, val in self.model.metadata[self.__class__._label_name].items():
            self.assertEqual(val, self.__class__._label_value)


class TemplateControllerSparseTests(TemplateControllerTests, unittest.TestCase):
    """Template controller with a sparse dataset."""

    @classmethod
    def _create_dataset(cls, tempdir):
        return _make_dataset(tempdir, param='sparse')


class TemplateControllerMiscTests(TemplateControllerTests, unittest.TestCase):
    """Template controller with a misc model."""

    @classmethod
    def _create_dataset(cls, tempdir):
        return _make_dataset(tempdir, param='misc')

    def test_a1_template_similarity(self):
        self.cluster_view.sort_by('id', 'desc')
        cl = 63
        self.supervisor.select_actions.select(cl)
        self.next()
        self.similarity_view.sort_by('id', 'desc')

        self.assertEqual(self.selected, [cl])

        self.next()
        self.assertEqual(self.selected, [cl, cl - 1])

        self.next()
        self.assertEqual(self.selected, [cl, cl - 2])

        self.merge()

        self.next_best()
        self.next()
        self.assertEqual(self.selected, [cl - 1, cl + 1])

    def test_template_feature_view_split(self):
        # NOTE; the misc dataset has no template_features.
        return


#------------------------------------------------------------------------------
# Test plugins
#------------------------------------------------------------------------------

def plugins_dir():
    """Path to the directory with the builtin plugins."""
    return (Path(__file__).parent / '../../../../plugins').resolve()


def plugin_paths():
    """Iterate over the plugin files."""
    assert plugins_dir().exists()
    yield from sorted(plugins_dir().glob('*.py'))


def plugin_names():
    """Iterate over the builtin plugin names."""
    pattern = re.compile(r'class (\w+)\(IPlugin\)\:')
    for plugin_path in plugin_paths():
        yield pattern.search(Path(plugin_path).read_text()).group(1)


def _make_plugin_test_case(plugin_name):
    """Generate a special test class with a plugin attached to the controller."""

    class TemplateControllerPluginTests(MinimalControllerTests, unittest.TestCase):

        @classmethod
        def _create_dataset(cls, tempdir):
            return _make_dataset(tempdir, param='dense', has_spike_attributes=False)

        @classmethod
        def get_controller(cls, tempdir):
            cls._dataset = cls._create_dataset(tempdir)
            return _template_controller(tempdir, cls._dataset.parent, plugins=[plugin_name])

        def test_a1_plugin_attached(self):
            """Check that the plugin has been attached."""
            self.assertTrue(plugin_name in self.controller.attached_plugins)

        def test_a2_minimal(self):
            """Select one cluster."""
            self.supervisor.reset_wizard()
            self.next_best()
            self.next()
            self.assertEqual(len(self.selected), 2)

    return TemplateControllerPluginTests


# Dynamically define test classes for each builtin plugin.
for plugin_name in plugin_names():
    globals()['TemplateController%sTests' % plugin_name] = _make_plugin_test_case(plugin_name)
