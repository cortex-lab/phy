# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import unittest

import numpy as np

from phylib.io.model import load_model
from phylib.io.tests.conftest import _make_dataset
from phylib.utils.testing import captured_output

from phy.apps.tests.test_base import BaseControllerTests
from ..gui import (
    template_describe, TemplateController, TemplateFeatureView)

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _template_controller(tempdir, model):
    return TemplateController(
        dir_path=model.dir_path, config_dir=tempdir / 'config', model=model,
        clear_cache=True, enable_threading=False)


def test_template_describe(qtbot, template_path):
    with captured_output() as (stdout, stderr):
        template_describe(template_path)
    assert '314' in stdout.getvalue()


class TemplateControllerTests(BaseControllerTests):
    @classmethod
    def _create_dataset(cls, tempdir):
        """To be overriden in child classes."""
        return _make_dataset(tempdir, param='dense', has_spike_attributes=False)

    @classmethod
    def get_controller(cls, tempdir):
        cls._dataset = cls._create_dataset(tempdir)
        model = load_model(cls._dataset)
        return _template_controller(tempdir, model)

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
        self.assertEqual(self.selected[-1], n + 1)

    def test_template_split_init(self):
        self.supervisor.actions.split_init()


class TemplateControllerDenseTests(TemplateControllerTests, unittest.TestCase):
    """Template controller with a dense dataset."""

    @classmethod
    def _create_dataset(cls, tempdir):
        return _make_dataset(tempdir, param='dense', has_spike_attributes=False)

    def test_z1_close_reopen(self):
        cluster_ids = self.cluster_ids
        spike_clusters = self.supervisor.clustering.spike_clusters

        # Save.
        self.supervisor.save()

        # Close the GUI.
        self.__class__._close_gui()
        # Reload the model of the same dataset.
        model = load_model(self.__class__._dataset)
        # Recreate the controller on the model.
        self.__class__._controller = _template_controller(self.__class__._tempdir, model)
        self.__class__._create_gui()

        # Check that the data has been saved.
        self.assertTrue(np.all(self.cluster_ids == cluster_ids))
        self.assertTrue(np.all(self.supervisor.clustering.spike_clusters == spike_clusters))

        # Check the label.
        for cl, val in self.model.metadata[self.__class__._label_name].items():
            assert val == self.__class__._label_value


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
