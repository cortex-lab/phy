# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from phylib.utils._misc import read_python
from phylib.utils.testing import captured_output
from phylib.utils import emit
import phy.cluster.views as cv
from phy.plot.tests import key_press, mouse_click
from ..gui import (
    TemplateController, template_describe, AmplitudeView, TemplateFeatureView, FeatureView)

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_template_describe(qtbot, template_path):
    with captured_output() as (stdout, stderr):
        template_describe(template_path)
    assert '314' in stdout.getvalue()


def test_template_gui_0(qtbot, tempdir, template_controller_full):
    controller, gui = template_controller_full


def test_template_gui_1(qtbot, tempdir, template_controller_full):
    controller, gui = template_controller_full
    # default_views=('WaveformView', 'CorrelogramView', 'AmplitudeView'))
    s = controller.supervisor

    controller.selection.cluster_ids
    controller.selection.colormap
    controller.selection.color_field

    s.select_actions.next()
    s.block()

    s.actions.move_best_to_good()
    s.block()

    assert len(s.selected) == 1
    s.select_actions.next()
    s.block()

    clu_to_merge = s.selected
    assert len(clu_to_merge) == 2

    s.actions.merge()
    s.block()

    clu_merged = s.selected[0]
    s.actions.move_all_to_mua()
    s.block()

    s.actions.split_init()
    s.block()

    s.select_actions.next()
    s.block()

    clu = s.selected[0]
    s.actions.label('some_field', 3)
    s.block()

    s.actions.move_all_to_good()
    s.block()

    assert s.cluster_meta.get('group', clu) == 'good'

    # Save and close.
    s.save()
    gui.close()

    # Create a new controller and a new GUI with the same data.
    params = read_python(tempdir / 'params.py')
    params['dat_path'] = controller.model.dat_path
    controller = TemplateController(
        config_dir=tempdir, clear_cache=True, dir_path=controller.model.dir_path, **params)

    gui = controller.create_gui()
    s = controller.supervisor
    qtbot.wait(1000)

    # Check that the data has been updated.
    assert s.get_labels('some_field')[clu - 1] is None
    assert s.get_labels('some_field')[clu] == 3

    assert s.cluster_meta.get('group', clu) == 'good'
    for clu in clu_to_merge:
        assert clu not in s.clustering.cluster_ids
    assert clu_merged in s.clustering.cluster_ids

    qtbot.wait(50)
    gui.close()


def test_template_gui_views(qtbot, template_controller_full):
    controller, gui = template_controller_full
    s = controller.supervisor

    s.select_actions.next()
    s.block()

    # Emulate filtering in cluster view.
    emit('table_filter', s.cluster_view, s.clustering.cluster_ids[::2])
    qtbot.wait(50)

    emit('table_filter', s.cluster_view, s.clustering.cluster_ids)
    qtbot.wait(50)

    # Emulate sorting in cluster view.
    emit('table_sort', s.cluster_view, s.clustering.cluster_ids[::-1])
    qtbot.wait(50)

    s.view_actions.colormap_rainbow()
    qtbot.wait(50)

    wv = gui.list_views(cv.WaveformView)
    if wv:
        wv = wv[0]
        wv.actions.toggle_mean_waveforms(True)
        wv.actions.next_waveforms_type()
        wv.actions.change_n_spikes_waveforms(200)

    tv = gui.list_views(cv.TraceView)
    if tv:
        tv = tv[0]
        tv.actions.go_to_next_spike()
        tv.actions.go_to_previous_spike()
        tv.actions.toggle_highlighted_spikes(True)
        mouse_click(qtbot, tv.canvas, (100, 100), modifiers=('Control',))
        tv.dock_widget.close()

    # Test raster view.
    rv = gui.list_views(cv.RasterView)
    if rv:
        rv = rv[0]
        s.view_actions.toggle_categorical_colormap(False)

        mouse_click(qtbot, rv.canvas, (10, 10), modifiers=('Control',))
        qtbot.wait(50)

        rv.dock_widget.close()
        qtbot.wait(50)

    # Test template view.
    tmpv = gui.list_views(cv.TemplateView)
    if tmpv:
        tmpv = tmpv[0]
        mouse_click(qtbot, tmpv.canvas, (100, 100), modifiers=('Control',))
        qtbot.wait(50)

        tmpv.dock_widget.close()
        qtbot.wait(50)


def test_template_gui_2(qtbot, template_controller):
    controller, gui = template_controller

    gui._create_and_add_view(cv.WaveformView)
    gui._create_and_add_view(cv.ProbeView)

    key_press(qtbot, gui, 'Down')
    key_press(qtbot, gui, 'Down')
    key_press(qtbot, gui, 'Space')
    key_press(qtbot, gui, 'G')
    key_press(qtbot, gui, 'Space')
    key_press(qtbot, gui, 'G', modifiers=('Alt',))
    key_press(qtbot, gui, 'Z')
    key_press(qtbot, gui, 'N', modifiers=('Alt',))
    key_press(qtbot, gui, 'Space')
    key_press(qtbot, gui, 'Enter')
    key_press(qtbot, gui, 'S', modifiers=('Control',))


def test_template_gui_undo_stack(qtbot, template_controller):
    controller, gui = template_controller
    s = controller.supervisor

    s.select_actions.next()
    s.block()

    s.select_actions.next()
    s.block()

    s.actions.merge()
    s.block()

    for _ in range(5):
        s.actions.undo()
        s.block()

        s.actions.redo()
        s.block()


def test_template_gui_new_views(qtbot, template_controller_full):
    """Test adding new views once clusters are selected."""
    controller, gui = template_controller_full

    controller.supervisor.next_best()
    controller.supervisor.block()

    controller.supervisor.next()
    controller.supervisor.block()

    for view_cls in controller.view_creator.keys():
        gui._create_and_add_view(view_cls)
        qtbot.wait(200)


def test_template_gui_sim(qtbot, template_controller):
    """Ensure that the similarity is refreshed when clusters change."""
    controller, gui = template_controller
    s = controller.supervisor

    s.cluster_view.sort_by('id', 'desc')
    s.select_actions.next()
    s.block()

    s.similarity_view.sort_by('id', 'desc')
    cl = 63
    assert s.selected == [cl]
    s.select_actions.next()
    s.block()

    assert s.selected == [cl, cl - 1]
    s.select_actions.next()
    s.block()

    assert s.selected == [cl, cl - 2]
    s.actions.merge()
    s.block()

    s.select_actions.next_best()
    s.block()

    s.select_actions.next()
    s.block()
    assert s.selected == [cl - 1, cl + 1]


def test_template_gui_amplitude(qtbot, tempdir, template_controller):
    controller, gui = template_controller
    s = controller.supervisor

    s.select_actions.next()
    s.block()

    av = gui.list_views(AmplitudeView)[0]

    cl = 63
    controller.get_template_amplitude(cl)
    controller.get_amplitudes(cl)
    controller.get_mean_spike_template_amplitudes(cl)
    controller.get_mean_spike_raw_amplitudes(cl)

    for _ in range(3):
        av.next_amplitude_type()
        s.select_actions.next()
        s.block()

    av.amplitude_name = 'feature'
    s.select_actions.next()
    s.block()

    # Select feature in feature view.
    fv = gui.list_views(FeatureView)[0]
    w, h = fv.canvas.get_size()
    w, h = w / 4, h / 4
    x, y = w / 2, h / 2
    mouse_click(qtbot, fv.canvas, (x, y), button='Left', modifiers=('Alt',))
    qtbot.wait(50)

    # Split.
    mouse_click(qtbot, av.canvas, (0, 0), modifiers=('Control',))
    mouse_click(qtbot, av.canvas, (w, 0), modifiers=('Control',))
    mouse_click(qtbot, av.canvas, (w, h), modifiers=('Control',))
    mouse_click(qtbot, av.canvas, (0, h), modifiers=('Control',))

    n = max(s.clustering.cluster_ids)

    s.actions.split()
    s.block()

    # Split one cluster => Two new clusters should be selected after the split.
    assert s.selected_clusters[:2] == [n + 1, n + 2]


def test_template_gui_split_template_feature(qtbot, tempdir, template_controller):
    controller, gui = template_controller
    s = controller.supervisor

    s.select_actions.next()
    s.block()
    s.select_actions.next()
    s.block()

    assert len(s.selected) == 2

    tfv = gui.list_views(TemplateFeatureView)
    if not tfv:
        return
    tfv = tfv[0]

    w, h = tfv.canvas.get_size()
    mouse_click(qtbot, tfv.canvas, (1, 1), modifiers=('Control',))
    mouse_click(qtbot, tfv.canvas, (w - 1, 1), modifiers=('Control',))
    mouse_click(qtbot, tfv.canvas, (w - 1, h - 1), modifiers=('Control',))
    mouse_click(qtbot, tfv.canvas, (1, h - 1), modifiers=('Control',))

    n = max(s.clustering.cluster_ids)

    s.actions.split()
    s.block()

    assert s.selected_clusters == [n + 1]
