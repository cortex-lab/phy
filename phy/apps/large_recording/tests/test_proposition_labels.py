"""SLAY annotations alter only sampled correlogram labels, with explicit overlaps."""

import numpy as np
from phylib.utils import emit

from phy.apps.large_recording.views import SampledCorrelogramView
from phy.cluster._proposition_annotations import WHITE, proposition_label_color
from phy.cluster._propositions import decode_curation_mapping
from phy.gui import GUI

from .test_adapter import make_controller


def _mapping(groups):
    return {
        'format_version': '2',
        'unit_ids': list(dict.fromkeys(unit for group in groups for unit in group)),
        'merges': [{'unit_ids': group} for group in groups],
    }


def _provider(ids, bin_size, window_size):
    n = len(ids)
    return np.ones((n, n, 5), dtype=np.int64), np.zeros((n, n))


def test_pdf_axis_colors_legend_overlap_and_label_only_refresh(qtbot, tmp_path):
    catalog = decode_curation_mapping(_mapping([[449, 461, 466, 467, 469], [479, 491, 498]]))
    keys = tuple(p.key for p in catalog.propositions)
    context = [catalog, keys[:1], keys]
    calls = []

    def provider(*args):
        calls.append(args)
        return _provider(*args)

    gui = GUI(config_dir=tmp_path / 'config')
    qtbot.addWidget(gui)
    view = SampledCorrelogramView(
        correlograms=provider, proposition_context=lambda: tuple(context), sample_rate=30000
    )
    try:
        view.attach(gui)
        with qtbot.waitExposed(gui):
            gui.show()
        view.on_select(cluster_ids=[449, 461, 466, 467, 469, 479, 491, 498, 511])
        texts = view.text_visual._acc.text
        assert texts == [str(unit) for unit in view.cluster_ids for _ in range(2)]
        expected = [proposition_label_color(0)] * 5 + [proposition_label_color(1)] * 3 + [WHITE]
        np.testing.assert_allclose(view.text_visual._acc.color, np.repeat(expected, 2, axis=0))
        assert 'P1 (active)' in view.proposition_legend.text()
        assert 'P2' in view.proposition_legend.text()
        assert 'pending now' in view.proposition_legend.toolTip()
        histogram = view.correlogram_visual._acc.hist.copy()
        colors = view.correlogram_visual._acc.color.copy()
        # Filtering/sorting tables affects neither membership nor label slots.
        context[2] = keys[:1]
        view.refresh_proposition_labels()
        assert len(calls) == 1
        assert 'hidden by proposition filter' in view.proposition_legend.toolTip()
        np.testing.assert_allclose(view.text_visual._acc.color, np.repeat(expected, 2, axis=0))
        # A review-only update invalidates the red label group without rebinning.
        context[0] = catalog.reject(keys[1])
        view.refresh_proposition_labels()
        assert len(calls) == 1
        assert view._proposition_labels.annotations[479].color == WHITE
        np.testing.assert_array_equal(view.correlogram_visual._acc.hist, histogram)
        np.testing.assert_array_equal(view.correlogram_visual._acc.color, colors)
        # An overlapping source proposition adds a marker to both axis labels.
        context[0] = decode_curation_mapping(
            _mapping([[449, 461, 466, 467, 469], [479, 491, 498], [449, 479]])
        )
        view.refresh_proposition_labels()
        assert texts != view.text_visual._acc.text
        assert view.text_visual._acc.text[:2] == ['449*', '449*']
        assert '449*: P1, P3' in view.proposition_legend.toolTip()
        assert '* multiple groups' in view.proposition_legend.text()
        assert len(calls) == 1
        # Hidden views adopt only the latest small annotation snapshot and
        # upload its labels when shown, without recomputing the histogram.
        view.dock.hide()
        qtbot.waitUntil(lambda: not view._dock_visible)
        context[0] = catalog.reject(keys[1])
        view.refresh_proposition_labels()
        assert view._proposition_labels_pending
        view.dock.show()
        qtbot.waitUntil(lambda: not view._proposition_labels_pending)
        np.testing.assert_array_equal(view.text_visual._acc.color[10:16], np.ones((6, 4)))
        assert len(calls) == 1
        view.toggle_labels(False)
        assert view.text_visual._hidden
        view.toggle_labels(True)
        assert not view.text_visual._hidden
    finally:
        gui.close()


def test_bounded_catalog_updates_reviews_undo_and_cleanup(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path, propositions=_mapping([[0, 1], [1, 2]]))
    gui = controller.create_gui(default_views=['SampledCorrelogramView'], do_prompt_save=False)
    qtbot.addWidget(gui)
    supervisor = controller.supervisor
    view = gui.get_view('SampledCorrelogramView')
    keys = tuple(p.key for p in supervisor.merge_propositions.catalog.propositions)
    try:
        with qtbot.waitExposed(gui):
            gui.show()
        supervisor._activate_merge_proposition(supervisor.merge_propositions_view, keys[0])
        supervisor.block()
        # The blue label group is independent of trace slot/reference identity.
        assert view._proposition_labels.annotations[1].color == proposition_label_color(0)
        assert supervisor.selection_color_indices[0] == 0
        assert supervisor.selection_color_indices[1] != 0
        before = dict(supervisor.selection_color_indices)
        supervisor._refresh_propositions()
        assert dict(supervisor.selection_color_indices) == before
        assert view._proposition_labels.annotations[1].text == '1*'
        supervisor.reject_merge_proposition()
        supervisor.block()
        assert [group.key for group in view._proposition_labels.groups] == [keys[1]]
        supervisor.undo()
        supervisor.block()
        assert view._proposition_labels.annotations[1].text == '1*'
        assert view._proposition_labels.groups[0].key == keys[0]
        supervisor.redo()
        supervisor.block()
        supervisor._reset_merge_proposition(supervisor.merge_propositions_view, keys[0])
        supervisor.block()
        assert view._proposition_labels.groups[0].key == keys[0]
        assert view._proposition_labels.annotations[1].text == '1*'

        supervisor._activate_merge_propositions(supervisor.merge_propositions_view, keys)
        supervisor.block()
        assert view._proposition_labels.annotations[2].color == proposition_label_color(1)
        supervisor._activate_merge_proposition(supervisor.merge_propositions_view, keys[0])
        supervisor.block()
        # Merged descendants never inherit pending source membership.
        target = supervisor.merge().added[0]
        supervisor.block()
        assert not view._proposition_labels.annotations[target].groups
        supervisor.undo()
        supervisor.block()
        assert view._proposition_labels.annotations[1].text == '1*'

        # Review-only refreshes use the small text projection even for unchanged IDs.
        def reject_count(*args, **kwargs):
            raise AssertionError('A label-only notification must not recompute CCGs.')

        monkeypatch.setattr(view, 'exclusion_correlograms', reject_count)
        supervisor._refresh_propositions()
        table = supervisor.merge_propositions_view
        table.filter("proposition == 'P1 · 0, 1'")
        assert table.visible_keys() == (keys[0],)
        assert 'P2: pending; hidden by proposition filter' in view.proposition_legend.toolTip()
        table.filter('')
        assert 'hidden by proposition filter' not in view.proposition_legend.toolTip()
        table.sort_by('proposition', 'desc')
        assert view._proposition_labels.groups[0].key == keys[0]
        # The adapter subscription is removed when its view closes.
        view.close()
        monkeypatch.setattr(view, 'refresh_proposition_labels', reject_count)
        emit('merge_propositions_changed', supervisor)
    finally:
        gui.close()
        controller.close()


def test_missing_propositions_keep_original_labels_and_no_legend(qtbot, tmp_path):
    gui = GUI(config_dir=tmp_path / 'config')
    qtbot.addWidget(gui)
    view = SampledCorrelogramView(correlograms=_provider, sample_rate=30000)
    try:
        view.attach(gui)
        view.on_select(cluster_ids=[0, 1])
        assert view.proposition_legend is None
        assert view.text_visual._acc.text == ['0', '0', '1', '1']
        np.testing.assert_array_equal(view.text_visual._acc.color, np.ones((4, 4)))
    finally:
        gui.close()
