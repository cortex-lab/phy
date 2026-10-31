"""Exact asynchronous summaries integrated with real bounded tables and plots."""

import json

import numpy as np
import pytest
from phylib.utils import connect, unconnect

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.apps.template.gui import TemplateController


def _files(path):
    return {p.name: hash_file(p) for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize('indexed', [False, 'rows', 'amplitude'])
def test_split_refreshes_views_and_table_then_merge_reopen(
    qtbot, tmp_path, monkeypatch, indexed, caplog
):
    caplog.set_level('DEBUG', logger='phy.apps.large_recording.summary_service')
    params, controller = make_controller(tmp_path)
    source_files = _files(params.parent)
    package = controller.dir_path
    manifest_hash = hash_file(package / '_phy_curation_manifest.json')
    range_queries = []
    if indexed:
        from phy.apps.large_recording.live_summaries import LiveSummaryEngine
        from phy.apps.large_recording.summary_store import build_summary_store

        store = build_summary_store(package, tmp_path / 'summary-store', batch_size=2)
        if indexed == 'amplitude':
            from phy.apps.large_recording.range_build import build_amplitude_index
            from phy.apps.large_recording.range_index import AmplitudeRangeIndex
            from phy.apps.large_recording.summary_store import SummaryStore

            ranges = build_amplitude_index(package, store / 'amplitude-index')
            original_resolve = AmplitudeRangeIndex.resolve

            def observe_range(self, expressions, *args, **kwargs):
                result = original_resolve(self, expressions, *args, **kwargs)
                range_queries.append((self.path, set(expressions)))
                return result

            def no_rows(*args, **kwargs):
                raise AssertionError('Amplitude-index GUI used the row fallback')

            monkeypatch.setattr(AmplitudeRangeIndex, 'resolve', observe_range)
            monkeypatch.setattr(SummaryStore, 'batches', no_rows)

        def no_source(*args, **kwargs):
            raise AssertionError('Indexed GUI opened full source')

        monkeypatch.setattr(LiveSummaryEngine, '_open', no_source)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    service = controller._live_summaries
    supervisor = controller.supervisor
    view = gui.get_view('UnitStabilityView')
    try:
        service.suspend()
        update = supervisor.split_source_amplitude(0, 4)
        selected, remainder = update.added
        supervisor.select([selected, remainder, 1])
        supervisor.block()
        assert controller.get_full_count(selected) is None
        assert view.get_data(selected).pending
        np.testing.assert_array_equal(view.get_data(1).counts, [1, 1, 1, 0])
        supervisor.save()
        pointer = package / '.phy-curation/current.json'
        saved_pointer = pointer.read_bytes()
        generation = json.loads(saved_pointer)['generation']
        saved_files = _files(package / '.phy-curation/generations' / generation)
        service.resume()
        qtbot.waitUntil(lambda: controller.get_full_count(selected) == 2, timeout=10000)
        supervisor.block()
        if indexed == 'amplitude':
            assert range_queries
            assert any(
                'Full summary job: path=amplitude-index' in r.getMessage() for r in caplog.records
            )
            assert all(path == ranges for path, _ in range_queries)
            assert {selected, remainder} <= set().union(*(units for _, units in range_queries))
        np.testing.assert_array_equal(view.get_data(selected).counts, [2, 0, 0, 0])
        np.testing.assert_array_equal(view.get_data(remainder).counts, [0, 0, 1, 1])
        np.testing.assert_allclose(view.get_data(selected).rates, [0.4, 0, 0, 0])
        np.testing.assert_allclose(view.get_data(selected).trends['amplitude'][:1], [1.5])
        row = supervisor.cluster_view._model.row_by_id(selected)
        assert row['full_count'] == 2
        assert row['fr'] == 0.1
        assert not supervisor.is_dirty()
        assert pointer.read_bytes() == saved_pointer
        assert _files(package / '.phy-curation/generations' / generation) == saved_files
        merged = supervisor.merge([selected, 1]).added[0]
        qtbot.waitUntil(lambda: controller.get_full_count(merged) == 5, timeout=10000)
        np.testing.assert_array_equal(view.get_data(merged).counts, [3, 1, 1, 0])
        supervisor.undo()
        assert controller.get_full_count(selected) == 2
        supervisor.redo()
        assert controller.get_full_count(merged) == 5
        supervisor.save()
        replay = apply_curation(params, package, tmp_path / 'complete-replay', batch_size=2)
        prepared = prepare_large(
            replay / 'params.py',
            tmp_path / 'reprepared',
            window_seconds=20,
            stratum_seconds=20,
            max_spikes=100,
            bin_seconds=5,
            batch_size=2,
        )
        baseline = json.loads((prepared / '_phy_curation_manifest.json').read_text())
        for unit in (merged, remainder):
            data = view.get_data(unit)
            row_index = baseline['unit_ids'].index(unit)
            for field in (
                'counts',
                'amplitude_sum',
                'amplitude_count',
                'depth_sum',
                'depth_count',
            ):
                expected = np.load(prepared / baseline['summaries'][field + '_file'])[row_index]
                actual = (
                    data.counts
                    if field == 'counts'
                    else data.trend_sums[field[:-4]]
                    if field.endswith('_sum')
                    else data.trend_counts[field[:-6]]
                )
                if field.endswith('_sum'):
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                else:
                    np.testing.assert_array_equal(actual, expected)
        gui.close()
        # Recreate the GUI around the same controller and retained derived cache.
        gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
        assert controller.get_full_count(merged) == 5
        assert not controller._live_summaries._suspended
    finally:
        gui.close()
        controller.close()
    reopened = TemplateController(dir_path=package, config_dir=tmp_path / 'reopen', plugins=[])
    gui = reopened.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    try:
        assert reopened.get_full_count(merged) == 5
        np.testing.assert_array_equal(
            gui.get_view('UnitStabilityView').get_data(merged).counts, [3, 1, 1, 0]
        )
        assert hash_file(package / '_phy_curation_manifest.json') == manifest_hash
        assert _files(params.parent) == source_files
    finally:
        gui.close()
        reopened.close()


def test_summary_worker_ignores_rolled_back_split(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    supervisor = controller.supervisor

    @connect(event='bounded_state_changed', sender=supervisor)
    def reject(sender):
        raise RuntimeError('observer rejects edit')

    try:
        with pytest.raises(RuntimeError, match='observer rejects'):
            supervisor.split_source_amplitude(0, 4)
        unconnect(reject)
        controller._live_summaries.synchronize()
        qtbot.waitUntil(lambda: not controller._live_summaries._timer.isActive())
        assert not controller.bounded_session.events
        assert controller._live_summaries._running is None
        assert controller.get_full_count(0) == 4
    finally:
        unconnect(reject)
        gui.close()
        controller.close()


def test_hidden_view_keeps_latest_selection_when_summary_arrives(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    view = gui.get_view('UnitStabilityView')
    service = controller._live_summaries
    try:
        service.suspend()
        controller.supervisor.select([0])
        controller.supervisor.block()
        view.on_select([0])
        view.defer_hidden_updates = True
        view._dock_visible = False
        children = controller.supervisor.split_source_amplitude(0, 4).added
        # A deferred selection must survive asynchronous results for prior IDs.
        view._pending_selection = (list(children), {})
        service.resume()
        qtbot.waitUntil(lambda: controller.get_full_count(children[0]) == 2)
        assert view._pending_selection[0] == children
        view._dock_visible = True
        view._flush_pending_selection()
        assert view.cluster_ids == tuple(children)
        assert all(not view.get_data(c).pending for c in children)
    finally:
        gui.close()
        controller.close()


def test_relocated_source_choice_is_local_and_survives_reopen(qtbot, tmp_path):
    params, controller = make_controller(tmp_path)
    package = controller.dir_path
    original_manifest = (package / '_phy_curation_manifest.json').read_bytes()
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    relocated = tmp_path / 'relocated-source'
    params.parent.rename(relocated)
    try:
        gui.file_actions.run('locate_summary_source', str(relocated))
        assert controller._live_summaries._source_path == relocated
    finally:
        gui.close()
        controller.close()
    reopened = TemplateController(dir_path=package, config_dir=tmp_path / 'config', plugins=[])
    gui = reopened.create_gui(default_views=[], do_prompt_save=False)
    try:
        assert reopened._live_summaries._source_path == relocated
        assert (package / '_phy_curation_manifest.json').read_bytes() == original_manifest
    finally:
        gui.close()
        reopened.close()
