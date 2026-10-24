"""Exact selected-unit comparisons and scoped GUI selection behavior."""

from itertools import product
from types import MappingProxyType

import numpy as np
import pytest
from phylib.utils import Bunch

from phy.apps.large_recording.stability import UnitStabilityView
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.utils.color import selected_cluster_color


@pytest.fixture
def summary_view(qtbot, tmp_path):
    arrays = {
        'edges': [0, 10, 20, 25],
        'exposure': [8.0, 0.0, 3.0],
        'counts': [[4, 0, 3], [12, 0, 6], [8, 0, 0]],
        'amplitude_sum': [[4.0, 0.0, 15.0], [30.0, 0.0, 8.0], [0.0, 0.0, 0.0]],
        'amplitude_count': [[2, 0, 3], [3, 0, 1], [0, 0, 0]],
        'depth_sum': [[40.0, 0.0, 150.0], [300.0, 0.0, 80.0], [0.0, 0.0, 0.0]],
        'depth_count': [[2, 0, 3], [3, 0, 1], [0, 0, 0]],
    }
    summary = {}
    for key, values in arrays.items():
        name = f'{key}.npy'
        np.save(tmp_path / name, np.asarray(values))
        summary[f'{key}_file'] = name
    membership = {1: [1], 2: [2], 3: [3], 4: [1, 2], 5: None}
    controller = Bunch(
        dir_path=tmp_path,
        model=Bunch(sample_rate=1.0, duration=25.0),
        evidence_windows=np.array([[0, 2], [22, 24]]),
        bounded_session=Bunch(
            unit_ids=[1, 2, 3],
            manifest={
                'summaries': summary,
                'bounded_counts': [1, 1, 0],
                'valid_intervals': [[0, 8], [20, 23]],
            },
        ),
        _source_units=lambda cluster: membership[cluster],
        get_spike_times=lambda cluster: pytest.fail('Bounded views must never read spike arrays'),
    )
    view = UnitStabilityView(controller)
    yield view
    view.close()


@pytest.mark.parametrize('selected', [[1, 2], [1, 2, 3]])
def test_exact_overlays_and_shared_exposure(summary_view, selected):
    view = summary_view
    view.on_select(selected)
    axes = list(view.canvas.iter_ax())
    assert len(axes[0].lines) == len(selected)
    assert [line.get_label() for line in axes[0].lines] == list(map(str, selected))
    expected = {1: [0.5, np.nan, 1.0], 2: [1.5, np.nan, 2.0], 3: [1.0, np.nan, 0.0]}
    for index, (cluster, line) in enumerate(zip(selected, axes[0].lines)):
        np.testing.assert_allclose(line.get_ydata(), expected[cluster], equal_nan=True)
        np.testing.assert_allclose(line.get_xdata(), np.array([5, 15, 22.5]) / 3600)
        np.testing.assert_allclose(line.get_color(), selected_cluster_color(index))
    assert '3' in view._footer.get_text()  # Unit without bounded evidence remains inspectable.
    assert len(axes[1].lines) == 2  # No fabricated all-NaN trend for unit 3.
    assert any(
        'Amplitude measurement unavailable for units: 3' in t.get_text() for t in axes[1].texts
    ) == (3 in selected)


def test_weighted_merged_trends_and_pending_do_not_hide_valid_units(summary_view):
    view = summary_view
    merged = view.get_data(4)
    np.testing.assert_array_equal(merged.counts, [16, 0, 9])
    np.testing.assert_allclose(merged.rates, [2.0, np.nan, 3.0], equal_nan=True)
    np.testing.assert_allclose(merged.trends['amplitude'], [6.8, np.nan, 5.75], equal_nan=True)
    np.testing.assert_allclose(merged.trends['depth'], [68.0, np.nan, 57.5], equal_nan=True)
    view.on_select([5, 4])
    ax = next(view.canvas.iter_ax())
    assert len(ax.lines) == 1 and ax.lines[0].get_label() == '4'
    assert any('Unit 5: Split evidence pending' in t.get_text() for t in ax.texts)


def test_lazy_metrics_selected_cache_and_all_toggle_layouts(summary_view, monkeypatch):
    view = summary_view
    loads = []
    original = view._array

    def load(name):
        loads.append(name)
        return original(name)

    monkeypatch.setattr(view, '_array', load)
    view.show_amplitude = view.show_depth = False
    view.on_select([1, 2])
    assert not any('amplitude' in name or 'depth' in name for name in loads)
    common = view._common()
    lookups = common.lookup
    view.toggle_amplitude(True)
    assert any('amplitude' in name for name in loads)
    assert not any('depth' in name for name in loads)
    loads.clear()
    view.toggle_amplitude(False)
    view.toggle_amplitude(True)
    assert not loads
    assert view._common() is common and view._common().lookup is lookups
    axes = list(view.canvas.iter_ax())
    callback_counts = {k: len(v) for k, v in view.canvas.canvas.callbacks.callbacks.items()}
    for _ in range(2):
        for combination in product((False, True), repeat=3):
            for metric, enabled in zip(view._metrics, combination):
                getattr(view, f'toggle_{metric}')(enabled)
            assert tuple(view.state[f'show_{metric}'] for metric in view._metrics) == combination
            assert list(view.canvas.iter_ax()) == axes
            assert {
                k: len(v) for k, v in view.canvas.canvas.callbacks.callbacks.items()
            } == callback_counts
            visible = [ax for ax in axes if ax.get_visible()]
            if any(combination):
                assert len(visible) == sum(combination)
                assert visible[-1].get_xlabel() == 'Elapsed acquisition clock (hours)'
                assert all(not ax.get_xlabel() for ax in visible[:-1])
                assert visible[0].get_position().y1 == pytest.approx(
                    0.91 if len(visible) == 1 else 0.15 + 0.76 - 0.16 * 0.76 / len(visible)
                )
            else:
                assert any('No metrics enabled' in t.get_text() for t in visible[0].texts)
    assert view.state.show_depth
    view.on_select([3])
    assert set(view._selected_summaries) == {3}


def test_selection_limit_palette_and_clear_lifecycle(summary_view):
    view = summary_view
    # Repeat source rows under distinct cluster IDs to test the display bound.
    view.controller._source_units = lambda cluster: [1]
    selected = list(range(20, 30))
    sender = Bunch(
        selection_color_indices=MappingProxyType({c: 9 - i for i, c in enumerate(selected)})
    )
    view.on_select_threaded(sender, selected)
    assert view.cluster_ids == tuple(selected)
    assert selected == list(range(20, 30))
    assert len(view._selected_summaries) == 8
    ax = next(view.canvas.iter_ax())
    assert len(ax.lines) == 8
    assert view.status == 'Showing first 8 of 10 selected units.'
    for i, line in enumerate(ax.lines):
        np.testing.assert_allclose(line.get_color(), selected_cluster_color(9 - i))
    view.auto_update = False
    view.on_select_threaded(sender, [])
    assert view.cluster_ids == tuple(selected)
    view.auto_update = True
    view.defer_hidden_updates = True
    view._dock_visible = False
    view.on_select_threaded(sender, selected)
    view._dock_visible = True
    view._flush_pending_selection()
    np.testing.assert_allclose(ax.lines[0].get_color(), selected_cluster_color(9))
    view._dock_visible = False
    view.on_select_threaded(sender, [])
    assert view.cluster_ids == () and view._pending_selection == ([], {})
    assert len(ax.lines) == 8
    view._dock_visible = True
    view._flush_pending_selection()
    assert not ax.lines
    view.on_select([1])
    view._closed = True
    view.on_select_threaded(sender, [])
    assert view.cluster_ids == (1,)
    view._closed = False


def test_ordinary_continuous_assumption_and_invalidation(qtbot):
    times = [np.array([0.0, 1.0, 9.0])]
    calls = []

    def get_times(cluster):
        calls.append(cluster)
        return times[0]

    controller = Bunch(model=Bunch(duration=10.0), get_spike_times=get_times)
    view = UnitStabilityView(controller)
    try:
        view.on_select([7, 8])
        data = view.get_data(7)
        assert data.counts.sum() == 3 and not data.trends
        assert 'continuous acquisition assumed' in data.provenance
        np.testing.assert_allclose(data.exposure, 0.1)
        assert calls == [7, 8]
        times[0] = np.array([2.0])
        view._invalidate()
        assert view.get_data(7).counts.sum() == 1
    finally:
        view.close()


def test_gui_initial_authoritative_selection_actions_and_curation(qtbot, tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        supervisor.select([0])
        supervisor.block()
        supervisor.similarity_view.select([1])
        supervisor.block()
        view = UnitStabilityView(controller)
        view.attach(gui)
        assert view.cluster_ids == tuple(supervisor.selected) == (0, 1)
        for metric in view._metrics:
            assert view.actions.get(f'toggle_{metric}').isChecked()
        view.actions.get('toggle_amplitude').trigger()
        assert not view.show_amplitude
        assert not view.actions.get('toggle_amplitude').isChecked()
        view.toggle_depth(False)
        assert not view.actions.get('toggle_depth').isChecked()
        state = view.state.copy()
        supervisor.toggle_merge_mode()
        supervisor.block()
        supervisor.similarity_view.select([2, 1])
        supervisor.block()
        other = UnitStabilityView(controller)
        other.attach(gui)
        assert other.cluster_ids == tuple(supervisor.selected)
        assert other.cluster_ids
        supervisor.add_to_merge((2,), insertion=1)
        supervisor.block()
        assert other.cluster_ids == tuple(supervisor.selected)
        assert other.cluster_ids[1] == 2
        supervisor.similarity_view.sort_by('id', 'desc')
        supervisor.block()
        assert other.cluster_ids == tuple(supervisor.selected)
        for index, line in enumerate(next(other.canvas.iter_ax()).lines):
            cluster = int(line.get_label())
            np.testing.assert_allclose(
                line.get_color(),
                selected_cluster_color(supervisor.selection_color_indices[cluster]),
            )
        # Save and restore through GUI view state, including QAction checks.
        other.close()
        view.close()
        restored = UnitStabilityView(controller)
        restored.attach(gui)
        assert restored.show_firing_rate
        assert not restored.show_amplitude and not restored.show_depth
        assert not restored.actions.get('toggle_amplitude').isChecked()
        assert not restored.actions.get('toggle_depth').isChecked()
        view = restored
        view.set_state(state)
        assert not view.show_amplitude and not view.show_depth
        supervisor.toggle_merge_mode()
        supervisor.block()
        supervisor.merge([0, 1])
        supervisor.block()
        assert view.get_data(100).counts.sum() == 7
        supervisor.undo()
        supervisor.block()
        assert view.get_data(0).counts.sum() == 4
        supervisor.redo()
        supervisor.block()
        assert view.get_data(100).counts.sum() == 7
        supervisor.undo()
        supervisor.block()
        up = supervisor.split_source_amplitude(0, 2)
        supervisor.block()
        children = up.added
        for cluster in children:
            assert view.get_data(cluster).pending
        view.on_select([children[0], 1])
        assert [line.get_label() for line in next(view.canvas.iter_ax()).lines] == ['1']
        supervisor.undo()
        supervisor.block()
        assert view.get_data(0).counts.sum() == 4
        supervisor.redo()
        supervisor.block()
        assert view.get_data(children[0]).pending
        view.close()
        assert not view._summary_arrays and not view._selected_summaries
    finally:
        gui.close()
        controller.close()


def test_500_hour_500_bin_comparison_stays_bounded(summary_view):
    view = summary_view
    bins = 500
    view._common_data = Bunch(
        edges=np.linspace(0, 500 * 3600, bins + 1),
        exposure=np.full(bins, 3600.0),
        windows=np.array([[0, 30], [499 * 3600, 499 * 3600 + 30]]),
        intervals=[[0, 500 * 3600]],
        footer='',
        lookup={1: 0, 2: 1, 3: 2},
        summary={'counts_file': 'virtual_counts'},
    )
    view._summary_arrays['virtual_counts'] = np.ones((3, bins), dtype=np.int64)
    view.controller._source_units = lambda cluster: [1]
    view.on_select(list(range(8)))
    assert len(view._selected_summaries) == 8
    assert all(len(line.get_xdata()) == bins for line in next(view.canvas.iter_ax()).lines)


def test_summary_aggregation_reads_one_row_at_a_time(summary_view):
    view = summary_view
    reads = []

    class RowGrid:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, row):
            assert isinstance(row, (int, np.integer)), 'Never materialize multiple unit rows'
            reads.append(row)
            return self.values[row]

    for key in ('counts', 'amplitude_sum', 'amplitude_count', 'depth_sum', 'depth_count'):
        name = f'{key}.npy'
        view._summary_arrays[name] = RowGrid(np.load(view.controller.dir_path / name))
    data = view.get_data(4)
    np.testing.assert_array_equal(data.counts, [16, 0, 9])
    assert reads == [0, 1] + [0, 0, 1, 1] * 2


def test_ordinary_gui_cluster_event_refreshes_cached_counts(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    times = [np.array([0.0, 1.0])]
    ordinary = Bunch(
        supervisor=controller.supervisor,
        model=Bunch(duration=20.0),
        get_spike_times=lambda cluster: times[0],
    )
    view = UnitStabilityView(ordinary)
    try:
        controller.supervisor.select([0])
        controller.supervisor.block()
        view.attach(gui)
        assert view.get_data(0).counts.sum() == 2
        times[0] = np.array([0.0, 1.0, 2.0, 3.0])
        controller.supervisor.label('group', 'noise', cluster_ids=[0])
        controller.supervisor.block()
        ax = next(view.canvas.iter_ax())
        assert np.nansum(ax.lines[0].get_ydata() * 0.2) == pytest.approx(4)
        assert view.get_data(0).counts.sum() == 4
    finally:
        view.close()
        gui.close()
        controller.close()
