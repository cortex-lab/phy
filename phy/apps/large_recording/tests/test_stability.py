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
    view.display_limit = 8
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
    assert 'Showing first 8 of 10 selected units' in view.status
    assert 'does not change the selection or merge operation' in view.status
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


def test_gui_initial_authoritative_selection_actions_and_curation(qtbot, tmp_path, monkeypatch):
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
        # Exercise the real dock camera callback with the Matplotlib Qt widget.
        from phy.apps.large_recording import stability

        default_path = stability.screenshot_default_path
        paths = []

        def screenshot_path(widget, dir=None):
            path = default_path(widget, dir=tmp_path)
            paths.append(path)
            return path

        monkeypatch.setattr(stability, 'screenshot_default_path', screenshot_path)
        view.dock._dock_widgets['screenshot'].click()
        assert paths[-1].read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert paths[-1].name.endswith('_UnitStabilityView.png')
        assert view.screenshot(tmp_path) == paths[-1]
        view.actions.run('set_rate_line_width', 0.75)
        view.actions.run('set_display_limit', 64)
        view.actions.run('set_display_bin_seconds', 20)
        assert 'display_bin_seconds' in view.local_state_attrs
        assert 'display_bin_seconds' not in view.state_attrs
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
        assert restored.rate_line_width == 0.75
        assert restored.display_limit == 64
        assert restored.display_bin_seconds == 20
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
        assert view.get_display_data(100).counts.sum() == 7
        supervisor.undo()
        supervisor.block()
        assert view.get_data(0).counts.sum() == 4
        assert view.get_display_data(0).counts.sum() == 4
        supervisor.redo()
        supervisor.block()
        assert view.get_data(100).counts.sum() == 7
        assert view.get_display_data(100).counts.sum() == 7
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
        assert view.get_display_data(0).counts.sum() == 4
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
        assert view.get_display_data(0).counts.sum() == 4
    finally:
        view.close()
        gui.close()
        controller.close()


def test_display_limit_all_and_large_values_preserve_selection(summary_view):
    view = summary_view
    assert view.display_limit == 32
    view.controller._source_units = lambda cluster: [1]
    selected = list(range(40))
    view.on_select(selected)
    assert view.cluster_ids == tuple(selected)
    assert len(view.displayed_ids) == len(next(view.canvas.iter_ax()).lines) == 32
    assert 'merge operation' in view.status
    for limit in (0, 64, 128):
        view.set_display_limit(limit)
        assert view.displayed_ids == tuple(selected)
        assert len(next(view.canvas.iter_ax()).lines) == 40
        assert not view.status
        assert view.cluster_ids == tuple(selected)
    for ax in view.canvas.iter_ax():
        assert ax.get_legend() is None
    assert all(line.get_linewidth() == 0.6 for line in next(view.canvas.iter_ax()).lines)
    view.set_rate_line_width(0.7)
    assert all(line.get_linewidth() == 0.7 for line in next(view.canvas.iter_ax()).lines)


@pytest.mark.parametrize('value', [None, 'bad', np.nan, np.inf, -1, 0])
def test_invalid_line_width_state_and_action(summary_view, value):
    view = summary_view
    with pytest.raises((ValueError, TypeError)):
        view.set_rate_line_width(value)
    view.set_state({'rate_line_width': value})
    assert view.rate_line_width == 0.6


@pytest.mark.parametrize('value', [None, 'bad', np.nan, np.inf, -1, 1.5])
def test_invalid_display_limit_state_and_action(summary_view, value):
    view = summary_view
    with pytest.raises((ValueError, TypeError)):
        view.set_display_limit(value)
    view.set_state({'display_limit': value})
    assert view.display_limit == 32


@pytest.mark.parametrize('value', [None, 'bad', np.nan, np.inf, -1, 5, 15])
def test_invalid_display_bins_state_and_action(summary_view, value):
    view = summary_view
    with pytest.raises((ValueError, TypeError)):
        view.set_display_bin_seconds(value)
    view.set_state({'display_bin_seconds': value})
    assert view.display_bin_seconds == 0


@pytest.mark.parametrize(
    'seconds, expected_edges, counts, exposure, amplitude',
    [
        (0, [0, 10, 20, 25], [16, 0, 9], [8, 0, 3], [6.8, np.nan, 5.75]),
        (20, [0, 20, 25], [16, 9], [8, 3], [6.8, 5.75]),
        (30, [0, 25], [25], [11], [57 / 9]),
        (100, [0, 25], [25], [11], [57 / 9]),
    ],
)
def test_display_rebin_exact_merged_primitives(
    summary_view, seconds, expected_edges, counts, exposure, amplitude
):
    view = summary_view
    view.set_display_bin_seconds(seconds)
    data = view.get_display_data(4)
    np.testing.assert_array_equal(data.edges, expected_edges)
    np.testing.assert_array_equal(data.counts, counts)
    np.testing.assert_array_equal(data.exposure, exposure)
    np.testing.assert_allclose(data.trends['amplitude'], amplitude, equal_nan=True)
    expected_rates = np.full(len(exposure), np.nan)
    np.divide(counts, exposure, out=expected_rates, where=np.asarray(exposure) > 0)
    np.testing.assert_allclose(data.rates, expected_rates, equal_nan=True)
    view.on_select([4])
    axes = list(view.canvas.iter_ax())
    for ax in axes:
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(), (data.edges[:-1] + data.edges[1:]) / 7200
        )
    assert all(ax.get_legend() is None for ax in axes)


def test_300_second_native_bins_and_weighted_trends(summary_view, monkeypatch):
    view = summary_view
    common = view._common()
    common.edges = np.array([0, 300, 600, 900, 1000])
    common.exposure = np.array([300.0, 0.0, 100.0, 40.0])
    common.intervals = [[0, 300], [600, 700], [900, 940]]
    # Means 1 and 9 have different finite counts: their aggregate is 7, not 5.
    view._summary_arrays.update(
        {
            'counts.npy': np.array([[30, 0, 20, 4]]),
            'amplitude_sum.npy': np.array([[1.0, 0.0, 27.0, 20.0]]),
            'amplitude_count.npy': np.array([[1, 0, 3, 2]]),
            'depth_sum.npy': np.array([[10.0, 0.0, 270.0, 200.0]]),
            'depth_count.npy': np.array([[1, 0, 3, 2]]),
        }
    )
    source = view.get_data(1)
    assert view._display_factor() == 1
    np.testing.assert_allclose(source.rates, [0.1, np.nan, 0.2, 0.1], equal_nan=True)
    monkeypatch.setattr(view, '_array', lambda name: pytest.fail('Cached rows need no I/O'))
    for seconds, edges, counts, exposures, means in (
        (600, [0, 600, 1000], [30, 24], [300, 140], [1, 47 / 5]),
        (900, [0, 900, 1000], [50, 4], [400, 40], [7, 10]),
        (1800, [0, 1000], [54], [440], [8]),
    ):
        view.set_display_bin_seconds(seconds)
        data = view.get_display_data(1)
        np.testing.assert_array_equal(data.edges, edges)
        np.testing.assert_array_equal(data.counts, counts)
        np.testing.assert_array_equal(data.exposure, exposures)
        np.testing.assert_allclose(data.trends['amplitude'], means)
        assert view.get_display_data(1) is data
    view.set_display_bin_seconds(0)
    assert view.get_display_data(1) is source


def test_cache_budget_independent_of_limit_and_released(summary_view):
    view = summary_view
    view.controller._source_units = lambda cluster: [1]
    view.display_limit = 0
    budget = 2 * view._data_bytes(view.get_data(-1))
    view._invalidate()
    view.summary_cache_bytes = budget
    for cluster in range(20):
        data = view.get_display_data(cluster)
        assert data.counts.sum() == 7
        assert sum(map(view._data_bytes, view._selected_summaries.values())) <= budget
    assert 19 in view._selected_summaries and 0 not in view._selected_summaries
    view.summary_cache_bytes = 0
    assert view.get_data(30).counts.sum() == 7
    assert not view._selected_summaries
    view._invalidate()
    assert not view._selected_summaries


def test_6156_bins_many_units_cached_operation_bounds(summary_view, monkeypatch):
    view = summary_view
    bins = 6156
    common = view._common()
    common.edges = np.arange(bins + 1) * 300.0
    common.exposure = np.full(bins, 300.0)
    common.summary = {'counts_file': 'virtual_counts'}
    common.intervals = [[0, bins * 300]]
    rows = []

    class Grid:
        def __getitem__(self, row):
            rows.append(row)
            return np.ones(bins, dtype=np.int64)

    view._summary_arrays['virtual_counts'] = Grid()
    view.controller._source_units = lambda cluster: [1]
    view.show_amplitude = view.show_depth = False
    view.on_select(list(range(12)))
    assert len(rows) == 12
    assert len(next(view.canvas.iter_ax()).lines) == 12
    assert all(len(line.get_ydata()) == bins for line in next(view.canvas.iter_ax()).lines)
    monkeypatch.setattr(view, '_array', lambda name: pytest.fail('Must reuse cached source rows'))
    view.set_display_bin_seconds(900)
    view.plot()
    assert len(rows) == 12
    assert all(len(line.get_ydata()) == bins // 3 for line in next(view.canvas.iter_ax()).lines)
    view.set_display_bin_seconds(0)
    assert len(rows) == 12


def test_display_gap_and_unattached_cache_release(summary_view):
    view = summary_view
    common = view._common()
    common.exposure = np.array([0.0, 0.0, 3.0])
    view.set_display_bin_seconds(20)
    data = view.get_display_data(4)
    assert data.exposure[0] == 0
    assert np.isnan(data.rates[0])
    assert np.isnan(data.trends['amplitude'][0])
    assert np.isnan(data.trends['depth'][0])
    assert view._selected_summaries and view._summary_arrays
    view.close()
    assert not view._selected_summaries and not view._summary_arrays
    assert view._common_data is None


def test_pending_summaries_also_obey_cache_budget(summary_view):
    view = summary_view
    view.controller._source_units = lambda cluster: None
    pending = view.get_data(5)
    view.summary_cache_bytes = 2 * view._data_bytes(pending)
    for cluster in range(20):
        assert view.get_data(cluster).pending
    assert len(view._selected_summaries) <= 2
    assert (
        sum(map(view._data_bytes, view._selected_summaries.values())) <= view.summary_cache_bytes
    )


@pytest.mark.parametrize(
    'edges',
    [
        [0, 10, 25, 30],
        [0, 10, 20, 35],
        [0, 10, 20, 20],
        [0, 10, 20, np.inf],
    ],
)
def test_seconds_coarsening_rejects_nonuniform_or_invalid_native_bins(summary_view, edges):
    view = summary_view
    view._common().edges = np.asarray(edges, dtype=float)
    assert view._display_factor(0) == 1
    with pytest.raises(ValueError, match='requires uniform native bins'):
        view.set_display_bin_seconds(20)
    assert view.display_bin_seconds == 0
    view.set_state({'display_bin_seconds': 20})
    assert view.display_bin_seconds == 0


def test_seconds_coarsening_supports_short_final_native_bin(summary_view):
    view = summary_view
    # The fixture's native widths are 10, 10, 5 seconds.
    assert view._display_factor(20) == 2
    view.set_display_bin_seconds(20)
    data = view.get_display_data(1)
    np.testing.assert_array_equal(data.edges, [0, 20, 25])
    np.testing.assert_array_equal(data.counts, [4, 3])
    np.testing.assert_array_equal(data.exposure, [8, 3])
