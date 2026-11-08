"""The same user commands on a full sorting and genuine bounded evidence."""

import gc
import shutil
from contextlib import contextmanager

import numpy as np
import pytest
from phylib.io.model import load_model
from phylib.utils.event import _EVENT

import tools.compare_curation_modes as comparison
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.tests.test_adapter import make_gui_sorting
from phy.apps.template.gui import TemplateController
from phy.gui.qt import QApplication, QCoreApplication, QEvent
from phy.plot.tests import mouse_drag
from phy.plot.transform import NDC, Range


def _make_source(path):
    params = make_gui_sorting(path)
    seconds = np.r_[np.arange(50), np.arange(60, 120)]
    units = np.array([0, 1, 2, 50, 51, 99], dtype=np.int32)
    clusters = np.tile(units, len(seconds))
    times = np.repeat(seconds * 30000, len(units)) + np.tile(
        np.arange(len(units)) * 300, len(seconds)
    )
    amplitudes = np.repeat(
        np.where(seconds % 2 == 0, 1, np.where(seconds % 4 == 1, 4, 7)), len(units)
    ).astype(np.float32)
    arrays = {
        'spike_times.npy': times.astype(np.int64),
        'spike_clusters.npy': clusters,
        'spike_templates.npy': (np.arange(len(times)) % 3).astype(np.int32),
        'amplitudes.npy': amplitudes,
        'spike_positions.npy': np.column_stack((np.zeros(len(times)), times / 30000)),
        'spike_depths.npy': (times / 30000).astype(np.float64),
        'pc_features.npy': np.arange(len(times) * 4, dtype=np.float32).reshape(-1, 2, 2),
        'whitening_mat.npy': np.eye(3, dtype=np.float32),
        'whitening_mat_inv.npy': np.eye(3, dtype=np.float32),
    }
    waveform_ids = np.arange(0, len(times), 7, dtype=np.int64)
    arrays.update(
        {
            '_phy_spikes_subset.spikes.npy': waveform_ids,
            '_phy_spikes_subset.channels.npy': np.tile([0, 1], (len(waveform_ids), 1)),
            '_phy_spikes_subset.waveforms.npy': np.arange(
                len(waveform_ids) * 8 * 2, dtype=np.float32
            ).reshape(-1, 8, 2),
        }
    )
    for name, values in arrays.items():
        np.save(path / name, values)
    params.write_text(
        params.read_text().replace('num_samples = 600000', 'num_samples = 3600000')
        + 'valid_intervals = [[0, 1500000], [1800000, 3600000]]\n'
    )
    (path / 'cluster_group.tsv').write_text(
        'cluster_id\tgroup\n0\tgood\n1\tmua\n2\tgood\n50\tgood\n51\tmua\n99\tnoise\n'
    )
    (path / 'cluster_snr.tsv').write_text(
        'cluster_id\tsnr\n0\t8\n1\t10\n2\t12\n50\t16\n51\t18\n99\t14\n'
    )
    (path / 'cluster_si_unit_ids.tsv').write_text(
        'cluster_id\tsi_unit_id\n0\t0\n1\t1\n2\t2\n50\t50\n51\t51\n99\t99\n'
    )
    (path / 'cluster_notes.tsv').write_text(
        'cluster_id\tnotes\n50\tproducer parent A\n51\tproducer parent B\n99\toriginal control\n'
    )
    return params


def _hashes(path):
    return {str(p.relative_to(path)): hash_file(p) for p in path.rglob('*') if p.is_file()}


@contextmanager
def _open(path, config, bounded):
    baseline = {id(item) for item in _EVENT._callbacks}
    model = None if bounded else load_model(path / 'params.py')
    controller = TemplateController(
        dir_path=path,
        model=model,
        config_dir=config,
        plugins=[],
        enable_threading=False,
    )
    controller.n_spikes_amplitudes = 5
    controller.n_spikes_amplitudes_total = 5
    controller.n_spikes_amplitudes_background = 8
    view_name = 'EvidenceAmplitudeView' if bounded else 'AmplitudeView'
    gui = None
    try:
        # Explicit view_count prevents saved/default layout from adding other
        # renderers: the controller, Supervisor and real amplitude provider stay native.
        gui = controller.create_gui(
            view_count={view_name: 1}, default_views=[view_name], do_prompt_save=False
        )
        gui.resize(900, 700)
        gui.show()
        QApplication.processEvents()
        view = gui.get_view(view_name)
        if not bounded:
            view.amplitudes_type = 'template'
        yield controller, gui, view
    finally:
        if gui is not None:
            gui.close()
            gui.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            QApplication.processEvents()
        controller.close()
        # Global callbacks may own controller/history cycles after closing a GUI.
        # Retain exactly the callbacks that preceded this isolated controller.
        _EVENT._callbacks = [item for item in _EVENT._callbacks if id(item) in baseline]
        del controller, model
        gc.collect()


def _snapshot(supervisor):
    return (
        supervisor.clustering.spike_clusters.copy(),
        {
            field: {
                int(unit): supervisor.cluster_meta.get(field, int(unit))
                for unit in supervisor.clustering.cluster_ids
            }
            for field in ('group', 'notes')
        },
    )


def _assert_snapshot(supervisor, expected):
    actual = _snapshot(supervisor)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def _split(qtbot, controller, gui, view, parent, threshold):
    supervisor = controller.supervisor
    supervisor.select([parent])
    supervisor.block()
    qtbot.wait(20)
    view.plot()
    ids = supervisor.clustering.spikes_per_cluster[parent].copy()
    amplitudes = np.asarray(controller.model.amplitudes[ids]).reshape(-1)
    assert np.all(np.isfinite(amplitudes))
    drawn = next(b for b in view.get_clusters_data() if b.cluster_id == parent)
    assert len(drawn.spike_ids) <= 5 < len(ids), 'The display must not be the split input.'
    selected = amplitudes < threshold
    assert selected.any() and (~selected).any(), 'Both children must have retained evidence.'
    width, height = view.canvas.get_size()
    point = (width // 3, height // 2)
    target = Range(view.data_bounds, NDC).apply([[0, threshold]])[0][1]
    normalized_y = 1 - 2 * point[1] / height
    pan = view.canvas.panzoom.pan
    view.canvas.panzoom.pan = (
        pan[0],
        normalized_y / view.canvas.panzoom._zoom_aspect()[1] - target,
    )
    # Move the viewport by representable steps if inverse-transform rounding
    # lands just above the requested boundary; no stored value is rewritten.
    draw_threshold = threshold
    for _ in range(16):
        if view._threshold_from_window_pos(point) <= threshold:
            break
        draw_threshold = np.nextafter(draw_threshold, -np.inf)
        below = Range(view.data_bounds, NDC).apply([[0, draw_threshold]])[0][1]
        view.canvas.panzoom.pan = (
            pan[0],
            normalized_y / view.canvas.panzoom._zoom_aspect()[1] - below,
        )
    mouse_drag(
        qtbot,
        view.canvas,
        (point[0] - 10, point[1]),
        point,
        button='right',
        modifiers=('Alt',),
    )
    assert view.split_threshold == pytest.approx(threshold, abs=1e-12, rel=0)
    np.testing.assert_array_equal(selected, amplitudes < view.split_threshold)
    action = supervisor.actions.get('split')
    assert action.isEnabled()
    action.trigger()
    supervisor.block()
    after = supervisor.clustering.spike_clusters[ids]
    low, rest = np.unique(after[selected]), np.unique(after[~selected])
    assert len(low) == len(rest) == 1 and low[0] != rest[0]
    return int(low[0]), int(rest[0])


def _run_ledger(qtbot, path, config, bounded, actions, aliases):
    aliases = dict(aliases)
    history, future = [], []
    opened = _open(path, config, bounded)
    controller, gui, view = opened.__enter__()
    try:
        for action in actions:
            op, *args = action
            supervisor = controller.supervisor
            if op == 'reopen':
                opened.__exit__(None, None, None)
                opened = _open(path, config, bounded)
                controller, gui, view = opened.__enter__()
                history.clear()
                future.clear()
                continue
            if op == 'save':
                supervisor.save()
                continue
            if op in ('undo', 'redo'):
                origin, destination = (history, future) if op == 'undo' else (future, history)
                current = _snapshot(supervisor)
                expected = origin.pop() if origin else current
                if expected is not current:
                    destination.append(current)
                getattr(supervisor, op)()
                supervisor.block()
                _assert_snapshot(supervisor, expected)
                continue
            history.append(_snapshot(supervisor))
            future.clear()
            if op == 'merge':
                result, *parents = args
                up = supervisor.merge([aliases[parent] for parent in parents])
                supervisor.block()
                assert len(up.added) == 1
                aliases[result] = int(up.added[0])
            elif op == 'split':
                parent, low, rest, threshold = args
                aliases[low], aliases[rest] = _split(
                    qtbot, controller, gui, view, aliases[parent], threshold
                )
            elif op == 'label':
                parent, field, value = args
                supervisor.label(field, value, cluster_ids=[aliases[parent]])
                supervisor.block()
                assert supervisor.cluster_meta.get(field, aliases[parent]) == value
            else:
                raise AssertionError(op)
        supervisor = controller.supervisor
        state = _snapshot(supervisor)
        supervisor.save()
    finally:
        opened.__exit__(None, None, None)
    return aliases, state


ROUND_ONE = [
    ('merge', 'merged', 'a', 'b'),
    ('undo',),
    ('redo',),
    ('split', 'merged', 'low', 'rest', 4.0),
    ('undo',),
    ('redo',),
    ('merge', 'final', 'low', 'c'),
    ('label', 'final', 'group', 'good'),
    ('label', 'rest', 'group', 'mua'),
    ('label', 'final', 'notes', 'reviewed merged child'),
    ('label', 'rest', 'notes', 'reviewed remainder'),
    ('label', 'rest', 'group', 'noise'),
    ('undo',),
    ('redo',),
    ('undo',),
    ('label', 'rest', 'notes', 'branch kept'),
    ('redo',),  # A new edit discarded the noise redo branch.
    ('label', 'control', 'group', None),
    ('label', 'control', 'notes', 'cleared group on unchanged membership'),
    ('save',),
    ('reopen',),
    ('save',),
]

ROUND_TWO = [
    ('merge', 'reunited', 'final', 'rest'),
    ('split', 'reunited', 'second_low', 'second_rest', 4.0),
    ('label', 'second_low', 'group', 'good'),
    ('label', 'second_rest', 'group', 'mua'),
    ('label', 'second_low', 'notes', 'reviewed round two low'),
    # Seed an explicit note before editing it: ordinary inherited parent notes
    # and bounded new-membership clearing intentionally have different baselines.
    ('label', 'second_rest', 'notes', 'round two draft'),
    ('label', 'second_rest', 'notes', 'reviewed round two remainder'),
    ('undo',),
    ('redo',),
    ('save',),
    ('reopen',),
    ('save',),
]


EXPECTED_ONE = {
    'final': {'group': 'good', 'notes': 'reviewed merged child'},
    'rest': {'group': 'mua', 'notes': 'branch kept'},
    'control': {'group': None, 'notes': 'cleared group on unchanged membership'},
}
EXPECTED_TWO = {
    'second_low': {'group': 'good', 'notes': 'reviewed round two low'},
    'second_rest': {'group': 'mua', 'notes': 'reviewed round two remainder'},
    'control': {'group': None, 'notes': 'cleared group on unchanged membership'},
}
NO_EDITS = [('save',), ('reopen',), ('save',)]
UNREVIEWED = [('merge', 'unreviewed', 'review_a', 'review_b'), *NO_EDITS]


@pytest.mark.parametrize('window,budget,seed', [(2, 32, 0), (3, 48, 11), (4, 48, 23)])
def test_same_controller_actions_match_full_replay(qtbot, tmp_path, capsys, window, budget, seed):
    params = _make_source(tmp_path / 'source')
    source_hashes = _hashes(params.parent)
    ordinary = tmp_path / 'ordinary'
    shutil.copytree(params.parent, ordinary)
    ordinary_aliases = bounded_aliases = {
        'a': 0,
        'b': 1,
        'c': 2,
        'review_a': 50,
        'review_b': 51,
        'control': 99,
    }
    source_params = params
    for round_index, (actions, expected_manual) in enumerate(
        zip(
            (ROUND_ONE, ROUND_TWO, NO_EDITS, UNREVIEWED),
            (EXPECTED_ONE, EXPECTED_TWO, EXPECTED_TWO, EXPECTED_TWO),
            strict=True,
        )
    ):
        round_source_hashes = _hashes(source_params.parent)
        package = prepare_large(
            source_params,
            tmp_path / f'bounded-{round_index}',
            seed=seed + round_index,
            max_spikes=budget,
            window_seconds=window,
            stratum_seconds=30,
            bin_seconds=10,
            batch_size=17,
        )
        source_labels = np.load(source_params.parent / 'spike_clusters.npy').reshape(-1)
        kept = np.load(package / '_phy_spike_global_ids.npy')
        assert 0 < len(kept) < len(source_labels)
        for unit in np.unique(source_labels):
            assert np.any(source_labels[kept] == unit)
            assert np.count_nonzero(source_labels[kept] == unit) < np.count_nonzero(
                source_labels == unit
            ), 'Every active unit must contain unseen source spikes.'
        ordinary_aliases, ordinary_state = _run_ledger(
            qtbot,
            ordinary,
            tmp_path / 'config-ordinary',
            False,
            actions,
            ordinary_aliases,
        )
        bounded_aliases, bounded_state = _run_ledger(
            qtbot,
            package,
            tmp_path / 'config-bounded',
            True,
            actions,
            bounded_aliases,
        )
        for aliases, state in (
            (ordinary_aliases, ordinary_state),
            (bounded_aliases, bounded_state),
        ):
            for alias, fields in expected_manual.items():
                for field, value in fields.items():
                    assert state[1][field][aliases[alias]] == value
        expected_labels = {
            bounded_aliases[alias]: fields for alias, fields in expected_manual.items()
        }
        output = apply_curation(
            source_params, package, tmp_path / f'curated-{round_index}', batch_size=17
        )
        report = comparison.compare_outputs(
            ordinary,
            output,
            fields=('group', 'notes'),
            chunk_size=13,
            source=params.parent,
            expected_labels=expected_labels,
            bounded_dir=package,
        )
        assert report['ok'], report.get('error', report)
        assert report['file_contract']['ok'], report
        for difference in report['permitted_membership_policy_differences']:
            if difference['field'] in ('group', 'notes'):
                assert round_index == 3, difference
                assert difference['right_id'] == bounded_aliases['unreviewed'], difference
            else:
                assert difference['field'] in ('snr', 'si_unit_id'), difference
            assert difference['right'] == '', difference
        assert report['rows'] == len(source_labels)
        if round_index == 2:
            no_op_files = [p.name for p in source_params.parent.glob('*.npy')] + [
                'cluster_group.tsv',
                'cluster_notes.tsv',
            ]
            for name in no_op_files:
                assert hash_file(source_params.parent / name) == hash_file(output / name), name
        assert _hashes(source_params.parent) == round_source_hashes
        assert _hashes(params.parent) == source_hashes
        source_params = output / 'params.py'
    captured = capsys.readouterr()
    assert 'Traceback (most recent call last)' not in captured.err, captured.err
