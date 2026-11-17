"""Batch CLI coverage and window-policy regressions."""

import json
import signal
from concurrent.futures import CancelledError
from io import StringIO

import pytest
from click.testing import CliRunner

from phy.apps import phycli
from phy.apps.large_recording.progress import ProgressReporter
from phy.apps.large_recording.tests.test_source import make_sorting


def test_progress_reporter_throttles_and_reports_counts_rate_and_eta():
    now = [0.0]
    output = StringIO()
    progress = ProgressReporter(stream=output, interval=2, clock=lambda: now[0])

    progress('building full summaries', 0, 100)
    now[0] = 1
    progress('building full summaries', 50, 100)
    assert output.getvalue().count('\n') == 1
    now[0] = 2
    progress('building full summaries', 50, 100)
    progress('building full summaries', 100, 100)

    lines = output.getvalue().splitlines()
    assert len(lines) == 3
    assert '0 / 100 spikes' in lines[0]
    assert '50 / 100 spikes (50.0%)' in lines[1]
    assert '25 spikes/s' in lines[1]
    assert 'elapsed 2s' in lines[1]
    assert 'ETA 2s' in lines[1]
    assert '100 / 100 spikes (100.0%)' in lines[2]


def test_progress_reporter_handles_unknown_totals_and_stage_changes():
    output = StringIO()
    progress = ProgressReporter(stream=output, interval=60, clock=lambda: 0.0)
    progress('hashing source bytes', 0, 0)
    progress('copying source files', 12, None)
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('hashing source bytes: 0 bytes, elapsed 0s')
    assert lines[1].startswith('copying source files: 12 bytes, elapsed 0s')
    assert 'nan' not in output.getvalue().lower()


@pytest.mark.parametrize('done,total', [(-1, 10), (0, -1)])
def test_progress_reporter_rejects_negative_counts(done, total):
    progress = ProgressReporter(stream=StringIO())
    with pytest.raises(ValueError, match='nonnegative'):
        progress('building full summaries', done, total)


def test_progress_reporter_cancel_is_cooperative():
    progress = ProgressReporter(stream=StringIO())
    progress.cancel()
    with pytest.raises(CancelledError, match='cancelled'):
        progress('building full summaries', 1, 10)


def test_sigint_cancels_immediately_and_restores_handler():
    original = signal.getsignal(signal.SIGINT)
    progress = ProgressReporter(stream=StringIO())

    with pytest.raises(CancelledError, match='cancelled'), progress:
        assert signal.getsignal(signal.SIGINT) == progress._handle_sigint
        progress._handle_sigint(signal.SIGINT, None)

    assert signal.getsignal(signal.SIGINT) == original


def test_apply_curation_cli_uses_workers_and_keeps_progress_off_stdout(tmp_path, monkeypatch):
    from phy.apps.large_recording import replay

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'params.py').write_text('sample_rate = 1\n')
    bounded = tmp_path / 'bounded'
    bounded.mkdir()
    output = tmp_path / 'curated'
    calls = {}

    def fake_apply(source_params, bounded_dir, output_dir, **kwargs):
        calls.update(kwargs)
        kwargs['progress']('replaying assignments', 1, 4)
        kwargs['progress']('replaying assignments', 4, 4)
        return output

    monkeypatch.setattr(replay, 'apply_curation', fake_apply)
    result = CliRunner().invoke(
        phycli,
        [
            'apply-curation',
            str(source / 'params.py'),
            str(bounded),
            str(output),
            '--workers',
            '3',
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f'Published complete curated sorting: {output}\n'
    assert 'replaying assignments: 1 / 4 spikes' in result.stderr
    assert 'replaying assignments: 4 / 4 spikes' in result.stderr
    assert calls['workers'] == 3
    assert callable(calls['cancel'])


def test_cancelled_cli_operation_reports_error_on_stderr(tmp_path, monkeypatch):
    from phy.apps.large_recording import replay

    source = tmp_path / 'source'
    source.mkdir()
    params = source / 'params.py'
    params.write_text('sample_rate = 1\n')
    bounded = tmp_path / 'bounded'
    bounded.mkdir()

    def cancel_apply(*args, **kwargs):
        kwargs['progress'].cancel()
        kwargs['cancel']()

    monkeypatch.setattr(replay, 'apply_curation', cancel_apply)
    result = CliRunner().invoke(
        phycli,
        ['apply-curation', str(params), str(bounded), str(tmp_path / 'curated')],
    )

    assert result.exit_code == 1
    assert result.stdout == ''
    assert 'Operation cancelled.' in result.stderr


def test_apply_curation_cli_reports_bytes_and_spikes(tmp_path):
    from phy.apps.large_recording.curation import BoundedSession
    from phy.apps.large_recording.tests.test_live_summaries import package

    source, bounded = package(tmp_path)
    session = BoundedSession(bounded)
    session.record_merge([0, 1], 100, {'group': 'good'})
    session.save()
    session.close()
    output = tmp_path / 'curated'
    result = CliRunner().invoke(
        phycli,
        [
            'apply-curation',
            str(source / 'params.py'),
            str(bounded),
            str(output),
            '--workers',
            '2',
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == f'Published complete curated sorting: {output}\n'
    assert 'copying source files:' in result.stderr and 'bytes' in result.stderr
    assert 'replaying assignments:' in result.stderr and 'spikes' in result.stderr


def test_prepare_cli_reports_waveform_coverage_and_window_policy(tmp_path):
    params = make_sorting(tmp_path / 'source')
    output = tmp_path / 'bounded'
    result = CliRunner().invoke(
        phycli,
        [
            'prepare-large',
            str(params),
            str(output),
            '--workers',
            '2',
            '--window-policy',
            'generated',
            '--window-seconds',
            '20',
            '--stratum-seconds',
            '20',
        ],
    )
    assert result.exit_code == 0, result.output
    assert f'Prepared bounded evidence: {output}' in result.stdout
    assert '5 retained / 5 source; 0 omitted' in result.stdout
    assert 'writing bounded rows:' in result.stderr
    assert 'copying static arrays:' in result.stderr and 'bytes' in result.stderr
    manifest = json.loads((output / '_phy_curation_manifest.json').read_text())
    assert manifest['selection']['window_policy'] == 'generated'


def test_prepare_cli_explains_missing_producer_windows(tmp_path):
    params = make_sorting(tmp_path / 'source')
    output = tmp_path / 'bounded'
    result = CliRunner().invoke(
        phycli,
        ['prepare-large', str(params), str(output), '--window-policy', 'producer'],
    )
    assert result.exit_code == 1
    assert 'producer' in result.output.lower()
    assert not output.exists()


def test_build_and_attach_summary_store_commands(tmp_path):
    from phy.apps.large_recording.live_summaries import LiveSummaryEngine
    from phy.apps.large_recording.summary_store import SummaryStore
    from phy.apps.large_recording.tests.test_live_summaries import package

    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    runner = CliRunner()
    result = runner.invoke(
        phycli, ['build-summary-store', str(bounded), str(output), '--batch-size', '2']
    )
    assert result.exit_code == 0, result.output
    assert 'building summary store:' in result.stderr
    assert '(100.0%)' in result.stderr
    (bounded / '.phy' / 'live-summary-store.json').unlink()
    result = runner.invoke(phycli, ['attach-summary-store', str(bounded), str(output)])
    assert result.exit_code == 0, result.output
    with LiveSummaryEngine(bounded) as engine, SummaryStore(engine) as store:
        assert sum(len(data) for data in store.batches(0, 2)) == 4


def test_prepare_with_summary_store_and_recoverable_optional_failure(tmp_path):
    params = make_sorting(tmp_path / 'source')
    runner = CliRunner()
    bounded = tmp_path / 'bounded'
    output = tmp_path / 'store'
    result = runner.invoke(
        phycli, ['prepare-large', str(params), str(bounded), '--summary-store', str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.is_dir()
    failed_bounded = tmp_path / 'bounded-second'
    result = runner.invoke(
        phycli, ['prepare-large', str(params), str(failed_bounded), '--summary-store', str(output)]
    )
    assert result.exit_code == 1
    assert 'Bounded package is ready' in result.output
    assert 'build-summary-store to retry' in result.output
    assert (failed_bounded / '_phy_curation_manifest.json').is_file()


def test_upgrade_row_store_and_attach_index_commands(tmp_path):
    from phy.apps.large_recording.summary_store import summary_store_reference
    from phy.apps.large_recording.tests.test_live_summaries import package

    _, bounded = package(tmp_path)
    store, ranges = tmp_path / 'store', tmp_path / 'ranges'
    runner = CliRunner()
    result = runner.invoke(
        phycli, ['build-summary-store', str(bounded), str(store), '--rows-only']
    )
    assert result.exit_code == 0, result.output
    assert 'amplitude_index' not in json.loads(summary_store_reference(bounded))
    result = runner.invoke(
        phycli, ['build-amplitude-index', str(bounded), str(ranges), '--workers', '2']
    )
    assert result.exit_code == 0, result.output
    assert 'sorting amplitude ranges:' in result.stderr
    assert 'amplitude_index' in json.loads(summary_store_reference(bounded))
    runner.invoke(phycli, ['attach-summary-store', str(bounded), str(store)])
    result = runner.invoke(phycli, ['attach-amplitude-index', str(bounded), str(ranges)])
    assert result.exit_code == 0, result.output
    assert 'amplitude_index' in json.loads(summary_store_reference(bounded))


def test_combined_budget_fails_before_row_preparation(tmp_path):
    from phy.apps.large_recording.tests.test_live_summaries import package

    _, bounded = package(tmp_path)
    store = tmp_path / 'store'
    result = CliRunner().invoke(
        phycli, ['build-summary-store', str(bounded), str(store), '--max-bytes', '1000000']
    )
    assert result.exit_code == 1
    assert 'Combined store estimate' in result.output
    assert not store.exists()


def test_default_pipeline_preserves_newer_choice_between_stages(tmp_path, monkeypatch):
    from phy.apps.large_recording import summary_store
    from phy.apps.large_recording.range_build import build_amplitude_index
    from phy.apps.large_recording.tests.test_live_summaries import package

    _, bounded = package(tmp_path)
    original = summary_store.build_summary_store
    chosen = tmp_path / 'newer-index'

    def choose_between_stages(*args, **kwargs):
        result = original(*args, **kwargs)
        build_amplitude_index(bounded, chosen)
        return result

    monkeypatch.setattr(summary_store, 'build_summary_store', choose_between_stages)
    store = tmp_path / 'store'
    result = CliRunner().invoke(phycli, ['build-summary-store', str(bounded), str(store)])
    assert result.exit_code == 1
    assert 'Row snapshot ready' in result.output
    assert 'changed before' in result.output
    assert not (store / 'amplitude-index').exists()
    reference = json.loads(summary_store.summary_store_reference(bounded))
    assert reference['amplitude_index']['path'] == str(chosen)
