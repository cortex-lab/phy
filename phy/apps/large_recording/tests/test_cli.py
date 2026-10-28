"""Batch CLI coverage and window-policy regressions."""

import json

from click.testing import CliRunner

from phy.apps import phycli
from phy.apps.large_recording.tests.test_source import make_sorting


def test_prepare_cli_reports_waveform_coverage_and_window_policy(tmp_path):
    params = make_sorting(tmp_path / 'source')
    output = tmp_path / 'bounded'
    result = CliRunner().invoke(
        phycli,
        [
            'prepare-large',
            str(params),
            str(output),
            '--window-policy',
            'generated',
            '--window-seconds',
            '20',
            '--stratum-seconds',
            '20',
        ],
    )
    assert result.exit_code == 0, result.output
    assert '5 retained / 5 source; 0 omitted' in result.output
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
    assert 'building summary store: 100%' in result.output
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
    result = runner.invoke(phycli, ['build-amplitude-index', str(bounded), str(ranges)])
    assert result.exit_code == 0, result.output
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
