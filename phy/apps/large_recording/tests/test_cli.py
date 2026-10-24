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
