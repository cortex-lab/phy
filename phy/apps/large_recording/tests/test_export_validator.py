"""Focused contract and report tests for the AIND export validator."""

import json

import tools.validate_aind_export as validator
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.tests.test_source import make_sorting
from tools.validate_aind_export import main, validate_export


def test_validator_reports_engineering_evidence_and_input_hashes(tmp_path):
    params = make_sorting(tmp_path / 'source')
    params.write_text(
        params.read_text()
        + "time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': 1.0}\n"
        + 'valid_intervals = [[0, 20]]\n'
    )
    report = validate_export(
        params,
        phy_revision='phy-test-revision',
        exporter_revision='exporter-test-revision',
        batch_size=4,
    )

    assert report['status'] == 'passed'
    assert report['failures'] == []
    assert report['engineering_validation_only'] is True
    assert report['acquisition_truth_established'] is False
    assert report['source']['num_spikes'] == 12
    assert report['source']['num_samples'] == 20
    assert report['source']['exposure']['valid_samples'] == 20
    assert report['source']['input_files']['spike_times.npy']['sha256']
    assert report['phy_revision'] == 'phy-test-revision'
    assert report['exporter_revision'] == 'exporter-test-revision'
    assert report['timing_seconds']['total'] >= 0


def test_validator_preparation_report_matches_manifest_and_preserves_source(tmp_path):
    params = make_sorting(tmp_path / 'source')
    params.write_text(
        params.read_text().replace('num_samples = 20', 'num_samples = 120')
        + "time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': 1.0}\n"
        + 'valid_intervals = [[0, 120]]\n'
    )
    source_path = params.parent
    before_hashes = {
        path.name: hash_file(path) for path in source_path.iterdir() if path.is_file()
    }
    output_dir = tmp_path / 'bounded'

    report = validate_export(
        params,
        phy_revision='phy-test-revision',
        exporter_revision='exporter-test-revision',
        output_dir=output_dir,
        batch_size=4,
    )

    assert report['status'] == 'passed', report['failures']
    output = report['prepared_output']
    manifest = json.loads((output_dir / '_phy_curation_manifest.json').read_text())
    disk_bytes = sum(path.stat().st_size for path in output_dir.rglob('*') if path.is_file())
    assert output['num_spikes'] == manifest['bounded_num_spikes']
    assert output['num_units'] == len(manifest['unit_ids'])
    assert output['waveform_coverage'] == manifest['waveform_coverage']
    assert output['budgets'] == manifest['budgets']
    assert output['output_bytes'] == disk_bytes
    assert report['source']['input_files'] == {
        name: {'bytes': (source_path / name).stat().st_size, 'sha256': digest}
        for name, digest in before_hashes.items()
    }
    assert {
        path.name: hash_file(path) for path in source_path.iterdir() if path.is_file()
    } == before_hashes


def test_validator_reports_source_rejection_as_json_failure(tmp_path):
    params = make_sorting(tmp_path / 'source')
    (params.parent / 'timestamps.json').write_text(json.dumps({'frames': []}))

    report = validate_export(
        params,
        phy_revision='phy-test-revision',
        exporter_revision='exporter-test-revision',
    )

    assert report['status'] == 'failed'
    assert len(report['failures']) == 1
    assert 'explicit time_base' in report['failures'][0]['message']
    assert report['acquisition_truth_established'] is False


def test_validator_requires_explicit_aind_time_and_exposure_contract(tmp_path):
    params = make_sorting(tmp_path / 'source')

    report = validate_export(
        params,
        phy_revision='phy-test-revision',
        exporter_revision='exporter-test-revision',
    )

    assert report['status'] == 'failed'
    assert (
        report['failures'][0]['message'] == 'AIND export requires an explicit time_base contract.'
    )


def test_validator_rejects_nested_preparation_destination(tmp_path):
    params = make_sorting(tmp_path / 'source')
    params.write_text(
        params.read_text()
        + "time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': 1.0}\n"
        + 'valid_intervals = [[0, 20]]\n'
    )

    report = validate_export(
        params,
        phy_revision='phy-test-revision',
        exporter_revision='exporter-test-revision',
        output_dir=params.parent / 'derived',
    )

    assert report['status'] == 'failed'
    assert 'outside the immutable source' in report['failures'][0]['message']
    assert not (params.parent / 'derived').exists()


def test_cli_report_cannot_overwrite_source_params(tmp_path, capsys):
    params = make_sorting(tmp_path / 'source')
    original = params.read_bytes()

    result = main(
        [
            str(params),
            '--phy-revision',
            'phy-test-revision',
            '--exporter-revision',
            'exporter-test-revision',
            '--report',
            str(params),
        ]
    )

    assert result == 1
    assert params.read_bytes() == original
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed'
    assert 'outside the immutable source' in report['failures'][0]['message']


def test_cli_report_cannot_target_future_prepared_output(tmp_path, capsys):
    params = make_sorting(tmp_path / 'source')
    params.write_text(
        params.read_text()
        + "time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': 1.0}\n"
        + 'valid_intervals = [[0, 20]]\n'
    )
    output_dir = tmp_path / 'prepared'
    report_path = output_dir / 'params.py'

    result = main(
        [
            str(params),
            '--phy-revision',
            'phy-test-revision',
            '--exporter-revision',
            'exporter-test-revision',
            '--output-dir',
            str(output_dir),
            '--report',
            str(report_path),
        ]
    )

    assert result == 1
    assert not output_dir.exists()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed'
    assert 'prepared output directories' in report['failures'][0]['message']


def test_cli_report_exclusive_create_preserves_racing_file(tmp_path, monkeypatch, capsys):
    params = make_sorting(tmp_path / 'source')
    report_path = tmp_path / 'report.json'

    def create_report_during_validation(*args, **kwargs):
        report_path.write_text('keep me', encoding='utf-8')
        return {'status': 'passed'}

    monkeypatch.setattr(validator, 'validate_export', create_report_during_validation)
    result = main(
        [
            str(params),
            '--phy-revision',
            'phy-test-revision',
            '--exporter-revision',
            'exporter-test-revision',
            '--report',
            str(report_path),
        ]
    )

    assert result == 1
    assert report_path.read_text(encoding='utf-8') == 'keep me'
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed'
    assert 'already exists' in report['failures'][0]['message']
