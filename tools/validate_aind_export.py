"""Validate an AIND-produced sorting against Phy's declared source contract.

This checks software interoperability. It cannot establish acquisition truth;
the exporter revision and authoritative exposure/filler policy must be supplied
and reviewed by the producer.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.source import Source, hash_file

REPORT_FORMAT = 'phy_aind_export_validation'
REPORT_VERSION = 1


def _failure(error):
    return {'type': type(error).__name__, 'message': str(error)}


def _cli_failure(error, args):
    report = {
        'format': REPORT_FORMAT,
        'format_version': REPORT_VERSION,
        'status': 'failed',
        'failures': [_failure(error)],
        'engineering_validation_only': True,
        'acquisition_truth_established': False,
        'phy_revision': args.phy_revision,
        'exporter_revision': args.exporter_revision,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return 1


def _input_inventory(source_path):
    """Hash regular files in the source without interpreting their contents."""
    return {
        path.name: {'bytes': path.stat().st_size, 'sha256': hash_file(path)}
        for path in sorted(source_path.iterdir())
        if path.is_file()
    }


def validate_export(
    params_path,
    *,
    phy_revision,
    exporter_revision,
    output_dir=None,
    max_spikes=2_000_000,
    max_bytes=536_870_912,
    batch_size=100_000,
):
    """Return a JSON-serializable validation report; never edits the source."""
    started = time.perf_counter()
    params_path = Path(params_path).resolve()
    source_path = params_path.parent
    report = {
        'format': REPORT_FORMAT,
        'format_version': REPORT_VERSION,
        'status': 'failed',
        'failures': [],
        'engineering_validation_only': True,
        'acquisition_truth_established': False,
        'requires_authoritative_exposure_and_filler_policy': True,
        'phy_revision': phy_revision,
        'exporter_revision': exporter_revision,
        'source': {'params_path': str(params_path), 'path': str(source_path)},
        'budgets': {'max_spikes': int(max_spikes), 'max_bytes': int(max_bytes)},
        'timing_seconds': {},
    }
    if not phy_revision or not exporter_revision:
        report['failures'].append(
            _failure(ValueError('Both Phy and exporter revisions must be supplied.'))
        )
        report['timing_seconds']['total'] = time.perf_counter() - started
        return report
    try:
        phase = time.perf_counter()
        with Source(params_path, batch_size=batch_size) as source:
            report['timing_seconds']['source_validation'] = time.perf_counter() - phase
            report['source'].update(
                {
                    'sha256': source.identity['sha256'],
                    'num_spikes': source.n_spikes,
                    'num_units': len(source.unit_ids),
                    'sample_rate_hz': source.sample_rate,
                    'num_samples': source.num_samples,
                    'time_base': source.time_base,
                    'exposure': {
                        'valid_intervals': source.valid_intervals.tolist(),
                        'valid_samples': int(
                            (source.valid_intervals[:, 1] - source.valid_intervals[:, 0]).sum()
                        ),
                        'valid_seconds': float(
                            (source.valid_intervals[:, 1] - source.valid_intervals[:, 0]).sum()
                            / source.sample_rate
                        ),
                    },
                    'array_counts': {
                        name: {'rows': int(len(array)), 'shape': list(array.shape)}
                        for name, array in sorted(source.arrays.items())
                    },
                }
            )
            source_root = source.path
            declared_intervals = (
                'valid_intervals' in source.params
                or 'valid_intervals' in source.upstream_recording
            )
            if source.time_base is None:
                raise ValueError('AIND export requires an explicit time_base contract.')
            if source.time_base['kind'] != 'elapsed_ticks':
                raise ValueError('AIND export time_base.kind must be elapsed_ticks.')
            if not declared_intervals:
                raise ValueError(
                    'AIND export requires explicit authoritative valid_intervals in params.py '
                    'or the waveform manifest.'
                )
        phase = time.perf_counter()
        report['source']['input_files'] = _input_inventory(source_root)
        report['timing_seconds']['input_hashing'] = time.perf_counter() - phase
        if output_dir is not None:
            destination = Path(output_dir).resolve()
            if destination == source_root or source_root in destination.parents:
                raise ValueError('Prepared output must be outside the immutable source directory.')
            if destination.exists():
                raise ValueError(f'Prepared output must be fresh; already exists: {destination}.')
            phase = time.perf_counter()
            prepared = prepare_large(
                params_path,
                destination,
                max_spikes=max_spikes,
                max_bytes=max_bytes,
                batch_size=batch_size,
            )
            report['timing_seconds']['prepare'] = time.perf_counter() - phase
            manifest = json.loads((prepared / '_phy_curation_manifest.json').read_text())
            report['prepared_output'] = {
                'path': str(prepared),
                'num_spikes': manifest['bounded_num_spikes'],
                'num_units': len(manifest['unit_ids']),
                'waveform_coverage': manifest['waveform_coverage'],
                'budgets': manifest['budgets'],
                'output_bytes': sum(p.stat().st_size for p in prepared.rglob('*') if p.is_file()),
            }
    except Exception as error:
        report['failures'].append(_failure(error))
    if not report['failures']:
        report['status'] = 'passed'
    report['timing_seconds']['total'] = time.perf_counter() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('params', type=Path, help='AIND sorting params.py')
    parser.add_argument('--phy-revision', required=True)
    parser.add_argument('--exporter-revision', required=True)
    parser.add_argument(
        '--output-dir', type=Path, help='Optional fresh bounded package destination'
    )
    parser.add_argument('--max-spikes', type=int, default=2_000_000)
    parser.add_argument('--max-bytes', type=int, default=536_870_912)
    parser.add_argument('--batch-size', type=int, default=100_000)
    parser.add_argument(
        '--report', type=Path, help='Write JSON report to this path (default stdout)'
    )
    args = parser.parse_args(argv)
    if args.report:
        source_root = args.params.resolve().parent
        report_path = args.report.resolve()
        protected_roots = [source_root]
        if args.output_dir is not None:
            protected_roots.append(args.output_dir.resolve())
        if any(report_path == root or root in report_path.parents for root in protected_roots):
            failure = ValueError(
                'Report path must be outside the immutable source and prepared output directories.'
            )
        elif args.report.exists() or args.report.is_symlink():
            failure = ValueError(f'Report path must be fresh; already exists: {args.report}.')
        else:
            failure = None
        if failure is not None:
            return _cli_failure(failure, args)
    report = validate_export(
        args.params,
        phy_revision=args.phy_revision,
        exporter_revision=args.exporter_revision,
        output_dir=args.output_dir,
        max_spikes=args.max_spikes,
        max_bytes=args.max_bytes,
        batch_size=args.batch_size,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.report:
        try:
            with args.report.open('x', encoding='utf-8') as stream:
                stream.write(payload)
        except FileExistsError:
            return _cli_failure(
                FileExistsError(f'Report path must be fresh; already exists: {args.report}.'),
                args,
            )
    else:
        sys.stdout.write(payload)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
