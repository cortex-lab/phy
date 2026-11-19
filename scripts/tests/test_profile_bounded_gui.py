"""Guard the standalone GUI profiler's input boundaries without opening Qt."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'profile_bounded_gui.py'
SPEC = importlib.util.spec_from_file_location('profile_bounded_gui', SCRIPT)
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


def arguments(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / '_phy_curation_manifest.json').write_text('{}')
    return SimpleNamespace(
        bounded=source,
        output=tmp_path / 'output',
        repeats=1,
        merges=0,
        events=None,
        max_copy_bytes=1024,
        timeout=1,
        merge_undo=False,
        profile=False,
    )


@pytest.mark.parametrize('location', ['input', 'input/nested', '.'])
def test_nested_output_rejected_before_creation(tmp_path, location):
    args = arguments(tmp_path)
    args.output = tmp_path / location
    with pytest.raises(ValueError, match='non-nested'):
        profiler.run(args)
    assert sorted(p.name for p in args.bounded.iterdir()) == ['_phy_curation_manifest.json']


def test_copy_budget_and_symlinks_rejected_before_output(tmp_path):
    args = arguments(tmp_path)
    (args.bounded / 'large').write_bytes(b'x' * 1024)
    with pytest.raises(ValueError, match='max-copy-bytes'):
        profiler.run(args)
    assert not args.output.exists()
    (args.bounded / 'large').unlink()
    (args.bounded / 'alias').symlink_to(args.bounded / '_phy_curation_manifest.json')
    with pytest.raises(ValueError, match='Symbolic links'):
        profiler.run(args)
    assert not args.output.exists()


def test_failed_display_still_reports_unchanged_source(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    cache = args.bounded / '.phy'
    cache.mkdir()
    (cache / 'state.json').write_text('{"external": "/not/a/profiling/input"}')
    files, _ = profiler._inventory(args.bounded, args.max_copy_bytes)
    before = profiler._hashes(args.bounded, files)

    def fail_display(*a, **kw):
        raise OSError('test display failure')

    monkeypatch.setattr(
        profiler.subprocess,
        'check_output',
        lambda *a, **kw: 'test-revision' if kw.get('text') else b'test-revision',
    )
    monkeypatch.setattr(profiler.subprocess, 'Popen', fail_display)
    with pytest.raises(OSError, match='test display failure'):
        profiler.run(args)
    report = json.loads((args.output / 'result.json').read_text())
    assert report['status'] == 'failed'
    assert report['source_unchanged']
    assert report['source_hashes_before'] == report['source_hashes_after'] == before
    assert report['omitted_local_cache_files'] == 1
    assert not (args.output / 'working-package' / '.phy').exists()
    assert (args.output / 'working-package' / '_phy_curation_manifest.json').read_text() == '{}'


def test_seed_clears_every_membership_metadata_field(tmp_path):
    from phy.apps.large_recording.curation import BoundedSession
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.tests.test_adapter import make_gui_sorting

    params = make_gui_sorting(tmp_path / 'source')
    (params.parent / 'cluster_snr.tsv').write_text(
        'cluster_id\tsnr\n0\t10\n1\t20\n2\t30\n99\t40\n'
    )
    bounded = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    profiler._seed(bounded, 3, 1)
    with BoundedSession(bounded) as session:
        target = session.events[0]['target']
        assert session.metadata['snr'][target] is None
        assert len(session.events) == 3
