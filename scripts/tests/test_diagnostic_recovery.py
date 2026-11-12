"""Recovery and malformed-evidence tests; no Qt or GPU is required."""

import contextlib
import errno
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest


def test_bundle_supports_filesystems_without_hardlinks(tmp_path, fake_process, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, 'Hard links unavailable')

    monkeypatch.setattr(diagnostic.os, 'link', unsupported)
    target = tmp_path / 'report.zip'
    collect(tmp_path / 'workspace', '--output', str(target))
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        assert 'results.json' in bundle.namelist()


def test_bundle_fallback_preserves_a_concurrently_created_report(
    tmp_path, fake_process, monkeypatch
):
    target = tmp_path / 'report.zip'

    def appeared(*args):
        target.write_bytes(b'previous report')
        raise FileExistsError(errno.EEXIST, 'Report appeared concurrently')

    monkeypatch.setattr(diagnostic.os, 'link', appeared)
    with pytest.raises(FileExistsError):
        collect(tmp_path / 'workspace', '--output', str(target))
    assert target.read_bytes() == b'previous report'


def test_bundle_fallback_removes_its_own_failed_copy(tmp_path, fake_process, monkeypatch):
    import shutil

    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, 'Hard links unavailable')

    def failed_copy(source, destination):
        destination.write(b'partial')
        raise OSError('Interrupted copy')

    monkeypatch.setattr(diagnostic.os, 'link', unsupported)
    monkeypatch.setattr(shutil, 'copyfileobj', failed_copy)
    target = tmp_path / 'report.zip'
    with pytest.raises(OSError, match='Interrupted copy'):
        collect(tmp_path / 'workspace', '--output', str(target))
    assert not target.exists()


@pytest.mark.parametrize('replace', ['runs', 'attempt'])
def test_recovery_never_writes_through_directory_symlink(tmp_path, fake_process, replace):
    workspace = tmp_path / 'workspace'
    collect(workspace)
    results = read_results(workspace)
    results['probes'][0]['status'] = 'RUNNING'
    (workspace / 'results.json').write_text(json.dumps(results))
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'result.json').write_text('unrelated data')
    directory = workspace / 'runs' if replace == 'runs' else workspace / 'runs' / 'run-0001'
    directory.rename(tmp_path / 'original')
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlinks unavailable')
    with pytest.raises(ValueError, match='directory'):
        diagnostic.main(
            ['--workspace', str(workspace), '--bundle', '--output', str(tmp_path / 'report.zip')]
        )
    assert (outside / 'result.json').read_text() == 'unrelated data'


SCRIPT = Path(__file__).resolve().parents[1] / 'diagnose_graphics.py'
SPEC = importlib.util.spec_from_file_location('diagnostic_recovery', SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.fixture
def fake_process(monkeypatch):
    calls = []
    monkeypatch.setenv('DISPLAY', 'mock-display')

    def run(command, env, timeout):
        calls.append(command)
        return {
            'status': 'PASS',
            'details': {'status': 'PASS'},
            'last_stage': 'mocked',
            'events': [],
        }, 'Retained terminal evidence.\n'

    monkeypatch.setattr(diagnostic, 'run_process', run)
    return calls


def collect(workspace, *extra):
    return diagnostic.main(
        [
            '--workspace',
            str(workspace),
            '--probe',
            'inventory',
            '--configuration',
            'baseline',
            *extra,
        ]
    )


def read_results(workspace):
    return json.loads((workspace / 'results.json').read_text())


@pytest.mark.skipif(os.name != 'posix', reason='SIGKILL owner-death regression')
def test_lock_is_released_by_os_after_controller_death(tmp_path):
    ready = tmp_path / 'ready'
    code = (
        'import importlib.util, pathlib, time; '
        f's=importlib.util.spec_from_file_location("d", {str(SCRIPT)!r}); '
        'd=importlib.util.module_from_spec(s); s.loader.exec_module(d); '
        f'lock=d._workspace_lock(pathlib.Path({str(tmp_path)!r})); lock.__enter__(); '
        f'pathlib.Path({str(ready)!r}).write_text("ready"); time.sleep(30)'
    )
    proc = subprocess.Popen(
        [sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(ValueError, match='live run'), diagnostic._workspace_lock(tmp_path):
            pass
        proc.kill()
        proc.wait(timeout=5)
        assert (tmp_path / '.lock').exists()
        with diagnostic._workspace_lock(tmp_path):
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def test_lock_refuses_symlink(tmp_path):
    target = tmp_path / 'unrelated'
    target.write_text('do not change')
    try:
        (tmp_path / '.lock').symlink_to(target)
    except OSError:
        pytest.skip('symlinks unavailable')
    with pytest.raises(ValueError, match='symlink'), diagnostic._workspace_lock(tmp_path):
        pass
    assert target.read_text() == 'do not change'


def test_resume_preserves_orphan_facts_without_reusing_directory(tmp_path, fake_process):
    workspace = tmp_path / 'workspace'
    collect(workspace)
    orphan = workspace / 'runs' / 'run-0002'
    orphan.mkdir()
    (orphan / 'unrelated.txt').write_text('never bundle an unjournaled directory sweep')
    results = read_results(workspace)
    results['invocations'][0]['status'] = 'RUNNING'
    (workspace / 'results.json').write_text(json.dumps(results))
    output = tmp_path / 'report.zip'
    collect(workspace, '--output', str(output))
    updated = read_results(workspace)
    assert updated['probes'][-1]['attempt_id'] == 'run-0003'
    assert updated['invocations'][0]['status'] == 'INTERRUPTED'
    abandoned = updated['abandoned_attempts'][0]
    assert abandoned['attempt_id'] == 'run-0002'
    assert 'probe' not in abandoned
    assert 'returncode' not in abandoned
    with zipfile.ZipFile(output) as bundle:
        assert not any(name.startswith('runs/run-0002/') for name in bundle.namelist())


def test_bundle_recovers_journaled_intent_without_inventing_child_outcome(tmp_path, fake_process):
    workspace = tmp_path / 'workspace'
    collect(workspace)
    results = read_results(workspace)
    pending = dict(
        results['probes'][0],
        attempt_id='run-0002',
        status='RUNNING',
        details={},
        events=[],
        log_path='runs/run-0002/probe.log',
        result_path='runs/run-0002/result.json',
    )
    directory = workspace / 'runs' / 'run-0002'
    directory.mkdir()
    (directory / 'probe.log').write_text('Partial native evidence survived.\n')
    (directory / 'result.json').write_text(json.dumps(pending))
    results['probes'].append(pending)
    results['invocations'][0]['status'] = 'RUNNING'
    (workspace / 'results.json').write_text(json.dumps(results))
    output = tmp_path / 'recovered.zip'
    diagnostic.main(['--workspace', str(workspace), '--bundle', '--output', str(output)])
    assert len(fake_process) == 1
    with zipfile.ZipFile(output) as bundle:
        recovered = json.loads(bundle.read('results.json'))['probes'][-1]
        assert recovered['status'] == 'INTERRUPTED'
        assert recovered['details'] == {}
        assert 'returncode' not in recovered
        assert b'Partial native evidence' in bundle.read('runs/run-0002/probe.log')


def test_missing_and_corrupt_artifacts_do_not_drop_log_or_zip(tmp_path, fake_process, monkeypatch):
    def run(command, env, timeout):
        artifact_dir = Path(command[command.index('--_artifacts') + 1])
        (artifact_dir / 'corrupt.png').write_bytes(b'\x89PNG\r\n\x1a\ntruncated')
        return {
            'status': 'FAIL',
            'last_stage': 'native-failure',
            'details': {'status': 'FAIL'},
            'events': [
                {'event': 'artifact', 'path': 'missing.png'},
                {'event': 'artifact', 'path': 'corrupt.png'},
                {'event': 'artifact', 'path': '../outside.png'},
            ],
        }, 'Important failure traceback.\n'

    monkeypatch.setattr(diagnostic, 'run_process', run)
    workspace, output = tmp_path / 'workspace', tmp_path / 'report.zip'
    collect(workspace, '--output', str(output))
    with zipfile.ZipFile(output) as bundle:
        result = json.loads(bundle.read('results.json'))
        probe = result['probes'][0]
        assert probe['status'] == 'FAIL'
        assert not probe['artifacts']
        assert len(probe['artifact_omissions']) == 3
        assert b'Important failure traceback' in bundle.read(probe['log_path'])
        assert not any(name.endswith('.png') for name in bundle.namelist())
    collect(workspace)
    assert read_results(workspace)['probes'][-1]['attempt_id'] == 'run-0002'


def test_typed_malformed_protocol_is_retained_as_findings(tmp_path, fake_process, monkeypatch):
    malformed = [
        {'event': 'python-message', 'level': 'WARNING', 'message': ['wrong type']},
        {'event': 'qt-message', 'severity': 1, 'message': {}},
        {'event': 'context', 'GL_RENDERER': []},
        {'event': 'application', 'platform': []},
    ]

    def run(*args):
        return {
            'status': 'PASS',
            'last_stage': 'mocked',
            'details': {'status': 'PASS'},
            'events': malformed,
        }, 'Original malformed output.\n'

    monkeypatch.setattr(diagnostic, 'run_process', run)
    workspace, output = tmp_path / 'workspace', tmp_path / 'report.zip'
    collect(workspace, '--output', str(output))
    with zipfile.ZipFile(output) as bundle:
        result = json.loads(bundle.read('results.json'))
        assert len(result['findings']) == 4
        assert all(e['event'] == 'protocol-error' for e in result['probes'][0]['events'])
        assert b'Original malformed output' in bundle.read('runs/run-0001/probe.log')
    # The sanitized workspace remains valid on the next public invocation.
    collect(workspace)


def test_raw_protocol_parser_rejects_typed_invalid_fields():
    code = (
        'print(\'PHY_DIAGNOSTIC {"event":"python-message","level":"WARNING","message":[]}\'); '
        'print(\'PHY_DIAGNOSTIC {"event":"context","GL_RENDERER":[]}\'); '
        'print(\'PHY_DIAGNOSTIC {"event":"result","status":"PASS"}\')'
    )
    result, log = diagnostic.run_process([sys.executable, '-c', code], os.environ.copy(), 5)
    assert result['status'] == 'PASS'
    assert sum(e['event'] == 'protocol-error' for e in result['events']) == 2
    assert '"message":[]' in log


@pytest.mark.parametrize('marker', [None, [], 'wrong shape'])
def test_malformed_marker_is_a_clear_error(tmp_path, fake_process, marker):
    collect(tmp_path)
    (tmp_path / '.phy-diagnostic.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='ownership marker'):
        collect(tmp_path)


def test_unsafe_manifest_paths_are_rejected_without_file_access(tmp_path, fake_process):
    collect(tmp_path)
    result = read_results(tmp_path)
    result['probes'][0]['log_path'] = 'runs/../../outside.log'
    (tmp_path / 'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Unsafe or inconsistent'):
        collect(tmp_path)


def test_unexpected_controller_failure_still_packages_journal_and_traceback(
    tmp_path, fake_process, monkeypatch
):
    def fail(*args):
        raise RuntimeError('Unexpected controller-side error')

    monkeypatch.setattr(diagnostic, 'run_process', fail)
    workspace, output = tmp_path / 'workspace', tmp_path / 'report.zip'
    collect(workspace, '--output', str(output))
    with zipfile.ZipFile(output) as bundle:
        result = json.loads(bundle.read('results.json'))
        assert result['invocations'][0]['status'] == 'ERROR'
        assert result['probes'][0]['status'] == 'INTERRUPTED'
        assert b'Unexpected controller-side error' in bundle.read('runs/run-0001/probe.log')


def test_corrupt_stored_png_does_not_block_bundle(tmp_path, fake_process):
    collect(tmp_path)
    result = read_results(tmp_path)
    artifacts = tmp_path / 'runs' / 'run-0001' / 'artifacts'
    (artifacts / 'later-corrupted.png').write_bytes(b'not a PNG')
    result['probes'][0]['artifacts'] = ['runs/run-0001/artifacts/later-corrupted.png']
    (tmp_path / 'results.json').write_text(json.dumps(result))
    output = tmp_path / 'report.zip'
    diagnostic.main(['--workspace', str(tmp_path), '--bundle', '--output', str(output)])
    with zipfile.ZipFile(output) as bundle:
        assert json.loads(bundle.read('results.json'))['bundle_omissions']
        assert not any(name.endswith('.png') for name in bundle.namelist())
