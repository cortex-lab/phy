"""Exercise the standalone collector without requiring Qt or a display."""

import base64
import copy
import importlib.util
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'diagnose_graphics.py'
SPEC = importlib.util.spec_from_file_location('diagnose_graphics', SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def run_python(code, timeout=5):
    return diagnostic.run_process([sys.executable, '-c', code], os.environ.copy(), timeout)


def test_success_with_native_stderr():
    result, log = run_python(
        "import sys; print('native warning', file=sys.stderr); "
        'print(\'PHY_DIAGNOSTIC {"event":"stage","stage":"draw"}\'); '
        'print(\'PHY_DIAGNOSTIC {"event":"result","status":"PASS"}\')'
    )
    assert result['status'] == 'PASS'
    assert result['last_stage'] == 'draw'
    assert 'native warning' in log


@pytest.mark.parametrize('termination', ['raise SystemExit(19)', 'import os; os._exit(19)'])
def test_late_process_failure_invalidates_pass(termination):
    result, _ = run_python(
        'print(\'PHY_DIAGNOSTIC {"event":"result","status":"PASS"}\', flush=True); ' + termination
    )
    assert result['status'] == 'FAIL'
    assert result['returncode'] == 19


@pytest.mark.skipif(os.name != 'posix', reason='POSIX signal return code')
def test_native_signal_is_reported():
    result, _ = run_python('import os, signal; os.kill(os.getpid(), signal.SIGKILL)')
    assert result['status'] == 'FAIL'
    assert result['returncode'] == -9


def test_hung_probe_is_killed_and_partial_stage_preserved():
    result, log = run_python(
        'import time; print(\'PHY_DIAGNOSTIC {"event":"stage","stage":"show"}\', '
        'flush=True); time.sleep(30)',
        timeout=0.3,
    )
    assert result['status'] == 'TIMEOUT'
    assert result['last_stage'] == 'show'
    assert 'show' in log


def test_log_size_is_bounded(monkeypatch):
    monkeypatch.setattr(diagnostic, 'MAX_LOG_BYTES', 100)
    result, log = run_python("print('x' * 1000)")
    assert result['log_truncated']
    assert len(log) < 150


def test_malformed_protocol_does_not_destroy_report():
    result, _ = run_python(
        "print('PHY_DIAGNOSTIC null'); print('PHY_DIAGNOSTIC []'); "
        'print(\'PHY_DIAGNOSTIC {"not_an_event":1}\'); '
        "print('PHY_DIAGNOSTIC invalid'); "
        'print(\'PHY_DIAGNOSTIC {"event":"result","status":"PASS"}\')'
    )
    assert result['status'] == 'PASS'
    assert len(result['events']) == 1


def test_timeout_preserves_fault_handler_thread_stack():
    result, log = run_python(
        'import faulthandler, time; faulthandler.enable(); '
        'faulthandler.dump_traceback_later(0.1); time.sleep(30)',
        timeout=0.5,
    )
    assert result['status'] == 'TIMEOUT'
    assert 'Timeout' in log
    assert '<string>' in log


def test_diagnostic_checks_survive_python_optimization():
    code = (
        'import importlib.util; '
        f's=importlib.util.spec_from_file_location("kit", {str(SCRIPT)!r}); '
        'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
        'm.require(False, "expected diagnostic failure")'
    )
    result, log = diagnostic.run_process([sys.executable, '-O', '-c', code], os.environ.copy(), 5)
    assert result['returncode'] != 0
    assert 'expected diagnostic failure' in log


def test_redaction_of_structured_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    value = diagnostic.redact_data({'path': [str(tmp_path / 'file'), str(Path.home() / 'env')]})
    assert value == {'path': [f'<CWD>{os.sep}file', f'<HOME>{os.sep}env']}


def test_collection_without_phy_or_display(tmp_path):
    """The stdlib-only parent must still package failures when dependencies are absent."""
    target = tmp_path / 'report.zip'
    empty_env = tmp_path / 'empty-env'
    # Copied uv-managed Python 3.10 executables lose their stdlib location when
    # creating a nested venv. Use symlinks on POSIX, as `python -m venv` does.
    venv.EnvBuilder(with_pip=False, symlinks=os.name != 'nt').create(empty_env)
    interpreter = empty_env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = os.environ.copy()
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'PYTHONPATH'):
        env.pop(key, None)
    env['UNRELATED_SECRET'] = 'must-not-appear-in-bundle'
    completed = subprocess.run(
        [
            str(interpreter),
            str(SCRIPT),
            '--baseline-only',
            '--output',
            str(target),
            '--symptoms',
            'Plots freeze',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(target) as bundle:
        results = json.loads(bundle.read('results.json'))
        assert results['symptoms'] == 'Plots freeze'
        assert results['probes'][0]['status'] == 'FAIL'  # no site-packages
        if sys.platform.startswith('linux'):
            assert all(p['status'] == 'UNAVAILABLE' for p in results['probes'][1:])
        assert bundle.read('diagnose_graphics.py') == SCRIPT.read_bytes()
        assert 'REPORT.md' in bundle.namelist()
        assert 'README.md' in bundle.namelist()
        assert all(
            'must-not-appear-in-bundle' not in bundle.read(name).decode()
            for name in bundle.namelist()
        )
        assert 'No configuration passed' in bundle.read('REPORT.md').decode()
    # Never overwrite a report from a previous diagnostic run.
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic.main(['--output', str(target), '--baseline-only', '--timeout', '0.01'])
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    'statuses, expected',
    [
        (['PASS'] * 5, 'Baseline synthetic rendering passed'),
        (['PASS', 'PASS', 'FAIL', 'PASS', 'PASS'], 'No configuration passed'),
        (['PASS'] * 4 + ['FAIL'], 'No configuration passed'),
        (['UNAVAILABLE'] * 5, 'No configuration passed'),
    ],
)
def test_report_never_claims_partial_success_as_workaround(statuses, expected):
    report = diagnostic.make_report(
        {
            'script_sha256': '123',
            'python': 'test',
            'session': 'SSH',
            'symptoms': '',
            'configurations': [{'name': 'baseline', 'overrides': {}}],
            'probes': [
                {
                    'configuration': 'baseline',
                    'probe': kind,
                    'status': status,
                    'last_stage': 'draw',
                }
                for kind, status in zip(('qt', 'native', 'phy', 'docked', 'scene'), statuses)
            ],
        }
    )
    assert expected in report
    assert ('phy template-gui' in report) == all(status == 'PASS' for status in statuses)


def test_non_linux_does_not_invent_xcb(monkeypatch, tmp_path, workspace_runner):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    diagnostic.main(['--workspace', str(tmp_path), '--probe', 'qt'])
    results = json.loads((tmp_path / 'results.json').read_text())
    assert [p['configuration'] for p in results['probes']] == ['baseline']
    assert len(workspace_runner) == 1


def test_linux_comparison_preserves_baseline(monkeypatch, tmp_path, workspace_runner):
    monkeypatch.setattr(sys, 'platform', 'linux')
    diagnostic.main(['--workspace', str(tmp_path), '--probe', 'qt'])
    results = json.loads((tmp_path / 'results.json').read_text())
    configs = {c['name']: c['overrides'] for c in results['configurations']}
    assert configs['baseline'] == {}
    assert configs['software'] == {'LIBGL_ALWAYS_SOFTWARE': 'true'}
    assert configs['xcb-software'] == {'LIBGL_ALWAYS_SOFTWARE': 'true', 'QT_QPA_PLATFORM': 'xcb'}
    assert configs['xcb-glx'] == {
        'QT_QPA_PLATFORM': 'xcb',
        'QT_XCB_GL_INTEGRATION': 'xcb_glx',
        'PYOPENGL_PLATFORM': 'glx',
    }
    assert len(workspace_runner) == 5


def test_metrics_control_preserves_requested_coverage(tmp_path, workspace_runner):
    diagnostic.main(
        [
            '--workspace',
            str(tmp_path),
            '--probe',
            'native',
            '--configuration',
            'baseline',
            '--capture-mode',
            'metrics',
        ]
    )
    result = json.loads((tmp_path / 'results.json').read_text())
    assert result['probes'][0]['capture_mode'] == 'metrics'
    assert '--_artifacts' not in workspace_runner[0][0]
    assert diagnostic._summary_results(result)['probes'][0]['status'] == 'PARTIAL'


def test_input_block_control_is_not_a_normal_workaround(tmp_path, workspace_runner):
    diagnostic.main(
        [
            '--workspace',
            str(tmp_path),
            '--probe',
            'native',
            '--configuration',
            'baseline',
            '--input-mode',
            'block',
        ]
    )
    result = json.loads((tmp_path / 'results.json').read_text())
    assert result['probes'][0]['input_mode'] == 'block'
    assert diagnostic._summary_results(result)['probes'][0]['status'] == 'PARTIAL'


def test_controls_do_not_obscure_failed_normal_rerun():
    normal = {'configuration': 'baseline', 'probe': 'docked', 'status': 'PASS'}
    control = dict(normal, input_mode='block')
    failed = dict(normal, status='FAIL')
    summary = diagnostic._summary_results({'probes': [normal, control, failed]})
    assert len(summary['probes']) == 2
    assert summary['probes'][0]['status'] == 'FAIL'
    assert summary['probes'][1]['status'] == 'PARTIAL'


@pytest.mark.skipif(importlib.util.find_spec('PyQt5') is None, reason='Qt not installed')
@pytest.mark.parametrize('kind, expected', [('qt', 'PASS'), ('native', 'UNAVAILABLE')])
def test_qt_plugin_debug_teardown_and_headless_classification(kind, expected):
    # Leaving a Python Qt message callback installed during plugin unload caused
    # a native crash after an apparently successful probe.
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', QT_DEBUG_PLUGINS='1')
    result, log = diagnostic.run_process([sys.executable, str(SCRIPT), '--_probe', kind], env, 10)
    assert result['status'] == expected, log
    assert result['returncode'] == 0


@pytest.fixture
def workspace_runner(monkeypatch):
    calls = []
    # The process runner is mocked; this name ensures display discovery does not
    # skip the desired probe when these tests run on a displayless CI worker.
    monkeypatch.setenv('DISPLAY', 'diagnostic-test-display')

    def fake_run_process(command, env, timeout):
        calls.append((command, env, timeout))
        kind = command[command.index('--_probe') + 1]
        details = {'status': 'PASS'}
        if kind == 'inventory':
            details.update(phy_version='2.1.0', qt_runtime_version='5.15')
        return {
            'status': 'PASS',
            'returncode': 0,
            'last_stage': 'mocked',
            'details': details,
            'events': [],
        }, 'Mocked probe; no Qt or rendering was executed.\n'

    monkeypatch.setattr(diagnostic, 'run_process', fake_run_process)
    return calls


def test_public_capabilities_require_no_probe_execution(workspace_runner, capsys):
    assert diagnostic.main(['--list-probes']) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert set(catalog['probes']) == {'inventory', 'qt', 'native', 'phy', 'docked', 'scene'}
    assert catalog['workspace_attempt_max'] == 64
    assert catalog['workspace_artifact_bytes_max'] == 30_000_000
    assert not workspace_runner


def test_selected_attempts_accumulate_and_bundle_without_rerunning(
    tmp_path, workspace_runner, capsys
):
    workspace = tmp_path / 'workspace'
    assert (
        diagnostic.main(
            ['--workspace', str(workspace), '--probe', 'inventory', '--configuration', 'baseline']
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert first['zip'] is None
    assert (
        diagnostic.main(
            ['--workspace', str(workspace), '--probe', 'native', '--configuration', 'baseline']
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert second['latest_attempts'][0]['attempt_id'] == 'run-0002'
    result = json.loads((workspace / 'results.json').read_text())
    assert [p['probe'] for p in result['probes']] == ['inventory', 'native']
    assert [p['attempt_id'] for p in result['probes']] == ['run-0001', 'run-0002']
    output = tmp_path / 'combined.zip'
    (workspace / 'AGENT_NOTES.md').write_text('Hypothesis only; see run-0002.\n')
    assert (
        diagnostic.main(['--workspace', str(workspace), '--bundle', '--output', str(output)]) == 0
    )
    bundled = json.loads(capsys.readouterr().out)
    assert bundled['attempts'] == 2
    assert len(workspace_runner) == 2
    with zipfile.ZipFile(output) as archive:
        assert len(json.loads(archive.read('results.json'))['probes']) == 2
        assert 'runs/run-0001/probe.log' in archive.namelist()
        assert 'runs/run-0002/probe.log' in archive.namelist()
        assert archive.read('AGENT_NOTES.md').decode() == 'Hypothesis only; see run-0002.\n'
        assert '.phy-diagnostic.json' not in archive.namelist()
        assert archive.read('diagnose_graphics.py') == SCRIPT.read_bytes()
        assert 'Baseline synthetic rendering passed' not in archive.read('REPORT.md').decode()


def test_workspace_refuses_changed_graphics_session(tmp_path, workspace_runner, monkeypatch):
    workspace = tmp_path / 'workspace'
    diagnostic.main(
        ['--workspace', str(workspace), '--probe', 'inventory', '--configuration', 'baseline']
    )
    monkeypatch.setenv('QT_QPA_PLATFORM', 'another-backend')
    with pytest.raises(ValueError, match='identity differs'):
        diagnostic.main(
            ['--workspace', str(workspace), '--probe', 'inventory', '--configuration', 'baseline']
        )
    assert len(workspace_runner) == 1
    with diagnostic._workspace_lock(workspace):
        assert (workspace / '.lock').is_file()


def test_workspace_budget_checkpoints_completed_attempts(tmp_path, workspace_runner, monkeypatch):
    original = diagnostic.run_process
    clock = {'seconds': 0.0}
    monkeypatch.setattr(diagnostic.time, 'monotonic', lambda: clock['seconds'])

    def slow_fake(*args):
        result = original(*args)
        clock['seconds'] += 1
        return result

    monkeypatch.setattr(diagnostic, 'run_process', slow_fake)
    workspace = tmp_path / 'workspace'
    diagnostic.main(
        [
            '--workspace',
            str(workspace),
            '--probe',
            'qt',
            '--probe',
            'native',
            '--configuration',
            'baseline',
            '--budget',
            '0.5',
        ]
    )
    result = json.loads((workspace / 'results.json').read_text())
    assert result['invocations'][-1]['status'] == 'BUDGET_EXHAUSTED'
    assert result['invocations'][-1]['unperformed_attempts'] == 1
    assert len(result['probes']) == 1
    assert len(workspace_runner) == 1
    assert (workspace / result['probes'][0]['result_path']).is_file()


@pytest.mark.parametrize('relative', ['../outside', '/absolute', '..\\outside', 'C:outside'])
def test_workspace_artifact_path_safety(tmp_path, relative):
    with pytest.raises(ValueError, match='Unsafe artifact'):
        diagnostic._safe_file(tmp_path, relative)


def test_workspace_refuses_unowned_nonempty_directory(tmp_path, workspace_runner):
    (tmp_path / 'unrelated.txt').write_text('leave this alone')
    with pytest.raises(ValueError, match='not owned by this kit'):
        diagnostic.main(['--workspace', str(tmp_path), '--probe', 'inventory'])
    assert (tmp_path / 'unrelated.txt').read_text() == 'leave this alone'
    assert not (tmp_path / '.lock').exists()
    assert not workspace_runner


def test_pass_with_warnings_keeps_deduplicated_findings(tmp_path, workspace_runner, monkeypatch):
    original = diagnostic.run_process

    def with_warnings(*args):
        result, log = original(*args)
        warning = {'event': 'qt-message', 'severity': 1, 'message': 'Relevant Qt warning'}
        result['events'] = [
            warning,
            warning,
            {'event': 'qt-message', 'severity': 4, 'message': 'Information'},
            {
                'event': 'python-message',
                'level': 'DEBUG',
                'render_exception': True,
                'message': 'Swallowed paint error',
            },
        ]
        return result, log

    monkeypatch.setattr(diagnostic, 'run_process', with_warnings)
    workspace = tmp_path / 'workspace'
    diagnostic.main(
        ['--workspace', str(workspace), '--probe', 'qt', '--configuration', 'baseline']
    )
    result = json.loads((workspace / 'results.json').read_text())
    assert result['probes'][0]['status'] == 'PASS'
    assert {f['observation'] for f in result['findings']} == {
        'Relevant Qt warning',
        'Swallowed paint error',
    }
    assert all(f['evidence'] == ['runs/run-0001/probe.log'] for f in result['findings'])


def test_only_announced_synthetic_pngs_are_bundled(tmp_path, workspace_runner, monkeypatch):
    original = diagnostic.run_process
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=='
    )

    def with_artifacts(command, env, timeout):
        result, log = original(command, env, timeout)
        directory = Path(command[command.index('--_artifacts') + 1])
        (directory / 'announced.png').write_bytes(png)
        (directory / 'unannounced.png').write_bytes(png)
        (directory / 'private.txt').write_text('not an allowed artifact')
        result['events'] = [
            {
                'event': 'artifact',
                'path': 'announced.png',
                'scene_capture': {'expected': 'synthetic scientific content'},
            }
        ]
        return result, log

    monkeypatch.setattr(diagnostic, 'run_process', with_artifacts)
    workspace = tmp_path / 'workspace'
    output = tmp_path / 'report.zip'
    diagnostic.main(
        [
            '--workspace',
            str(workspace),
            '--probe',
            'scene',
            '--configuration',
            'baseline',
            '--output',
            str(output),
        ]
    )
    with zipfile.ZipFile(output) as archive:
        assert 'runs/run-0001/artifacts/announced.png' in archive.namelist()
        assert not any(
            name.endswith(('unannounced.png', 'private.txt')) for name in archive.namelist()
        )
        result = json.loads(archive.read('results.json'))
        assert result['probes'][0]['events'][0]['scene_capture']['expected']


@pytest.fixture
def exposed_surface():
    return {'visible': True, 'exposed': True, 'context_valid': True}


def observed_frame(pixel, timestamp=1.0, **fields):
    return dict(
        pixel=pixel, timestamp=timestamp, sampling_valid=True, context_matches=True, **fields
    )


def test_early_bad_frame_then_expected_retains_transient_evidence(exposed_surface):
    frames = [observed_frame([0, 0, 0, 255]), observed_frame([255, 0, 0, 255], timestamp=1.1)]
    original = copy.deepcopy(frames)
    result = diagnostic.assess_frames(frames, [255, 0, 0], exposed_surface, 3)
    assert result['status'] == 'PASS'
    assert result['classification'] == 'transient-frame-recovered'
    assert result['first_frame']['pixel'] == [0, 0, 0, 255]
    assert result['last_frame']['pixel'] == [255, 0, 0, 255]
    assert result['frame_count'] == 2
    assert frames == original


def test_late_regression_is_not_hidden_by_earlier_correct_frame(exposed_surface):
    frames = [observed_frame([255, 0, 0, 255]), observed_frame([0, 0, 0, 255], timestamp=1.1)]
    result = diagnostic.assess_frames(frames, [255, 0, 0], exposed_surface, 3)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'final-observed-frame-mismatch'
    assert result['first_frame']['pixel'] == [255, 0, 0, 255]
    assert result['last_frame']['pixel'] == [0, 0, 0, 255]


def test_persistent_wrong_pixels_are_a_bounded_observation_failure(exposed_surface):
    frames = [observed_frame([0, 0, 0, 255]), observed_frame([0, 0, 0, 255], timestamp=1.2)]
    result = diagnostic.assess_frames(frames, [255, 0, 0], exposed_surface, 4)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'final-observed-frame-mismatch'
    assert result['heartbeat_ticks'] == 4


def test_one_correct_frame_does_not_require_an_invented_second_frame(exposed_surface):
    frame = observed_frame([255, 0, 0, 255])
    result = diagnostic.assess_frames([frame], [255, 0, 0], exposed_surface, 0)
    assert result['status'] == 'PASS'
    assert result['classification'] == 'expected-frame-observed'
    assert result['first_frame'] == result['last_frame'] == frame
    assert result['frame_count'] == 1


@pytest.mark.parametrize('visible, exposed', [(False, False), (True, False), (False, True)])
def test_no_paint_without_visible_exposure_is_unavailable(visible, exposed):
    surface = {'visible': visible, 'exposed': exposed, 'context_valid': True}
    result = diagnostic.assess_frames([], [255, 0, 0], surface, 4)
    assert result['status'] == 'UNAVAILABLE'
    assert result['classification'] == 'surface-not-exposed'
    assert result['first_frame'] is None
    assert result['last_frame'] is None
    assert result['final_surface'] == surface


def test_exposed_surface_no_paint_with_qt_heartbeat_is_redraw_failure(exposed_surface):
    result = diagnostic.assess_frames([], [255, 0, 0], exposed_surface, 5)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'exposed-surface-did-not-paint'


def test_no_qt_dispatch_evidence_does_not_blame_missing_redraw(exposed_surface):
    result = diagnostic.assess_frames([], [255, 0, 0], exposed_surface, 0)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'event-dispatch-not-confirmed'


def test_missing_context_has_distinct_classification(exposed_surface):
    exposed_surface['context_valid'] = False
    result = diagnostic.assess_frames([], [255, 0, 0], exposed_surface, 5)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'context-unavailable'


def test_wrong_context_frame_never_becomes_success_from_later_correct_pixel(exposed_surface):
    wrong_context = observed_frame([0, 0, 0, 255])
    wrong_context['context_matches'] = False
    frames = [wrong_context, observed_frame([255, 0, 0, 255], timestamp=1.1)]
    result = diagnostic.assess_frames(frames, [255, 0, 0], exposed_surface, 3)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'wrong-current-context-during-paint'
    assert result['last_frame']['pixel'] == [255, 0, 0, 255]


def test_invalid_sampling_does_not_report_a_renderer_colour_failure(exposed_surface):
    frame = observed_frame([0, 0, 0, 255])
    frame['sampling_valid'] = False
    frame['viewport'] = [0, 0, 0, 0]
    result = diagnostic.assess_frames([frame], [255, 0, 0], exposed_surface, 2)
    assert result['status'] == 'UNAVAILABLE'
    assert result['classification'] == 'sampling-geometry-or-fbo-inconsistent'
    assert result['last_frame']['viewport'] == [0, 0, 0, 0]


def test_observation_exception_is_retained_without_false_colour_diagnosis(exposed_surface):
    frame = observed_frame(None)
    frame['observation_error'] = 'glReadPixels failed while observing the frame'
    result = diagnostic.assess_frames([frame], [255, 0, 0], exposed_surface, 2)
    assert result['status'] == 'UNAVAILABLE'
    assert result['classification'] == 'observation-failed'
    assert result['last_frame']['observation_error'] == frame['observation_error']


@pytest.mark.parametrize(
    'pixel, expected, matches',
    [
        ([255, 0, 0, 255], [255, 0, 0], True),
        ([235, 20, 20, 255], [255, 0, 0], True),
        ([0, 0, 0, 255], [255, 0, 0], False),
        ([255, 0, 0], [255, 0, 0], False),
        (None, [255, 0, 0], False),
    ],
)
def test_pixel_oracle_requires_complete_observation(pixel, expected, matches):
    assert diagnostic.pixel_matches(pixel, expected) is matches


def test_unknown_surface_state_is_inconclusive():
    surface = {'state_error': 'Qt object was deleted during surface observation'}
    result = diagnostic.assess_frames([], [255, 0, 0], surface, 3)
    assert result['status'] == 'UNAVAILABLE'
    assert result['final_surface']['state_error'] == surface['state_error']


def test_unexpected_transform_invalidates_color_oracle(exposed_surface):
    frame = observed_frame([0, 0, 0, 255])
    frame['expected_transform'] = {'pan': [0.0, 0.0], 'zoom': [1.0, 1.0]}
    frame['transform'] = {'pan': [0.2, 0.0], 'zoom': [1.5, 1.5]}
    result = diagnostic.assess_frames([frame], [255, 0, 0], exposed_surface, 5)
    assert result['status'] == 'UNAVAILABLE'
    assert result['classification'] == 'unexpected-view-transform'
    assert result['last_frame']['transform'] == frame['transform']


def test_missing_transform_is_not_a_renderer_color_failure(exposed_surface):
    frame = observed_frame([0, 0, 0, 255])
    frame['expected_transform'] = {'pan': [0.0, 0.0], 'zoom': [1.0, 1.0]}
    frame['transform'] = None
    result = diagnostic.assess_frames([frame], [255, 0, 0], exposed_surface, 5)
    assert result['status'] == 'UNAVAILABLE'


@pytest.mark.skipif(importlib.util.find_spec('PyQt5') is None, reason='Qt not installed')
@pytest.mark.parametrize('mode, deliveries', [('observe', 1), ('block', 0)])
def test_input_control_and_key_privacy(mode, deliveries):
    code = f"""import importlib.util
from PyQt5.QtCore import QObject, QEvent, QCoreApplication, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication
spec=importlib.util.spec_from_file_location('kit', {str(SCRIPT)!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
app=QApplication(['test']); app._diagnostic_cleanups=[]
app.setProperty('diagnosticInputMode', {mode!r})
class Receiver(QObject):
    deliveries=0
    def event(self, event):
        if event.type()==QEvent.KeyPress: self.deliveries+=1
        return True
receiver=Receiver(); m.watch_surface(receiver, 'synthetic')
QCoreApplication.sendEvent(receiver, QKeyEvent(QEvent.KeyPress, 65, Qt.NoModifier, 'PRIVATE_KEY_TEXT'))
assert receiver.deliveries=={deliveries}
for cleanup in app._diagnostic_cleanups: cleanup()
"""
    result, log = diagnostic.run_process(
        [sys.executable, '-c', code], dict(os.environ, QT_QPA_PLATFORM='offscreen'), 10
    )
    assert result['returncode'] == 0, log
    assert 'input-event' in log
    assert 'PRIVATE_KEY_TEXT' not in log


def test_child_preserves_structured_inconclusive_assessment():
    import os
    import sys

    assessment = {
        'status': 'UNAVAILABLE',
        'classification': 'sampling-geometry-or-fbo-inconsistent',
        'expected_rgb': [255, 0, 0],
        'first_frame': {'sampling_valid': False},
    }
    code = f"""import importlib.util, sys
spec = importlib.util.spec_from_file_location('kit', {str(SCRIPT)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assessment = {assessment!r}
error_class = getattr(m, 'ObservationError', None) or m.ObservationFailure
def broken_observation():
    raise error_class(assessment)
m.inventory = broken_observation
sys.exit(m.main(['--_probe', 'inventory']))
"""
    result, log = diagnostic.run_process([sys.executable, '-c', code], os.environ.copy(), 5)
    assert result['returncode'] == 0, log
    assert result['status'] == 'UNAVAILABLE'
    assert result['details']['assessment'] == assessment


@pytest.mark.skipif(
    os.environ.get('PHY_DIAGNOSTIC_GL_TESTS') != '1'
    or os.environ.get('QT_QPA_PLATFORM') in ('offscreen', 'minimal'),
    reason='Opt-in validation against a real graphical session',
)
@pytest.mark.parametrize(
    'control,mode,expected,classification',
    [
        ('color', 'observe', 'FAIL', 'final-observed-frame-mismatch'),
        ('wheel', 'observe', 'UNAVAILABLE', 'unexpected-view-transform'),
        ('wheel', 'block', 'PASS', None),
        ('hide', 'observe', 'UNAVAILABLE', 'surface-not-exposed'),
    ],
)
def test_real_gl_controlled_faults(tmp_path, control, mode, expected, classification):
    helper = Path(__file__).with_name('_graphics_controls.py')
    result, log = diagnostic.run_process(
        [sys.executable, str(helper), str(SCRIPT), control, mode, str(tmp_path)],
        os.environ.copy(),
        20,
    )
    assert result['status'] == expected, log[-4000:]
    assert any(e['event'] == 'injected-control' for e in result['events'])
    if classification:
        assert classification in result['details']['reason']
    if control == 'hide':
        assessment = result['details']['assessment']['canvases'][0]
        assert assessment['heartbeat_ticks'] > 0
        assert assessment['final_surface']['owners']['dock']['visible'] is False
    # Every announced image must still match its saved bytes after later phases.
    import hashlib

    for event in result['events']:
        if event['event'] == 'artifact' and event.get('sha256'):
            assert (
                hashlib.sha256((tmp_path / event['path']).read_bytes()).hexdigest()
                == event['sha256']
            )


@pytest.mark.skipif(
    os.environ.get('PHY_DIAGNOSTIC_GL_TESTS') != '1'
    or os.environ.get('QT_QPA_PLATFORM') in ('offscreen', 'minimal'),
    reason='Opt-in validation against a real graphical session',
)
@pytest.mark.parametrize('suppress_redraw', [False, True])
@pytest.mark.parametrize('capture_images', [False, True])
def test_scene_requires_selection_redraw_without_capture_request(
    tmp_path,
    suppress_redraw,
    capture_images,
):
    helper = Path(__file__).with_name('_graphics_controls.py')
    if suppress_redraw:
        control = 'scene-redraw' if capture_images else 'scene-redraw-metrics'
        command = [sys.executable, str(helper), str(SCRIPT), control, 'observe', str(tmp_path)]
    else:
        command = [sys.executable, str(SCRIPT), '--_probe', 'scene']
        if capture_images:
            command += ['--_artifacts', str(tmp_path)]
    result, log = diagnostic.run_process(command, os.environ.copy(), 20)
    if suppress_redraw:
        assert result['status'] == 'FAIL', log[-4000:]
        assert 'redraw-suppressed' in log
        stage = (
            'scene framebuffer capture' if capture_images else 'selection-triggered scene repaint'
        )
        assert 'Timed out waiting for ' + stage in result['details']['reason']
    else:
        assert result['status'] == 'PASS', log[-4000:]
        if not capture_images:
            assert result['details']['selections'] == [[0], [2]]
            return
        selections = [
            e
            for e in result['events']
            if e['event'] == 'scene_capture' and e.get('stage') in ('selected-0', 'clicked-2')
        ]
        assert len(selections) == 4
        assert all(e['redraw_source'] == 'selection' for e in selections)
        for view in ('waveform', 'correlogram'):
            frames = [e for e in selections if e['view'] == view]
            assert [e['selected_clusters'] for e in frames] == [[0], [2]]
            assert frames[0]['sha256'] != frames[1]['sha256']


@pytest.mark.skipif(
    os.environ.get('PHY_DIAGNOSTIC_GL_TESTS') != '1'
    or os.environ.get('QT_QPA_PLATFORM') in ('offscreen', 'minimal'),
    reason='Opt-in validation against a real graphical session',
)
@pytest.mark.parametrize('fail_scene', [False, True])
def test_scene_destroys_native_windows_before_child_returns(tmp_path, fail_scene):
    code = f"""import importlib.util
from PyQt5 import sip
from PyQt5.QtWidgets import QApplication
spec = importlib.util.spec_from_file_location('kit', {str(SCRIPT)!r})
kit = importlib.util.module_from_spec(spec); spec.loader.exec_module(kit)
canvases = []
original_watch, original_emit = kit.watch_surface, kit.emit
def watch(canvas, label):
    canvases.append(canvas)
    original_watch(canvas, label)
def emit(event, **fields):
    original_emit(event, **fields)
    if {fail_scene!r} and event == 'stage' and fields.get('stage') == 'scene-qt-table-click':
        raise RuntimeError('injected scene failure')
kit.watch_surface, kit.emit = watch, emit
status = kit.main(['--_probe', 'scene', '--_artifacts', {str(tmp_path)!r}])
assert status == {1 if fail_scene else 0}
assert len(canvases) == 2
assert QApplication.instance() is not None
assert all(sip.isdeleted(canvas) for canvas in canvases), 'native canvases survived cleanup'
assert all(sip.isdeleted(canvas.diagnostic_main_window) for canvas in canvases)
print('native scene windows destroyed before child returns', flush=True)
"""
    result, log = diagnostic.run_process([sys.executable, '-c', code], os.environ.copy(), 20)
    assert result['returncode'] == 0, log[-4000:]
    assert 'native scene windows destroyed before child returns' in log
    assert result['details']['status'] == ('FAIL' if fail_scene else 'PASS')


@pytest.mark.skipif(
    not sys.platform.startswith('linux')
    or os.environ.get('PHY_DIAGNOSTIC_GL_TESTS') != '1'
    or os.environ.get('QT_QPA_PLATFORM') in ('offscreen', 'minimal'),
    reason='Opt-in Linux desktop GLX/EGL comparison',
)
@pytest.mark.parametrize('binding', ['egl', 'glx'])
def test_xcb_glx_requires_matching_pyopengl_binding(binding):
    env = dict(
        os.environ,
        QT_QPA_PLATFORM='xcb',
        QT_XCB_GL_INTEGRATION='xcb_glx',
        PYOPENGL_PLATFORM=binding,
    )
    result, log = diagnostic.run_process(
        [sys.executable, str(SCRIPT), '--_probe', 'phy'],
        env,
        20,
    )
    if binding == 'glx':
        assert result['status'] == 'PASS', log[-4000:]
    else:
        assert result['status'] == 'FAIL', log[-4000:]
        assert 'Attempt to retrieve context when no valid context' in log
        assert 'candidate workaround' in log
        frames = [e for e in result['events'] if e['event'] == 'frame']
        assert frames and frames[0]['surface']['context_valid']


def test_capture_encoding_time_is_not_a_quiet_dispatch_interval():
    frame = observed_frame([255, 0, 0, 255], timestamp=0.0)
    frame.update(observation_completed_at=1.0, heartbeat_at_completion=5)
    # Encoding was slow, but no Qt dispatch after capture is yet established.
    assert not diagnostic.frame_has_settled(frame, [255, 0, 0], 2.0, 5)
    assert not diagnostic.frame_has_settled(frame, [255, 0, 0], 2.0, 6)
    frame['post_capture_dispatch_at'] = 2.0
    assert not diagnostic.frame_has_settled(frame, [255, 0, 0], 2.0, 6)
    assert diagnostic.frame_has_settled(frame, [255, 0, 0], 2.11, 7)


def test_late_frame_during_post_capture_dispatch_blocks_success(exposed_surface):
    first = observed_frame([255, 0, 0, 255], timestamp=0.0)
    first.update(
        observation_completed_at=1.0, heartbeat_at_completion=5, post_capture_dispatch_at=1.01
    )
    assert not diagnostic.frame_has_settled(first, [255, 0, 0], 1.02, 6)
    late = observed_frame([0, 0, 0, 255], timestamp=1.05)
    late.update(
        observation_completed_at=1.05, heartbeat_at_completion=6, post_capture_dispatch_at=1.07
    )
    assert not diagnostic.frame_has_settled(late, [255, 0, 0], 1.3, 10)
    result = diagnostic.assess_frames([first, late], [255, 0, 0], exposed_surface, 5)
    assert result['status'] == 'FAIL'
    assert result['classification'] == 'final-observed-frame-mismatch'


def test_last_successful_png_is_not_assigned_to_later_observation_error():
    frames = [
        dict(observed_frame([255, 0, 0, 255]), frame_id='0:1', image='first.png'),
        dict(observed_frame([255, 0, 0, 255], timestamp=1.1), frame_id='0:2'),
        dict(
            observed_frame(None, timestamp=1.2),
            frame_id='0:3',
            observation_error='Read failed during this later paint',
        ),
    ]
    assert diagnostic.attach_frame_capture(frames, {'frame_id': '0:2', 'path': 'last.png'})
    assert frames[0]['image'] == 'first.png'
    assert frames[1]['image'] == 'last.png'
    assert 'image' not in frames[2]


@pytest.mark.parametrize(
    'metadata',
    [
        {'frame_id': 'unknown', 'path': 'last.png'},
        {'path': 'last.png'},
        {'frame_id': '0:1', 'path': 'last.png'},
    ],
)
def test_png_link_requires_retained_successful_exact_frame(metadata):
    frames = [dict(observed_frame(None), frame_id='0:1', observation_error='Read failed')]
    original = copy.deepcopy(frames)
    assert not diagnostic.attach_frame_capture(frames, metadata)
    assert frames == original


@pytest.mark.parametrize('failure_status', ['FAIL', 'TIMEOUT'])
@pytest.mark.parametrize('failed_scenario', ['all', 'float-redock'])
def test_historical_or_targeted_failure_survives_later_full_pass(
    failure_status,
    failed_scenario,
):
    failed = {
        'attempt_id': 'run-earlier',
        'configuration': 'baseline',
        'probe': 'docked',
        'status': failure_status,
        'scenario': failed_scenario,
        'last_stage': 'redocking',
        'log_path': 'runs/run-earlier/probe.log',
    }
    results = {
        'script_sha256': 'test',
        'python': 'test',
        'session': 'test',
        'symptoms': '',
        'configurations': [{'name': 'baseline', 'overrides': {}}],
        'probes': [failed]
        + [
            {
                'configuration': 'baseline',
                'probe': kind,
                'status': 'PASS',
                'scenario': 'all',
                'last_stage': 'done',
            }
            for kind in ('qt', 'native', 'phy', 'docked', 'scene')
        ],
    }
    summary = diagnostic._summary_results(results)
    assert len(summary['retained_failures']) == 1
    report = diagnostic.make_report(summary)
    assert 'Baseline synthetic rendering passed.' not in report
    assert 'No application workaround is recommended' in report
    assert 'runs/run-earlier/probe.log' in report
    assert failure_status in report
    assert 'env phy template-gui' not in report


def test_failing_baseline_permits_only_consistently_passing_alternate_candidate():
    results = {
        'script_sha256': 'test',
        'python': 'test',
        'session': 'test',
        'symptoms': '',
        'configurations': [
            {'name': 'baseline', 'overrides': {}},
            {
                'name': 'xcb-software',
                'overrides': {'QT_QPA_PLATFORM': 'xcb', 'LIBGL_ALWAYS_SOFTWARE': 'true'},
            },
        ],
        'probes': [
            {
                'attempt_id': 'baseline-failure',
                'configuration': 'baseline',
                'probe': 'docked',
                'status': 'FAIL',
                'scenario': 'all',
                'last_stage': 'first draw',
                'log_path': 'runs/baseline-failure/probe.log',
            }
        ]
        + [
            {
                'configuration': config,
                'probe': kind,
                'status': 'PASS',
                'scenario': 'all',
                'last_stage': 'done',
            }
            for config in ('baseline', 'xcb-software')
            for kind in ('qt', 'native', 'phy', 'docked', 'scene')
        ],
    }
    report = diagnostic.make_report(diagnostic._summary_results(results))
    assert 'Failures or timeouts remain recorded' in report
    assert 'tentative application-verification candidates' in report
    assert 'runs/baseline-failure/probe.log' in report
    assert report.count('```sh') == 1
    assert 'env QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=true phy template-gui' in report
    assert 'env phy template-gui' not in report
    assert 'Baseline synthetic rendering passed.' not in report
