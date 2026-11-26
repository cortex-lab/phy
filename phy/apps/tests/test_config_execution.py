"""Explicit configuration execution reaches every params.py loading command."""

from contextlib import nullcontext
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from phylib.io.model import get_template_params

import phy.apps as apps
from phy.apps._config import _config_execution_kwargs
from phy.apps.template import gui as template_gui
from phy.apps.trace import gui as trace_gui


@pytest.mark.parametrize('trusted', [False, True])
@pytest.mark.parametrize('command', ['template-gui', 'template-describe', 'trace-gui'])
def test_cli_execution_option(tmp_path, monkeypatch, command, trusted):
    path = tmp_path / 'params.py'
    path.touch()
    calls = []
    module = trace_gui if command == 'trace-gui' else template_gui
    name = command.replace('-', '_')
    monkeypatch.setattr(module, name, lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(apps, 'capture_exceptions', nullcontext)
    result = CliRunner().invoke(
        apps.phycli, [command, str(path)] + (['--allow-python-execution'] if trusted else [])
    )
    assert result.exit_code == 0, result.output
    assert calls[0][0] == (str(path),)
    assert calls[0][1]['allow_python_execution'] is trusted


def test_profiled_template_forwards_option(tmp_path, monkeypatch):
    import builtins

    import phy.utils.profiling

    path = tmp_path / 'params.py'
    path.touch()
    calls = []
    monkeypatch.setattr(builtins, 'profile', object(), raising=False)
    monkeypatch.setattr(apps, 'capture_exceptions', nullcontext)
    monkeypatch.setattr(template_gui, 'template_gui', lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        phy.utils.profiling, '_profile', lambda prof, code, g, loc: eval(code, g, loc)
    )
    result = CliRunner().invoke(
        apps.phycli, ['template-gui', str(path), '--allow-python-execution']
    )
    assert result.exit_code == 0, result.output
    assert calls[0][1]['allow_python_execution'] is True


@pytest.mark.parametrize('trusted', [False, True])
@pytest.mark.parametrize('command', ['alf-convert', 'extract-waveforms'])
@pytest.mark.parametrize('n_probes', [1, 2])
def test_cli_model_and_merger_execution_option(tmp_path, monkeypatch, command, trusted, n_probes):
    import phylib.io.alf
    import phylib.io.merge
    import phylib.io.model

    path = tmp_path / 'params.py'
    path.touch()
    calls = []
    model = SimpleNamespace(save_spikes_subset_waveforms=lambda **kw: None, close=lambda: None)

    def load_model(path, *, allow_python_execution=False):
        calls.append(('load', allow_python_execution))
        return model

    class Merger:
        def __init__(self, *args, allow_python_execution=False):
            calls.append(('merge', allow_python_execution))

        def merge(self):
            return model

    monkeypatch.setattr(phylib.io.model, 'load_model', load_model)
    monkeypatch.setattr(phylib.io.merge, 'Merger', Merger)

    def convert(path, *, allow_python_execution=False):
        calls.append(('convert', allow_python_execution))

    monkeypatch.setattr(
        phylib.io.alf, 'EphysAlfCreator', lambda model: SimpleNamespace(convert=convert)
    )
    args = [command]
    if command == 'alf-convert':
        args += [str(tmp_path)] * n_probes + [str(tmp_path / 'alf')]
    else:
        args += [str(path)]
    if trusted:
        args.append('--allow-python-execution')
    result = CliRunner().invoke(apps.phycli, args)
    assert result.exit_code == 0, result.output
    expected = 'merge' if command == 'alf-convert' and n_probes == 2 else 'load'
    expected_calls = [(expected, trusted)]
    if command == 'alf-convert':
        expected_calls.append(('convert', trusted))
    assert calls == expected_calls


def test_old_phylib_has_no_trusted_fallback():
    calls = []

    def old_loader(path):
        calls.append(path)

    assert _config_execution_kwargs(old_loader, False) == {}
    with pytest.raises(RuntimeError, match='No fallback was attempted'):
        _config_execution_kwargs(old_loader, True)
    assert calls == []


@pytest.fixture
def hardened_config():
    if 'allow_python_execution' not in signature(get_template_params).parameters:
        pytest.skip('Requires phylib with the trusted configuration API')


@pytest.mark.parametrize('trusted', [False, True])
def test_template_loading_execution_is_explicit(tmp_path, monkeypatch, hardened_config, trusted):
    from phylib.io.tests.conftest import _make_dataset

    path = _make_dataset(tmp_path, param='dense', has_spike_attributes=False)
    marker = tmp_path / 'executed'
    path.write_text(
        path.read_text() + f"\nimport pathlib\npathlib.Path({str(marker)!r}).write_text('yes')\n"
    )

    # Exercise the real Template GUI loader, then stop before opening a GUI.
    class LoadedError(Exception):
        pass

    def loaded():
        raise LoadedError

    models = []
    original_load_model = template_gui.load_model

    def track_model(*args, **kwargs):
        model = original_load_model(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(template_gui, 'load_model', track_model)
    # Keep signature inspection identical to the real loader.
    track_model.__signature__ = signature(original_load_model)
    monkeypatch.setattr(template_gui, 'create_app', loaded)
    if trusted:
        try:
            with pytest.raises(LoadedError):
                template_gui.template_gui(path, allow_python_execution=True)
        finally:
            for model in models:
                model.close()
        assert marker.read_text() == 'yes'
    else:
        with pytest.raises(ValueError, match=r'params\.py:\d+'):
            template_gui.template_gui(path)
        assert not marker.exists()


def test_template_describe_trusted(tmp_path, hardened_config):
    from phylib.io.tests.conftest import _make_dataset

    path = _make_dataset(tmp_path, param='dense', has_spike_attributes=False)
    marker = tmp_path / 'described'
    path.write_text(
        path.read_text() + f"\nimport pathlib\npathlib.Path({str(marker)!r}).write_text('yes')\n"
    )
    with pytest.raises(ValueError):
        template_gui.template_describe(path)
    assert not marker.exists()
    template_gui.template_describe(path, allow_python_execution=True)
    assert marker.exists()


def test_trace_loading_execution_is_explicit(tmp_path, qtbot, hardened_config):
    expected = np.arange(1000, dtype='int16').reshape(500, 2)
    expected.tofile(tmp_path / 'recording.bin')
    marker = tmp_path / 'traced'
    path = tmp_path / 'params.py'
    path.write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('yes')\n"
        "dat_path = 'recording.bin'\nsample_rate = 1000.\n"
        "n_channels_dat = 2\ndtype = 'int16'\noffset = 0\n"
    )
    with pytest.raises(ValueError):
        trace_gui.create_trace_gui(path)
    assert not marker.exists()
    gui = trace_gui.create_trace_gui(path, allow_python_execution=True)
    try:
        np.testing.assert_array_equal(gui._trace_reader[:], expected)
        assert marker.exists()
    finally:
        gui.close()
