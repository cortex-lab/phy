"""Exercise SpikeInterface recordings through the application boundary."""

import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from phylib.io import traces as traces_module
from phylib.io.model import load_model
from phylib.utils import Bunch
from pytest import fixture, importorskip, mark, skip
from scipy.signal import butter, lfilter

from phy.apps.template.gui import TemplateController
from phy.apps.trace.gui import create_trace_gui
from phy.gui import GUI


@fixture
def si_dataset(tmp_path):
    if not hasattr(traces_module, 'SpikeInterfaceEphysReader'):
        skip('Requires phylib source with SpikeInterface support')
    si = importorskip('spikeinterface')
    from spikeinterface.preprocessing import astype, common_reference

    rng = np.random.default_rng(12)
    source = tmp_path / 'source'
    source.mkdir()
    raw = rng.integers(-1000, 1000, (2345, 7), dtype=np.int16)
    raw.tofile(source / 'traces.raw')
    recording = si.read_binary(
        source / 'traces.raw', sampling_frequency=1000.0, dtype='int16', num_channels=7
    )
    recording = common_reference(astype(recording, 'float32'), reference='global')
    recording.dump_to_json(source / 'recording.json', relative_to=source)
    params = tmp_path / 'params.py'
    params.write_text(
        "dat_path = 'source/recording.json'\nsample_rate = 1000.\n"
        "n_channels_dat = 7\ndtype = 'float32'\noffset = 0\n"
    )
    mapping = np.array([6, 0, 3])
    samples = np.array([2, 499, 1000, 2343])
    np.save(tmp_path / 'channel_map.npy', mapping)
    np.save(tmp_path / 'channel_positions.npy', [[0, 0], [0, 20], [0, 40]])
    np.save(tmp_path / 'spike_times.npy', samples)
    np.save(tmp_path / 'spike_templates.npy', np.array([0, 1, 0, 1], dtype=np.int32))
    np.save(tmp_path / 'templates.npy', rng.normal(size=(2, 82, 3)).astype('float32'))
    return Bunch(params=params, recording=recording, mapping=mapping, samples=samples)


@mark.parametrize('filter_name', ['raw', 'high_pass'])
def test_spikeinterface_template_views(qtbot, tmp_path, monkeypatch, si_dataset, filter_name):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    model = load_model(si_dataset.params)
    controller = TemplateController(
        model=model,
        dir_path=tmp_path,
        config_dir=tmp_path / 'config',
        plugin_dirs=[],
        clear_state=True,
        enable_threading=False,
    )
    gui = GUI(config_dir=tmp_path / 'config')
    views = []
    try:
        controller.raw_data_filter.set(filter_name)
        expected = si_dataset.recording.get_traces()[:, si_dataset.mapping]
        interval = (0.48, 0.53)
        trace_expected = expected[480:530].copy()
        trace_expected -= np.median(trace_expected, axis=0)
        if filter_name == 'high_pass':
            b, a = butter(3, 0.3, 'high')
            trace_expected = lfilter(b, a, trace_expected, axis=0)
            trace_expected = lfilter(b, a, trace_expected[::-1], axis=0)[::-1]
        assert_allclose(controller._get_traces(interval).data, trace_expected)
        assert_array_equal(controller._get_channel_labels(), si_dataset.mapping.astype(str))
        waveform = controller._get_waveforms(0)
        padded = np.pad(expected[:, waveform.channel_ids], ((41, 41), (0, 0)))
        wave_expected = np.stack(
            [
                padded[si_dataset.samples[i] : si_dataset.samples[i] + 82]
                for i in waveform.spike_ids
            ]
        )
        wave_expected -= np.median(wave_expected, axis=1)[:, None, :]
        if filter_name == 'high_pass':
            wave_expected = lfilter(b, a, wave_expected, axis=1)
            wave_expected = lfilter(b, a, wave_expected[:, ::-1], axis=1)[:, ::-1]
        assert_allclose(waveform.data, wave_expected)
        assert_array_equal(
            waveform.channel_positions, model.channel_positions[waveform.channel_ids]
        )
        for view in (controller.create_trace_view(), controller.create_waveform_view()):
            views.append(view)
            view.attach(gui)
            view.on_select(cluster_ids=[0])
        views[0].set_interval(interval)
    finally:
        gui.close()
        for view in views:
            view.canvas.close()
        controller.close()
    assert not model.traces._cache


@mark.parametrize('kind', ['json', 'zarr'])
def test_spikeinterface_trace_gui(qtbot, tmp_path, si_dataset, kind):
    path = si_dataset.params.parent / 'source' / 'recording.json'
    if kind == 'zarr':
        importorskip('zarr')
        path = tmp_path / 'recording.zarr'
        si_dataset.recording.save(folder=path, format='zarr')
    gui = create_trace_gui(str(path))
    try:
        assert_array_equal(gui._trace_reader[480:530], si_dataset.recording.get_traces()[480:530])
    finally:
        gui.close()
    assert not gui._trace_reader._cache


def test_binary_loading_does_not_import_spikeinterface(tmp_path):
    path = tmp_path / 'recording.bin'
    np.zeros((100, 2), dtype='int16').tofile(path)
    subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; from phylib.io.traces import get_ephys_reader; '
            'reader = get_ephys_reader(sys.argv[1], sample_rate=1000., '
            'n_channels_dat=2, dtype="int16"); reader[:10]; '
            'assert "spikeinterface" not in sys.modules',
            str(path),
        ],
        check=True,
    )


@mark.parametrize('view_name', ['create_trace_view', 'create_trace_image_view'])
def test_spikeinterface_explicit_navigation(qtbot, tmp_path, monkeypatch, si_dataset, view_name):
    import threading

    model = load_model(si_dataset.params)
    controller = TemplateController(
        model=model,
        dir_path=tmp_path,
        config_dir=tmp_path / 'config',
        plugin_dirs=[],
        clear_state=True,
        enable_threading=False,
    )
    reader = model.traces
    hints = []
    prefetch = reader.prefetch

    def hint(sample):
        hints.append(sample)
        prefetch(sample)

    monkeypatch.setattr(reader, 'prefetch', hint)
    view = getattr(controller, view_name)()
    started, release = threading.Event(), threading.Event()
    worker = None
    try:
        view.set_interval((0.1, 0.35))
        view.go_right()
        assert hints == []
        view.toggle_background_prefetch(True)
        view.go_right()
        assert hints == [899]
        qtbot.waitUntil(lambda: 1 in reader._cache, timeout=5000)
        controller._get_waveforms(0)
        assert hints == [899]  # Waveform demand never emits navigation hints.
        get_traces = reader.recording.get_traces

        def blocked_get_traces(**kwargs):
            if kwargs['start_frame'] == 2000:
                started.set()
                assert release.wait(5)
            return get_traces(**kwargs)

        monkeypatch.setattr(reader.recording, 'get_traces', blocked_get_traces)
        view.set_interval((1.5, 1.75))
        view.go_right()
        assert started.wait(5)
        worker = reader._blocks._worker
        controller.close()
        assert reader._blocks._closed
        assert not reader._cache
    finally:
        release.set()
        if worker is not None:
            worker.join(5)
            assert not worker.is_alive()
        view.canvas.close()
        controller.close()
