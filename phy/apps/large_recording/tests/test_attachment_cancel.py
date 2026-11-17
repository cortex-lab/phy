"""Late cancellation preserves complete artifacts and the prior attachment."""

from concurrent.futures import CancelledError
from contextlib import contextmanager

import pytest

from phy.apps.large_recording import range_index, summary_store
from phy.apps.large_recording.live_summaries import LiveSummaryEngine
from phy.apps.large_recording.range_build import build_amplitude_index
from phy.apps.large_recording.tests.test_live_summaries import package


@pytest.mark.parametrize('kind', ['rows', 'index'])
def test_builder_cancellation_reaches_attachment_lock(tmp_path, monkeypatch, kind):
    _, bounded = package(tmp_path)
    if kind == 'index':
        summary_store.build_summary_store(bounded, tmp_path / 'rows')
    reference = summary_store.summary_store_reference(bounded)
    stopped = False

    @contextmanager
    def waiting_lock(path, check_cancel=None):
        nonlocal stopped
        stopped = True
        assert check_cancel is not None
        check_cancel()
        pytest.fail('Attachment lock did not honor cancellation.')
        yield

    module = summary_store if kind == 'rows' else range_index
    monkeypatch.setattr(module, '_attachment_lock', waiting_lock)
    output = tmp_path / 'completed'
    builder = summary_store.build_summary_store if kind == 'rows' else build_amplitude_index
    with pytest.raises(CancelledError):
        builder(bounded, output, cancel=lambda: stopped)
    assert output.is_dir()
    assert summary_store.summary_store_reference(bounded) == reference
    # Publication completed before attachment: the artifact remains usable.
    with LiveSummaryEngine(bounded) as engine:
        if kind == 'rows':
            with summary_store.SummaryStore(engine, path=output) as store:
                assert set(store.index['units']) == set(map(str, engine.manifest['unit_ids']))
        else:
            index = range_index.AmplitudeRangeIndex(engine, path=output)
            index.close()
