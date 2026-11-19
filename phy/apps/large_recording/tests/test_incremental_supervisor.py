"""Native history and derived membership caches follow semantic transition scope."""

import numpy as np
import pytest
from phylib.utils import connect, unconnect

from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.apps.large_recording.tests.test_curation import make_session


class NoCopy(np.ndarray):
    def copy(self, *args, **kwargs):
        pytest.fail('An interactive command copied the complete spike assignment array.')


def test_thousand_event_label_and_undo_skip_spike_and_membership_work(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation, live_summaries, runtime
    from phy.apps.large_recording.supervisor import BoundedSupervisor
    from phy.cluster._propositions import MergePropositionController, decode_curation_mapping

    source, session = make_session(tmp_path)
    try:
        for i in range(1000):
            session.record_label('group', [0], 'good' if i % 2 else 'mua')
        supervisor = BoundedSupervisor(
            spike_clusters=session.assignments.copy(),
            session=session,
            cluster_labels=session.metadata,
        )
        mapping = {
            'format_version': '2',
            'unit_ids': session.unit_ids,
            'merges': [{'unit_ids': [0, 1]}],
        }
        supervisor.merge_propositions = MergePropositionController(
            decode_curation_mapping(mapping)
        )
        supervisor._project_proposition_state()
        session.restored_parent_ids  # Warm the one membership projection.
        revision = session.membership_revision
        supervisor.clustering._spike_clusters = supervisor.clustering._spike_clusters.view(NoCopy)
        session.assignments = session.assignments.view(NoCopy)
        original_equal = np.array_equal
        original_interpreter = runtime.interpret_events

        def forbid(*args, **kwargs):
            pytest.fail('A metadata-only action replayed or rebuilt membership state.')

        def compare(a, b, *args, **kwargs):
            assert np.shape(a) != session.assignments.shape, 'Full assignment scan during label.'
            return original_equal(a, b, *args, **kwargs)

        def interpret(units, metadata, events):
            assert len(events) == 1
            return original_interpreter(units, metadata, events)

        monkeypatch.setattr(curation, 'project_assignments', forbid)
        monkeypatch.setattr(live_summaries, 'membership_expressions', forbid)
        monkeypatch.setattr(supervisor, '_reassembled_results', forbid)
        monkeypatch.setattr(session, 'snapshot', forbid)
        monkeypatch.setattr(runtime, 'interpret_events', interpret)
        monkeypatch.setattr(np, 'array_equal', compare)
        supervisor.label('group', 'noise', [0])
        assert session.metadata['group'][0] == 'noise'
        assert session.membership_revision is revision
        supervisor.undo()
        assert session.metadata['group'][0] == 'good'
        assert session.membership_revision is revision
        supervisor.redo()
        assert session.metadata['group'][0] == 'noise'
        assert session.membership_revision is revision
    finally:
        session.close()
        source.close()


def test_bounded_native_undo_uses_local_patch(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    _, controller = make_controller(tmp_path)
    try:
        supervisor = controller.supervisor
        original = supervisor.clustering.spike_clusters.copy()
        supervisor.merge([0, 1], to=100)
        supervisor.clustering._spike_clusters_base = (
            supervisor.clustering._spike_clusters_base.view(NoCopy)
        )

        def forbid(*args, **kwargs):
            pytest.fail('Cached native undo replayed semantic assignments.')

        monkeypatch.setattr(curation, 'project_assignments', forbid)
        supervisor.undo()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, original)
        supervisor.redo()
        np.testing.assert_array_equal(
            supervisor.clustering.spike_clusters, supervisor.session.assignments
        )
    finally:
        controller.close()


def test_label_keeps_controller_membership_metric_caches(tmp_path, monkeypatch):
    from phy.apps.large_recording import live_summaries

    _, controller = make_controller(tmp_path)
    try:
        full_count = controller.get_full_count(0)
        waveform_count = controller.get_waveform_count(0)
        lineage = controller._lineage_cache
        waveforms = controller._waveform_membership_cache

        def forbid(*args, **kwargs):
            pytest.fail('A label invalidated a membership metric cache.')

        monkeypatch.setattr(live_summaries, 'membership_expressions', forbid)
        controller.supervisor.label('group', 'mua', [0])
        assert controller.get_full_count(0) == full_count
        assert controller.get_waveform_count(0) == waveform_count
        assert controller._lineage_cache is lineage
        assert controller._waveform_membership_cache is waveforms
        controller.supervisor.undo()
        assert controller.get_full_count(0) == full_count
        assert controller.get_waveform_count(0) == waveform_count
        assert controller._lineage_cache is lineage
        assert controller._waveform_membership_cache is waveforms
    finally:
        controller.close()


@pytest.mark.parametrize('operation', ['merge', 'split', 'undo', 'redo', 'restore'])
def test_observer_cannot_write_unrelated_public_assignment_rows(tmp_path, operation):
    _, controller = make_controller(tmp_path)
    supervisor = controller.supervisor
    if operation in ('undo', 'redo', 'restore'):
        supervisor.merge([0, 1], to=100)
    if operation == 'redo':
        supervisor.undo()
    before = supervisor.clustering.spike_clusters.copy()
    before_events = supervisor.session.events
    before_history = supervisor._global_history.current_position
    before_groups = dict(supervisor.clustering.spikes_per_cluster)
    before_revision = supervisor.session.membership_revision

    @connect(event='cluster', sender=supervisor.clustering)
    def mutate(sender, up):
        unrelated = sender.spikes_per_cluster[2][0]
        sender.spike_clusters[unrelated] = 99

    try:
        with pytest.raises(ValueError, match='read-only'):
            if operation == 'merge':
                supervisor.merge([0, 1])
            elif operation == 'split':
                supervisor.split_source_amplitude(0, 4)
            elif operation == 'restore':
                supervisor.restore_parents(100)
            else:
                getattr(supervisor, operation)()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
        np.testing.assert_array_equal(supervisor.session.assignments, before)
        assert supervisor.session.events == before_events
        assert supervisor._global_history.current_position == before_history
        assert supervisor.session.membership_revision is before_revision
        assert set(supervisor.clustering.spikes_per_cluster) == set(before_groups)
        for cluster, rows in before_groups.items():
            np.testing.assert_array_equal(supervisor.clustering.spikes_per_cluster[cluster], rows)
            np.testing.assert_array_equal(supervisor.session._assignment_groups[cluster], rows)
    finally:
        unconnect(mutate)
        controller.close()
