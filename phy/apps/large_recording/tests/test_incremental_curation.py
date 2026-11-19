"""Interactive commands must not replay unrelated historical spike assignments."""

import numpy as np
import pytest

from phy.apps.large_recording.tests.test_curation import make_session


def test_label_has_no_assignment_projection(tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    source, session = make_session(tmp_path)
    try:
        session.record_merge([0, 1], 100, {'group': 'good'})
        assignments = session.assignments

        def forbidden(*args, **kwargs):
            pytest.fail('A label projected spike assignments.')

        monkeypatch.setattr(curation, 'project_assignments', forbidden)
        event = session.preflight_label('group', [100], 'mua')
        assert session.events[-1]['operation'] == 'merge'
        session._append(event)
        assert session.assignments is assignments
        assert session.metadata['group'][100] == 'mua'
        with pytest.raises(ValueError, match='inactive'):
            session.record_label('group', [0], 'good')
        assert session.assignments is assignments
    finally:
        session.close()
        source.close()
