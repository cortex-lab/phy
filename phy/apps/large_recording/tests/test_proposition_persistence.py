"""Producer bytes, durable decisions and retirement survive bounded replay rounds."""

import json
from pathlib import Path

import pytest

from phy.apps.large_recording.adapter import BoundedTemplateController
from phy.apps.large_recording.curation import BoundedSession, _digest
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.source import Source, hash_file
from phy.apps.large_recording.tests.test_adapter import make_gui_sorting
from phy.apps.template.gui import TemplateController
from phy.cluster._propositions import PropositionReview, encode_review_mapping, proposition_key


class PropositionsDisabledController(BoundedTemplateController):
    enable_merge_propositions = False


def prepare(tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    producer = b'{"format_version":"2", "unit_ids":[0,1,2,99], "merges":[{"unit_ids":[0,1]},{"unit_ids":[1,2]}], "producer": {"unchanged":true}}\n'
    (params.parent / 'curation.json').write_bytes(producer)
    review = encode_review_mapping(
        {
            proposition_key([1, 2]): PropositionReview('rejected'),
            proposition_key([77, 78]): PropositionReview('accepted', [77, 78], 90),
        },
        source_sha256=hash_file(params.parent / 'curation.json'),
    )
    (params.parent / 'curation_review.json').write_text(json.dumps(review))
    bounded = prepare_large(params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20)
    return params, bounded, producer, review


@pytest.mark.parametrize('mode', ['direct', 'enabled', 'disabled'])
def test_three_noop_rounds_preserve_exact_source_and_complete_ledger(tmp_path, mode):
    params, bounded, producer, review = prepare(tmp_path)
    original_reviews = review['reviews']
    for round_index in range(3):
        if mode == 'direct':
            session = BoundedSession(bounded)
            session.save()
            session.close()
        else:
            controller_class = (
                PropositionsDisabledController if mode == 'disabled' else TemplateController
            )
            controller = controller_class(
                dir_path=bounded, config_dir=tmp_path / 'config', plugins=[]
            )
            try:
                if mode == 'disabled':
                    assert controller.supervisor.merge_propositions is None
                else:
                    assert controller.supervisor.merge_propositions is not None
                controller.supervisor.save()
            finally:
                controller.close()
        # A committed generation outranks every loose-root review state.
        root_review = bounded / 'curation_review.json'
        if round_index == 0:
            root_review.write_text('{malformed')
        elif round_index == 1:
            root_review.write_text(json.dumps(dict(review, reviews={})))
        else:
            root_review.unlink(missing_ok=True)
        reopened = TemplateController(dir_path=bounded, config_dir=tmp_path / 'config', plugins=[])
        try:
            assert (
                reopened.supervisor.merge_propositions.catalog.review_mapping()['reviews']
                == original_reviews
            )
        finally:
            reopened.close()
        output = apply_curation(params, bounded, tmp_path / f'curated-{round_index}')
        assert (output / 'curation.json').read_bytes() == producer
        assert (
            json.loads((output / 'curation_review.json').read_text())['reviews']
            == original_reviews
        )
        lineage = json.loads((output / '_phy_curation_lineage.json').read_text())
        assert lineage['proposition_documents'] == {
            name: hash_file(output / name) for name in ('curation.json', 'curation_review.json')
        }
        params = output / 'params.py'
        bounded = prepare_large(
            params, tmp_path / f'bounded-{round_index}', window_seconds=20, stratum_seconds=20
        )


def test_accepted_and_rejected_decisions_and_retirements_survive_replay(tmp_path):
    params, bounded, producer, review = prepare(tmp_path)
    controller = TemplateController(dir_path=bounded, config_dir=tmp_path / 'config', plugins=[])
    try:
        supervisor = controller.supervisor
        key = proposition_key([0, 1])
        supervisor.merge_propositions.accept(key, [0, 1], 100)
        supervisor.merge([0, 1], to=100)
        supervisor.save()
        expected_reviews = supervisor.merge_propositions.catalog.review_mapping()['reviews']
    finally:
        controller.close()
    output = apply_curation(params, bounded, tmp_path / 'curated')
    assert (output / 'curation.json').read_bytes() == producer
    assert json.loads((output / 'curation_review.json').read_text())['reviews'] == expected_reviews
    source = Source(output / 'params.py')
    try:
        assert source.retired_units == {
            '0': {'operation': 'merge', 'replacements': [100]},
            '1': {'operation': 'merge', 'replacements': [100]},
        }
    finally:
        source.close()
    second = prepare_large(
        output / 'params.py', tmp_path / 'second', window_seconds=20, stratum_seconds=20
    )
    session = BoundedSession(second)
    session.record_split(
        100,
        101,
        102,
        {'attribute': 'amplitudes.npy', 'operator': '<', 'threshold': 4.0, 'nonfinite': 'keep'},
    )
    session.save()
    session.close()
    final = apply_curation(output / 'params.py', second, tmp_path / 'final')
    lineage = json.loads((final / '_phy_curation_lineage.json').read_text())
    assert lineage['retired_units']['100'] == {
        'operation': 'threshold_split',
        'replacements': [102, 101],
    }
    assert lineage['retired_units']['0'] == {'operation': 'merge', 'replacements': [100]}


@pytest.mark.parametrize('target', ['curation.json', '_phy_curation_review_base.json'])
def test_mutation_after_load_rejects_save_and_replay(tmp_path, target):
    params, bounded, _, _ = prepare(tmp_path)
    session = BoundedSession(bounded)
    session.save()
    (bounded / target).write_text('{}')
    try:
        with pytest.raises(ValueError, match='changed'):
            session.save()
        with pytest.raises(ValueError, match='integrity'):
            apply_curation(params, bounded, tmp_path / 'bad')
        assert not (tmp_path / 'bad').exists()
    finally:
        session.close()


def test_generation_extra_document_requires_hash_inventory(tmp_path):
    _, bounded, _, _ = prepare(tmp_path)
    session = BoundedSession(bounded)
    generation = session.save()
    session.close()
    descriptor = json.loads((generation / 'checkpoint.json').read_text())
    descriptor['files'].pop('curation_review.json')
    (generation / 'checkpoint.json').write_text(json.dumps(descriptor))
    pointer = bounded / '.phy-curation/current.json'
    data = json.loads(pointer.read_text())
    data['sha256'] = _digest(descriptor)
    pointer.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash inventory'):
        BoundedSession(bounded)


@pytest.mark.parametrize('operation', ['prepare', 'replay'])
def test_copy_race_rejects_producer_mutation_without_publication(tmp_path, monkeypatch, operation):
    from phy.apps.large_recording import prepare as preparation
    from phy.apps.large_recording import replay

    params, bounded, _, _ = prepare(tmp_path)
    session = BoundedSession(bounded)
    session.save()
    session.close()
    module = preparation if operation == 'prepare' else replay
    original_copy = module.shutil.copyfile

    def mutate_copy(source, destination, *args, **kwargs):
        result = original_copy(source, destination, *args, **kwargs)
        if Path(destination).name == 'curation.json':
            Path(destination).write_text('{}')
        return result

    monkeypatch.setattr(module.shutil, 'copyfile', mutate_copy)
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='changed'):
        if operation == 'prepare':
            prepare_large(params, output, window_seconds=20, stratum_seconds=20)
        else:
            apply_curation(params, bounded, output)
    assert not output.exists()


@pytest.mark.parametrize(
    'filename, digest',
    [
        ('../curation.json', 'valid'),
        ('other.json', 'valid'),
        ('curation.json', None),
        ('curation.json', '0' * 64),
        ('curation.json', 'xyz'),
    ],
)
def test_bounded_inherited_review_requires_exact_filename_and_hash(tmp_path, filename, digest):
    params, _, _, review = prepare(tmp_path)
    review['source']['filename'] = filename
    if digest is None:
        review['source'].pop('sha256')
    elif digest != 'valid':
        review['source']['sha256'] = digest
    (params.parent / 'curation_review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError, match='filename/hash'):
        prepare_large(params, tmp_path / 'bad', window_seconds=20, stratum_seconds=20)
    assert not (tmp_path / 'bad').exists()


def test_gui_save_rejects_producer_mutation_since_load(tmp_path):
    _, bounded, _, _ = prepare(tmp_path)
    controller = TemplateController(dir_path=bounded, config_dir=tmp_path / 'config', plugins=[])
    try:
        (bounded / 'curation.json').write_text('{}')
        with pytest.raises(ValueError, match='changed'):
            controller.supervisor.save()
        assert not (bounded / '.phy-curation/current.json').exists()
    finally:
        controller.close()


def test_replay_rejects_source_producer_mutation(tmp_path):
    params, bounded, _, _ = prepare(tmp_path)
    session = BoundedSession(bounded)
    session.save()
    session.close()
    (params.parent / 'curation.json').write_text('{}')
    with pytest.raises(ValueError, match='changed since preparation'):
        apply_curation(params, bounded, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_legacy_bounded_propositions_require_repreparation(tmp_path):
    _, bounded, _, _ = prepare(tmp_path)
    path = bounded / '_phy_curation_manifest.json'
    manifest = json.loads(path.read_text())
    manifest.pop('proposition_documents')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reprepare'):
        BoundedSession(bounded)


def test_legacy_checkpoint_review_without_loose_root_requires_repreparation(tmp_path):
    from phy.apps.large_recording.curation import _hash_file
    from phy.apps.large_recording.tests.test_curation import make_session

    _, session = make_session(tmp_path)
    generation = session.save(extra_documents={'supplemental.json': {}})
    session.close()
    (generation / 'supplemental.json').rename(generation / 'curation_review.json')
    descriptor_path = generation / 'checkpoint.json'
    descriptor = json.loads(descriptor_path.read_text())
    descriptor['extra_documents'] = ['curation_review.json']
    descriptor['files'].pop('supplemental.json')
    descriptor['files']['curation_review.json'] = _hash_file(generation / 'curation_review.json')
    descriptor_path.write_text(json.dumps(descriptor))
    pointer_path = session.path / '.phy-curation/current.json'
    pointer = json.loads(pointer_path.read_text())
    pointer['sha256'] = _digest(descriptor)
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='reprepare'):
        BoundedSession(session.path)


@pytest.mark.parametrize('mode', ['direct', 'disabled'])
def test_suggestions_without_reviews_publish_empty_bound_ledger(tmp_path, mode):
    params, _, producer, _ = prepare(tmp_path)
    (params.parent / 'curation_review.json').unlink()
    bounded = prepare_large(params, tmp_path / 'empty', window_seconds=20, stratum_seconds=20)
    if mode == 'direct':
        session = BoundedSession(bounded)
        session.save()
        session.close()
    else:
        controller = PropositionsDisabledController(
            dir_path=bounded, config_dir=tmp_path / 'config', plugins=[]
        )
        try:
            assert controller.supervisor.merge_propositions is None
            controller.supervisor.save()
        finally:
            controller.close()
    output = apply_curation(params, bounded, tmp_path / 'curated')
    assert (output / 'curation.json').read_bytes() == producer
    assert json.loads((output / 'curation_review.json').read_text()) == {
        'format_version': '1',
        'source': {'filename': 'curation.json', 'sha256': hash_file(output / 'curation.json')},
        'reviews': {},
    }


def test_null_review_cannot_remove_inherited_decisions(tmp_path):
    _, bounded, _, review = prepare(tmp_path)
    session = BoundedSession(bounded)
    try:
        with pytest.raises(ValueError, match='ledger mapping'):
            session.save(extra_documents={'curation_review.json': None})
        assert session.extra_documents['curation_review.json'] == review
        assert not (bounded / '.phy-curation/current.json').exists()
    finally:
        session.close()


def test_committed_generation_cannot_omit_required_review_ledger(tmp_path):
    _, bounded, _, _ = prepare(tmp_path)
    session = BoundedSession(bounded)
    generation = session.save()
    session.close()
    descriptor_path = generation / 'checkpoint.json'
    descriptor = json.loads(descriptor_path.read_text())
    descriptor['extra_documents'] = []
    descriptor['files'].pop('curation_review.json')
    descriptor_path.write_text(json.dumps(descriptor))
    pointer_path = bounded / '.phy-curation/current.json'
    pointer = json.loads(pointer_path.read_text())
    pointer['sha256'] = _digest(descriptor)
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match='ledger mapping'):
        BoundedSession(bounded)
