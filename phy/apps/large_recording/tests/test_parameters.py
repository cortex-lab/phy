"""Replay parameters retain Python semantics and stop accumulating overrides."""

import ast

import pytest

from phy.apps.large_recording.parameters import (
    BEGIN,
    END,
    rewrite_replay_params,
    rewrite_replay_params_bytes,
)

VALUES = {'num_samples': 20, 'valid_intervals': [[0, 8], [10, 20]], 'dat_path': None}


@pytest.mark.parametrize('ending', ['', '\n', '\r\n'])
def test_managed_parameters_are_idempotent_and_keep_user_source(ending):
    original = '# User comment\nimport pathlib\nnum_samples = 7\n'
    original += 'derived = num_samples * 2\nvalid_intervals = [[0, 7]]\n'
    original += 'alias = dat_path = "relative.bin"' + ending
    rewritten = rewrite_replay_params(original, **VALUES)
    assert rewritten.startswith(original)
    for _ in range(3):
        assert rewrite_replay_params(rewritten, **VALUES) == rewritten
    namespace = {}
    exec(rewritten, namespace)
    assert namespace['derived'] == 14 and namespace['alias'] == 'relative.bin'
    assert {key: namespace[key] for key in VALUES} == VALUES
    assert rewritten.count(BEGIN) == rewritten.count(END) == 1


def test_legacy_literal_suffixes_collapse_without_deleting_originals():
    original = "num_samples = 20\nvalid_intervals = [[0, 8], [10, 20]]\ndat_path = r'None'\n"
    legacy = original + (
        "\nnum_samples = 20\nvalid_intervals = [[0, 8], [10, 20]]\ndat_path = 'None'\n" * 3
    )
    values = dict(VALUES, dat_path='None')
    rewritten = rewrite_replay_params(legacy, **values)
    assert rewritten.startswith(original)
    assert len(rewritten) < len(legacy)
    assert rewrite_replay_params(rewritten, **VALUES) == rewritten
    assert 'dat_path = None' in rewritten
    assert (
        sum(
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'num_samples'
            for node in ast.parse(rewritten).body
        )
        == 2
    )


def test_nonliteral_chained_conditional_and_commented_assignments_remain():
    original = 'num_samples = 10 + 10\nvalid_intervals = [[0, 20]]\ndat_path = None\n'
    original += 'if True:\n    num_samples = 19\n'
    original += "num_samples = valid_intervals = dat_path = 'keep'\n"
    original += 'num_samples = 20  # user annotation\nvalid_intervals = [[0, 8], [10, 20]]\ndat_path = None\n'
    assert rewrite_replay_params(original, **VALUES).startswith(original)


@pytest.mark.parametrize(
    'source',
    [
        BEGIN + '\nnum_samples = 1\n',
        END + '\n' + BEGIN + '\n',
        BEGIN + '\nprint("keep")\n' + END + '\n',
        BEGIN
        + '\nnum_samples = 20\nvalid_intervals = []\ndat_path = None\n'
        + END
        + '\nprint("keep")\n',
    ],
)
def test_malformed_or_modified_managed_blocks_fail_without_deleting_code(source):
    with pytest.raises(ValueError):
        rewrite_replay_params(source, **VALUES)


def test_marker_strings_and_quoted_paths_are_safe():
    source = f'documentation = """\n{BEGIN}\n{END}\n"""\n'
    path = "a'quote\\folder/\u00e9lectrode.bin"
    result = rewrite_replay_params(source, **dict(VALUES, dat_path=path))
    assert result.startswith(source)
    namespace = {}
    exec(result, namespace)
    assert namespace['dat_path'] == path


def test_crlf_prefix_is_preserved_byte_for_byte():
    original = '# \u00e9lectrode\r\nnum_samples = 7\r\nderived = num_samples * 2\r\n'
    result = rewrite_replay_params(original, **VALUES)
    assert result.startswith(original)
    assert '\n' not in result.replace('\r\n', '')
    assert rewrite_replay_params(result, **VALUES) == result


@pytest.mark.parametrize('encoding', ['utf-8-sig', 'cp1252'])
def test_python_encoding_and_bom_survive_with_unicode_generated_paths(encoding):
    text = f'# coding: {encoding}\r\n# \u00e9lectrode\r\nnum_samples = 7\r\n'
    original = text.encode(encoding)
    values = dict(VALUES, dat_path='\u00e9lectrode/\u4e2d\u6587.bin')
    result = rewrite_replay_params_bytes(original, **values)
    assert result.startswith(original)
    assert rewrite_replay_params_bytes(result, **values) == result
    namespace = {}
    exec(result, namespace)
    assert namespace['dat_path'] == values['dat_path']


def test_three_replay_rounds_preserve_crlf_and_stable_file_size(tmp_path):
    from phy.apps.large_recording.curation import BoundedSession
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.replay import apply_curation
    from phy.apps.large_recording.tests.test_curation import make_session

    source, session = make_session(tmp_path)
    params = source.params_path
    original = params.read_text().replace('\n', '\r\n').encode('utf8')
    params.write_bytes(original)
    source.close()
    session.save()
    sizes = []
    for index in range(3):
        output = apply_curation(params, session.path, tmp_path / f'output-{index}')
        session.close()
        params = output / 'params.py'
        content = params.read_bytes()
        assert content.startswith(original)
        assert content.count(BEGIN.encode()) == 1
        assert b'\n' not in content.replace(b'\r\n', b'')
        sizes.append(len(content))
        if index < 2:
            bounded = prepare_large(
                params,
                tmp_path / f'bounded-{index}',
                window_seconds=20,
                stratum_seconds=20,
                max_spikes=100,
                bin_seconds=5,
                batch_size=2,
            )
            session = BoundedSession(bounded)
            session.save()
    assert len(set(sizes)) == 1
