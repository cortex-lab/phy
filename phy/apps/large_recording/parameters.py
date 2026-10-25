"""Maintain replay-owned parameter overrides without rewriting user Python."""

import ast
import io
import tokenize

BEGIN = '# BEGIN PHY CURATION PARAMETERS'
END = '# END PHY CURATION PARAMETERS'
KEYS = ('num_samples', 'valid_intervals', 'dat_path')


def _assignments(nodes):
    values = {}
    for node, key in zip(nodes, KEYS, strict=True):
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id != key
            or node.col_offset != 0
        ):
            raise ValueError('Not a generated parameter assignment.')
        values[key] = ast.literal_eval(node.value)
    return values


def _without_managed_block(text):
    markers = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT and token.start[1] == 0 and token.string in (BEGIN, END):
            markers.append((token.string, token.start[0]))
    if not markers:
        return text
    if len(markers) != 2 or [item[0] for item in markers] != [BEGIN, END]:
        raise ValueError('Malformed Phy curation parameter block.')
    lines = text.splitlines(keepends=True)
    first, last = markers[0][1] - 1, markers[1][1] - 1
    if ''.join(lines[last + 1 :]).strip():
        raise ValueError('Phy curation parameter block must be the final content in params.py.')
    nodes = ast.parse(''.join(lines[first + 1 : last])).body
    if len(nodes) != 3:
        raise ValueError('Unexpected code inside Phy curation parameter block.')
    try:
        _assignments(nodes)
    except (ValueError, TypeError) as error:
        raise ValueError('Unexpected code inside Phy curation parameter block.') from error
    return ''.join(lines[:first])


def _strip_legacy_suffix(text, overrides):
    """Remove only recognizable duplicated trailing triples from the old writer."""
    while True:
        nodes = ast.parse(text).body
        if len(nodes) < 6:
            return text
        tail = nodes[-3:]
        try:
            values = _assignments(tail)
        except (ValueError, TypeError):
            return text
        if values != overrides:
            return text
        previous = {
            target.id
            for node in nodes[:-3]
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if not set(KEYS) <= previous:
            return text
        lines = text.splitlines(keepends=True)
        if any(node.lineno != node.end_lineno for node in tail):
            return text
        if [node.lineno for node in tail] != list(range(tail[0].lineno, tail[0].lineno + 3)):
            return text
        if any(
            lines[node.lineno - 1].rstrip('\r\n') != f'{key} = {values[key]!r}'
            for node, key in zip(tail, KEYS, strict=True)
        ):
            return text
        if ''.join(lines[tail[-1].end_lineno :]).strip():
            return text
        # The old writer inserted one blank line immediately before every triple.
        stop = tail[0].lineno - 1
        if stop and not lines[stop - 1].strip():
            stop -= 1
        text = ''.join(lines[:stop])


def rewrite_replay_params(text, *, num_samples, valid_intervals, dat_path):
    """Return source text with one stable, final block of replay-owned values.

    Original assignments may feed later user expressions and therefore remain
    intact. Only this marked block and confidently identified legacy generated
    suffixes are replaced; imports, comments and executable source are retained.
    """
    if not isinstance(text, str):
        raise TypeError('Parameter source must be text.')
    ast.parse(text)  # Validate syntax without executing user source again.
    overrides = dict(zip(KEYS, (num_samples, valid_intervals, dat_path), strict=True))
    # Recognize old suffix values before normalizing legacy no-raw-data strings.
    prefix = _strip_legacy_suffix(_without_managed_block(text), overrides)
    paths = dat_path if isinstance(dat_path, (list, tuple)) else [dat_path]
    if all(
        path is None or (isinstance(path, str) and path.strip() in ('', 'None')) for path in paths
    ):
        overrides['dat_path'] = None
    newline = '\r\n' if '\r\n' in text else '\n'
    if prefix and not prefix.endswith(newline * 2):
        prefix += newline if prefix.endswith(newline) else newline * 2
    block = newline.join([BEGIN, *(f'{key} = {overrides[key]!r}' for key in KEYS), END, ''])
    return prefix + block
