"""Lossless producer table layout and typed canonical unit metadata."""

import ast
import csv
import json
from pathlib import Path

SCHEMA_FILE = '_phy_cluster_metadata_schema.json'
POLICY = 'membership_v1'


def normalize_value(value):
    """Empty table cells represent missing values, including in typed JSON."""
    return None if value == '' else value


def cell(value):
    if value is None:
        return ''
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return str(value)


def _legacy_cell_matches(text, value):
    if text == str(value):
        return True
    if not isinstance(value, (dict, list)):
        return False
    try:
        # Old csv.writer used Python repr, while the JSON sidecar sorted keys
        # and converted tuples to lists. Compare JSON meaning without exec or
        # Python's bool/int equality conflating true with 1.
        candidate = json.loads(json.dumps(ast.literal_eval(text), allow_nan=False))
        options = {'sort_keys': True, 'separators': (',', ':'), 'allow_nan': False}
        return json.dumps(candidate, **options) == json.dumps(value, **options)
    except (SyntaxError, TypeError, ValueError):
        return False


def validate_schema(schema):
    if (
        not isinstance(schema, dict)
        or set(schema) != {'version', 'tables'}
        or schema['version'] != 1
    ):
        raise ValueError('Unsupported metadata table schema.')
    if not isinstance(schema['tables'], list):
        raise ValueError('Metadata schema tables must be an ordered list.')
    names, fields = set(), set()
    for table in schema['tables']:
        if not isinstance(table, dict) or set(table) != {'filename', 'delimiter', 'columns'}:
            raise ValueError('Invalid metadata table descriptor.')
        name, columns = table['filename'], table['columns']
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or '\\' in name
            or any(c in name for c in '\0\n\r')
            or not name.startswith('cluster_')
            or Path(name).suffix not in ('.tsv', '.csv')
            or Path(name).stem.casefold() == 'cluster_info'
            or name.casefold() in names
        ):
            raise ValueError(f'Unsafe, reserved or colliding metadata filename: {name!r}.')
        if table['delimiter'] != ('\t' if name.endswith('.tsv') else ','):
            raise ValueError(f'Invalid metadata delimiter: {name}.')
        if (
            not isinstance(columns, list)
            or len(columns) < 2
            or columns.count('cluster_id') != 1
            or any(
                not isinstance(f, str) or not f or any(c in f for c in '/\\\0\n\r\t')
                for f in columns
            )
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(f'Invalid or duplicate metadata columns: {name}.')
        for field in columns:
            if field != 'cluster_id':
                if field in fields:
                    raise ValueError(f'Duplicate metadata field {field!r} in {name}.')
                fields.add(field)
        names.add(name.casefold())
    return {'version': 1, 'tables': sorted(schema['tables'], key=lambda t: t['filename'])}


def complete_schema(schema, metadata):
    if 'cluster_id' in metadata:
        raise ValueError('cluster_id is reserved for the metadata unit index.')
    schema = json.loads(json.dumps(validate_schema(schema)))
    fields = {f for t in schema['tables'] for f in t['columns']}
    for field in sorted(set(metadata) - fields):
        schema['tables'].append(
            {
                'filename': f'cluster_{field}.tsv',
                'delimiter': '\t',
                'columns': ['cluster_id', field],
            }
        )
    return validate_schema(schema)


def read_metadata(path):
    path = Path(path)
    schema_path = path / SCHEMA_FILE
    declared = json.loads(schema_path.read_text()) if schema_path.exists() else None
    tables, parsed = [], {}
    filenames = sorted([*path.glob('cluster_*.tsv'), *path.glob('cluster_*.csv')])
    for filename in filenames:
        if filename.stem == 'cluster_info':
            continue
        with filename.open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream, delimiter='\t' if filename.suffix == '.tsv' else ',')
            columns = next(reader, [])
            table = {
                'filename': filename.name,
                'delimiter': '\t' if filename.suffix == '.tsv' else ',',
                'columns': columns,
            }
            tables.append(table)
            validate_schema({'version': 1, 'tables': tables})
            for field in columns:
                if field != 'cluster_id':
                    parsed.setdefault(field, {})
            seen = set()
            for row in reader:
                if len(row) != len(columns):
                    raise ValueError(f'Malformed metadata row in {filename.name}.')
                raw_id = row[columns.index('cluster_id')]
                try:
                    unit = int(raw_id)
                except ValueError as error:
                    raise ValueError(f'Invalid unit ID in {filename.name}: {raw_id!r}.') from error
                if unit < 0 or unit in seen:
                    raise ValueError(f'Duplicate or negative unit {unit} in {filename.name}.')
                seen.add(unit)
                for field, value in zip(columns, row):
                    if field != 'cluster_id':
                        parsed.setdefault(field, {})[unit] = (
                            value if declared is None else normalize_value(value)
                        )
    schema = validate_schema({'version': 1, 'tables': tables})
    if declared is not None and validate_schema(declared) != schema:
        raise ValueError('Metadata table files or columns disagree with metadata schema.')
    sidecar = path / '_phy_cluster_metadata.json'
    if not sidecar.exists():
        if declared is not None:
            raise ValueError('Metadata schema requires the canonical typed JSON sidecar.')
        return parsed, schema
    raw = json.loads(sidecar.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Cluster metadata JSON must be a field mapping.')
    canonical = {}
    for field, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError('Cluster metadata fields must map integer cluster IDs to values.')
        canonical[field] = {}
        for key, value in values.items():
            unit = int(key)
            if str(unit) != key or unit < 0:
                raise ValueError('Cluster metadata IDs must be nonnegative canonical integers.')
            canonical[field][unit] = value
    complete_schema(schema, canonical)  # Validate field-derived filenames as well.
    for table in tables:
        for field in table['columns']:
            if field == 'cluster_id':
                continue
            left, right = parsed.get(field, {}), canonical.get(field, {})
            for unit in set(left) | set(right):
                if declared is None and unit not in left and right.get(unit) is None:
                    continue  # Legacy writer omitted rows for JSON null values.
                if (
                    declared is None
                    and unit in left
                    and unit in right
                    and _legacy_cell_matches(left[unit], right[unit])
                ):
                    continue  # Legacy csv.writer used Python representations for typed values.
                if unit not in left or unit not in right or cell(left[unit]) != cell(right[unit]):
                    raise ValueError(
                        f'Metadata conflict in {table["filename"]}, field {field}, unit {unit}.'
                    )
    if declared is not None and set(canonical) != set(parsed):
        raise ValueError('Canonical metadata fields disagree with table schema.')
    return canonical, complete_schema(schema, canonical)


def write_metadata(path, metadata, schema):
    """Keep source basenames, delimiters and column order, including blank rows."""
    schema = complete_schema(schema, metadata)
    filenames = [table['filename'] for table in schema['tables']]
    for filename in [*filenames, '_phy_cluster_metadata.json', SCHEMA_FILE]:
        if (path / filename).is_symlink():
            raise ValueError(f'Metadata output must not overwrite a symbolic link: {filename}.')
    for table in schema['tables']:
        fields = [f for f in table['columns'] if f != 'cluster_id']
        units = sorted({u for f in fields for u in metadata.get(f, {})})
        with (path / table['filename']).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter=table['delimiter'])
            writer.writerow(table['columns'])
            for unit in units:
                writer.writerow(
                    [
                        unit if f == 'cluster_id' else cell(metadata.get(f, {}).get(unit))
                        for f in table['columns']
                    ]
                )
    (path / '_phy_cluster_metadata.json').write_text(json.dumps(metadata, sort_keys=True) + '\n')
    (path / SCHEMA_FILE).write_text(json.dumps(schema, sort_keys=True) + '\n')
    return schema
