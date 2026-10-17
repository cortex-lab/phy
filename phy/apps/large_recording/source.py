"""Validated, file-backed source sorting on one monotonic sample clock."""

import csv
import ctypes
import errno
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

PER_SPIKE = frozenset(
    (
        'spike_times.npy',
        'spike_clusters.npy',
        'spike_templates.npy',
        'amplitudes.npy',
        'pc_features.npy',
        'template_features.npy',
        'spike_positions.npy',
        'spike_depths.npy',
    )
)
STATIC = frozenset(
    (
        'templates.npy',
        'templates_ind.npy',
        'template_ind.npy',
        'channel_map.npy',
        'channel_positions.npy',
        'channel_shanks.npy',
        'channel_map_si.npy',
        'channel_groups.npy',
        'whitening_mat.npy',
        'whitening_mat_inv.npy',
        'similar_templates.npy',
        'pc_feature_ind.npy',
        'template_feature_ind.npy',
    )
)
WAVEFORMS = frozenset(
    (
        '_phy_spikes_subset.spikes.npy',
        '_phy_spikes_subset.channels.npy',
        '_phy_spikes_subset.waveforms.npy',
    )
)
WINDOW_SIDECARS = frozenset(
    (
        '_phy_spikes_subset.window_bounds.npy',
        '_phy_spikes_subset.window_types.npy',
        '_phy_spikes_subset.window_strata.npy',
    )
)


def hash_file(path, batch_bytes=1024 * 1024):
    """Hash bytes incrementally, including the array header."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(batch_bytes), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value):
    """Hash canonical JSON, normalizing integer metadata dictionary keys."""

    def normalize(item):
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    payload = json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def assignment_revision(path, metadata):
    """Identify mutable assignments and resolved cluster metadata."""
    path = Path(path)
    if path.is_dir():
        path = path / 'spike_clusters.npy'
    return hash_json({'assignments': hash_file(path), 'metadata': metadata})


def close_array(array):
    """Release a file-backed array handle, including on Windows."""
    mapping = getattr(array, '_mmap', None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def publish_directory(staged, destination):
    """Atomically publish a complete directory without replacing any target.

    Ordinary POSIX rename can replace an empty destination directory. Native
    exclusive rename prevents a racing publication from silently replacing it.
    """
    if os.name == 'nt':
        os.rename(staged, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux'):
        rename = getattr(library, 'renameat2', None)
        if rename is None:
            raise OSError(errno.ENOSYS, 'Atomic exclusive directory publication is unavailable.')
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1)
    elif sys.platform == 'darwin':
        rename = getattr(library, 'renamex_np', None)
        if rename is None:
            raise OSError(errno.ENOSYS, 'Atomic exclusive directory publication is unavailable.')
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged), os.fsencode(destination), 4)
    else:
        raise OSError(errno.ENOSYS, 'Atomic exclusive directory publication is unavailable.')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _vector(array, name):
    if array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1):
        return array.reshape(-1)
    raise ValueError(f'{name} must be a vector or column vector.')


def _metadata(path):
    sidecar = path / '_phy_cluster_metadata.json'
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError('Cluster metadata JSON must be a field mapping.')
        result = {}
        for field, values in metadata.items():
            if not isinstance(values, dict):
                raise ValueError('Cluster metadata fields must map integer cluster IDs to values.')
            result[field] = {int(key): value for key, value in values.items()}
            if any(key < 0 for key in result[field]):
                raise ValueError('Cluster metadata IDs must be nonnegative.')
        return result
    result = {}
    for filename in sorted([*path.glob('cluster_*.tsv'), *path.glob('cluster_*.csv')]):
        # cluster_info is generated GUI metrics, not a source label table.
        if filename.stem == 'cluster_info':
            continue
        with filename.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t' if filename.suffix == '.tsv' else ',')
            if not reader.fieldnames or 'cluster_id' not in reader.fieldnames:
                raise ValueError(f'{filename.name} requires a cluster_id column.')
            for row in reader:
                cluster_id = int(row['cluster_id'])
                if cluster_id < 0:
                    raise ValueError('Cluster metadata IDs must be nonnegative.')
                for field, value in row.items():
                    if field != 'cluster_id':
                        result.setdefault(field, {})[cluster_id] = value
    return result


class Source:
    """Read source arrays in batches without constructing a GUI model.

    Duration must come from producer metadata or an explicit argument: the last
    spike is never used as acquisition exposure. Intervals are half-open samples.
    """

    def __init__(
        self, params_path, duration_seconds=None, valid_intervals=None, batch_size=100000
    ):
        self.arrays = {}
        try:
            self._initialize(params_path, duration_seconds, valid_intervals, batch_size)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self, params_path, duration_seconds=None, valid_intervals=None, batch_size=100000
    ):
        self.params_path = Path(params_path).resolve()
        self.path = self.params_path.parent
        self.params = {
            key: value
            for key, value in runpy.run_path(str(self.params_path)).items()
            if not key.startswith('__')
        }
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive.')
        if 'dtype' in self.params:
            try:
                raw_dtype = np.dtype(self.params['dtype'])
            except (TypeError, ValueError) as error:
                raise ValueError('params.py dtype must be a valid raw numeric dtype.') from error
            if not np.issubdtype(raw_dtype, np.number):
                raise ValueError('params.py dtype must be a valid raw numeric dtype.')
        self.sample_rate = float(self.params.get('sample_rate', 0))
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError('A positive finite sample_rate is required.')
        upstream = self.path / '_phy_spikes_subset.manifest.json'
        recording = {}
        info = {}
        if upstream.exists():
            info = json.loads(upstream.read_text(encoding='utf-8'))
            if info.get('format') != 'phy_spikes_subset' or info.get('format_version') != 1:
                raise ValueError('Unsupported waveform manifest format or version.')
            recording = info.get('recording', {})
            upstream_rate = recording.get('sampling_frequency_hz', self.sample_rate)
            if upstream_rate != self.sample_rate:
                raise ValueError('Waveform manifest sampling rate disagrees with params.py.')
        if self.params.get('num_segments', recording.get('num_segments', 1)) != 1:
            raise ValueError('Only a single monotonic acquisition segment is supported.')
        duration_samples = None
        if duration_seconds is not None:
            duration_samples = float(duration_seconds) * self.sample_rate
            if (
                not np.isfinite(duration_samples)
                or duration_samples <= 0
                or duration_samples > np.iinfo(np.int64).max
                or not np.isclose(duration_samples, round(duration_samples), rtol=0, atol=1e-6)
            ):
                raise ValueError('duration_seconds must specify a positive integer sample count.')
            duration_samples = int(round(duration_samples))
        if (
            'num_samples' in self.params
            and 'num_samples' in recording
            and self.params['num_samples'] != recording['num_samples']
        ):
            raise ValueError(
                'Producer sample counts disagree between params and waveform manifest.'
            )
        declared_samples = self.params.get('num_samples', recording.get('num_samples'))
        if declared_samples is None:
            declared_samples = duration_samples
        elif duration_samples is not None and int(declared_samples) != duration_samples:
            raise ValueError('Explicit duration disagrees with acquisition metadata.')
        if declared_samples is None:
            raise ValueError(
                'Acquisition exposure is missing; provide duration_seconds or num_samples.'
            )
        self.num_samples = int(declared_samples)
        if (
            self.num_samples <= 0
            or self.num_samples > np.iinfo(np.int64).max
            or self.num_samples != declared_samples
        ):
            raise ValueError('num_samples must be a positive int64 sample count.')
        if (
            'valid_intervals' in self.params
            and 'valid_intervals' in recording
            and not np.array_equal(
                np.asarray(self.params['valid_intervals']),
                np.asarray(recording['valid_intervals']),
            )
        ):
            raise ValueError(
                'Producer valid_intervals disagree between params and waveform manifest.'
            )
        declared_intervals = self.params.get('valid_intervals', recording.get('valid_intervals'))
        if valid_intervals is not None and declared_intervals is not None:
            if not np.array_equal(np.asarray(valid_intervals), np.asarray(declared_intervals)):
                raise ValueError('Explicit valid_intervals disagree with acquisition metadata.')
        if valid_intervals is None:
            valid_intervals = declared_intervals
        intervals = [[0, self.num_samples]] if valid_intervals is None else valid_intervals
        raw_intervals = np.asarray(intervals)
        if raw_intervals.ndim != 2 or raw_intervals.shape[1] != 2 or not len(raw_intervals):
            raise ValueError('valid_intervals must be nonempty sample [start, end) pairs.')
        self.valid_intervals = raw_intervals.astype(np.int64)
        if (
            not np.array_equal(raw_intervals, self.valid_intervals)
            or np.any(self.valid_intervals[:, 0] < 0)
            or np.any(self.valid_intervals[:, 1] > self.num_samples)
            or np.any(self.valid_intervals[:, 1] <= self.valid_intervals[:, 0])
            or np.any(self.valid_intervals[1:, 0] < self.valid_intervals[:-1, 1])
        ):
            raise ValueError(
                'valid_intervals must be ordered, disjoint integer acquisition samples.'
            )
        names = {p.name for p in self.path.glob('*.npy')}
        unknown = names - PER_SPIKE - STATIC - WAVEFORMS - WINDOW_SIDECARS
        # Replay lineage and original optional sidecars are retained; bounded
        # outputs deliberately cannot be used as full-source sorting inputs.
        if unknown:
            raise ValueError(f'Unregistered source arrays: {", ".join(sorted(unknown))}.')
        self.arrays = {
            name: np.load(self.path / name, mmap_mode='r', allow_pickle=False)
            for name in sorted(names)
        }
        for required in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy'):
            if required not in self.arrays:
                raise ValueError(f'Missing required source array {required}.')
        self.n_spikes = len(_vector(self.arrays['spike_times.npy'], 'spike_times.npy'))
        self.metadata = _metadata(self.path)
        self.validate()
        counts = {}
        for batch in self.iter_batches():
            ids, values = np.unique(batch['spike_clusters.npy'], return_counts=True)
            for cluster_id, count in zip(ids.tolist(), values.tolist(), strict=True):
                counts[cluster_id] = counts.get(cluster_id, 0) + count
        for values in self.metadata.values():
            for cluster_id in values:
                counts.setdefault(cluster_id, 0)
        lineage_path = self.path / '_phy_curation_lineage.json'
        lineage = {}
        if lineage_path.exists():
            lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
            if (
                lineage.get('format') != 'phy_curation_lineage'
                or lineage.get('format_version') != 1
            ):
                raise ValueError('Unsupported curation lineage format or version.')
            for unit in lineage.get('unit_ids', []):
                if not isinstance(unit, int) or unit < 0:
                    raise ValueError('Lineage unit IDs must be nonnegative integers.')
                counts.setdefault(unit, 0)
        if any(unit < 0 or unit > np.iinfo(np.int32).max for unit in counts):
            raise ValueError(
                'Unit IDs must fit nonnegative int32 for the current Phy loader; spike sample/row coordinates remain int64.'
            )
        self.unit_ids = np.array(sorted(counts), dtype=np.int64)
        minimum_next = int(self.unit_ids[-1]) + 1 if len(self.unit_ids) else 0
        self.next_cluster_id = lineage.get('next_cluster_id', minimum_next)
        if not isinstance(self.next_cluster_id, int) or self.next_cluster_id < minimum_next:
            raise ValueError('Lineage next_cluster_id collides with the source unit catalog.')
        self.counts = np.array(
            [counts[cluster_id] for cluster_id in self.unit_ids], dtype=np.int64
        )
        immutable = sorted(names - {'spike_clusters.npy'} - WINDOW_SIDECARS)
        # Optional subset planning metadata describes a sampling policy, not a
        # new immutable spike-coordinate identity.
        acquisition = {
            key: self.params[key]
            for key in (
                'acquisition_id',
                'n_channels_dat',
                'dtype',
                'offset',
                'depth_axis',
                'amplitude_units',
                'depth_units',
                'position_units',
                'waveform_units',
            )
            if key in self.params
        }
        self.identity = {
            'format_version': 1,
            'acquisition': acquisition,
            'time_base': {
                'sample_rate': self.sample_rate,
                'num_samples': self.num_samples,
                'valid_intervals': self.valid_intervals.tolist(),
            },
            'arrays': {name: hash_file(self.path / name) for name in immutable},
        }
        self.identity['sha256'] = hash_json(self.identity)
        declared_source = info.get('source_sorting', {})
        if not isinstance(declared_source, dict):
            raise ValueError('Waveform manifest source_sorting must be a mapping.')
        if declared_source.get('num_spikes', self.n_spikes) != self.n_spikes:
            raise ValueError('Waveform reservoir source spike count does not match this sorting.')
        declared_times = declared_source.get('spike_times_sha256')
        if (
            declared_times is not None
            and declared_times != self.identity['arrays']['spike_times.npy']
        ):
            raise ValueError(
                'Waveform reservoir source coordinate fingerprint does not match this sorting.'
            )
        self.revision = assignment_revision(self.path, self.metadata)
        if lineage:
            if (
                lineage.get('coordinate_identity') != self.identity
                or lineage.get('revision') != self.revision
            ):
                raise ValueError(
                    'Curation lineage does not match immutable identity or assignments.'
                )

    def close(self):
        """Release canonical array mappings after a batch command."""
        for array in self.arrays.values():
            close_array(array)
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def iter_batches(self):
        """Yield bounded row slices; vector fields use one-dimensional views."""
        for start in range(0, self.n_spikes, self.batch_size):
            stop = min(start + self.batch_size, self.n_spikes)
            batch = {'start': start, 'stop': stop}
            for name in self.arrays.keys() & PER_SPIKE:
                array = self.arrays[name]
                if name in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy'):
                    array = _vector(array, name)
                batch[name] = array[start:stop]
            yield batch

    def validate(self):
        """Check source row, clock, sparse channel, and feature alignment."""
        for name in self.arrays.keys() & PER_SPIKE:
            if self.arrays[name].shape[0] != self.n_spikes:
                raise ValueError(f'{name} is not aligned with source spikes.')
        for name in ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy'):
            array = _vector(self.arrays[name], name)
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f'{name} must contain integer coordinates.')
        previous = -1
        for batch in self.iter_batches():
            times = batch['spike_times.npy']
            if (
                np.any(times < previous)
                or np.any(times[1:] < times[:-1])
                or np.any(times < 0)
                or np.any(times >= self.num_samples)
            ):
                raise ValueError(
                    'Spike sample times must be monotonic and within acquisition bounds.'
                )
            if len(times):
                previous = int(times[-1])
            interval_ids = np.searchsorted(self.valid_intervals[:, 0], times, side='right') - 1
            if np.any(interval_ids < 0) or np.any(times >= self.valid_intervals[interval_ids, 1]):
                raise ValueError('Spike times fall outside valid acquisition intervals.')
            for name in ('spike_clusters.npy', 'spike_templates.npy'):
                if np.any(batch[name] < 0):
                    raise ValueError(f'{name} must contain nonnegative IDs.')
            templates = self.arrays.get('templates.npy')
            if templates is not None and np.any(batch['spike_templates.npy'] >= len(templates)):
                raise ValueError('spike_templates.npy references an unavailable template.')
        waveforms = self.arrays.keys() & WAVEFORMS
        if waveforms and waveforms != WAVEFORMS:
            raise ValueError('Stored waveforms require spikes, channels, and waveforms together.')
        if waveforms:
            ids = self.arrays['_phy_spikes_subset.spikes.npy']
            if ids.ndim != 1 or ids.dtype != np.dtype('int64'):
                raise ValueError('Stored waveform row IDs must be a one-dimensional int64 array.')
            channels = self.arrays['_phy_spikes_subset.channels.npy']
            values = self.arrays['_phy_spikes_subset.waveforms.npy']
            n_channels = len(self.arrays.get('channel_map.npy', []))
            if (
                values.ndim != 3
                or channels.shape != (len(ids), values.shape[2])
                or len(values) != len(ids)
                or not np.issubdtype(channels.dtype, np.integer)
            ):
                raise ValueError('Stored waveform tensors and channels are not aligned.')
            last = -1
            # Sparse-padding checks create temporary boolean arrays and gathers;
            # cap their tensor bytes independently of the source row batch size.
            row_bytes = max(1, values.dtype.itemsize * int(np.prod(values.shape[1:])))
            waveform_batch = min(self.batch_size, max(1, 8 * 1024 * 1024 // row_bytes))
            for start in range(0, len(ids), waveform_batch):
                rows = ids[start : start + waveform_batch]
                slots = channels[start : start + waveform_batch]
                if (
                    np.any(rows <= last)
                    or np.any(rows[1:] <= rows[:-1])
                    or np.any(rows < 0)
                    or np.any(rows >= self.n_spikes)
                ):
                    raise ValueError('Stored waveform row IDs must be sorted, unique source rows.')
                if len(rows):
                    last = int(rows[-1])
                if np.any((slots < -1) | (slots >= n_channels)):
                    raise ValueError(
                        'Sparse waveform channels must index model channels or use -1.'
                    )
                block = values[start : start + waveform_batch]
                if np.any(np.moveaxis(block, 1, 2)[slots == -1] != 0):
                    raise ValueError('Sparse waveform padding channels must contain zeros.')
        channel_map = self.arrays.get('channel_map.npy')
        if channel_map is not None:
            channels = _vector(channel_map, 'channel_map.npy')
            if (
                not np.issubdtype(channels.dtype, np.integer)
                or np.any(channels < 0)
                or len(np.unique(channels)) != len(channels)
            ):
                raise ValueError('channel_map.npy must contain unique nonnegative raw column IDs.')
            raw_channels = self.params.get('n_channels_dat')
            if raw_channels is not None and (
                int(raw_channels) != raw_channels
                or raw_channels <= 0
                or np.any(channels >= raw_channels)
            ):
                raise ValueError('channel_map.npy contains columns outside n_channels_dat.')
            for name in ('channel_groups.npy', 'channel_shanks.npy', 'channel_map_si.npy'):
                if name in self.arrays and len(_vector(self.arrays[name], name)) != len(channels):
                    raise ValueError(f'{name} must align with model channels.')
            for name in ('whitening_mat.npy', 'whitening_mat_inv.npy'):
                if name in self.arrays and self.arrays[name].shape != (
                    len(channels),
                    len(channels),
                ):
                    raise ValueError(f'{name} must contain a model-channel square matrix.')
            templates = self.arrays.get('templates.npy')
            if templates is not None:
                # phylib uses singular template_ind; KS2's plural sidecar does
                # not turn a dense tensor into a sparse tensor in the loader.
                sparse = self.arrays.get('template_ind.npy')
                if templates.ndim != 3:
                    raise ValueError('templates.npy must be a template/sample/channel tensor.')
                if sparse is None and templates.shape[2] != len(channels):
                    raise ValueError('Dense templates must align with model channels.')
                for index_name in ('template_ind.npy', 'templates_ind.npy'):
                    sparse = self.arrays.get(index_name)
                    if sparse is None:
                        continue
                    if sparse.shape != (len(templates), templates.shape[2]):
                        raise ValueError(f'{index_name} must align with sparse templates.')
                    if not np.issubdtype(sparse.dtype, np.integer):
                        raise ValueError(f'{index_name} must contain integer channel indices.')
                    if np.any((sparse < -1) | (sparse >= len(channels))):
                        raise ValueError(f'{index_name} contains invalid model channel IDs.')
            positions = self.arrays.get('channel_positions.npy')
            if positions is not None and (
                positions.shape != (len(channels), 2) or not np.all(np.isfinite(positions))
            ):
                raise ValueError(
                    'channel_positions.npy must contain finite x/y model channel coordinates.'
                )
        for feature, indices in (
            ('pc_features.npy', 'pc_feature_ind.npy'),
            ('template_features.npy', 'template_feature_ind.npy'),
        ):
            if feature in self.arrays and indices not in self.arrays:
                raise ValueError(f'{feature} requires {indices}.')
            if feature in self.arrays:
                values = self.arrays[feature]
                lookup = self.arrays[indices]
                if lookup.ndim != 2 or values.ndim < 2 or values.shape[-1] != lookup.shape[1]:
                    raise ValueError(f'{feature} and {indices} sparse axes do not agree.')
                templates = self.arrays.get('templates.npy')
                if templates is not None and len(lookup) != len(templates):
                    raise ValueError(f'{indices} must align with templates.')
                if not np.issubdtype(lookup.dtype, np.integer):
                    raise ValueError(f'{indices} must contain integer indices.')
                limit = (
                    len(self.arrays.get('channel_map.npy', []))
                    if feature == 'pc_features.npy'
                    else len(lookup)
                )
                for start in range(0, len(lookup), self.batch_size):
                    rows = lookup[start : start + self.batch_size]
                    if np.any((rows < -1) | (rows >= limit)):
                        raise ValueError(f'{indices} contains invalid sparse indices.')
        return self
