
import csv
import glob
import logging
import os
import os.path as op
import shutil

import numpy as np
import scipy.io as sio

from phy.io.array import (_concatenate_virtual_arrays,
                          _index_of,
                          _spikes_in_clusters,
                          )
from phy.traces import WaveformLoader
from phy.utils import Bunch

logger = logging.getLogger(__name__)


def read_array(path):
    arr_name, ext = op.splitext(path)
    if ext == '.mat':
        return sio.loadmat(path)[arr_name]
    elif ext == '.npy':
        return np.load(path, mmap_mode='r')


def write_array(name, arr):
    np.save(name, arr)


def load_metadata(filename):
    """Load cluster metadata from a CSV file.

    Return (field_name, dictionary).

    """
    dic = {}
    if not op.exists(filename):
        return dic
    # Find whether the delimiter is tab or comma.
    with open(filename, 'r') as f:
        delimiter = '\t' if '\t' in f.readline() else ','
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header.
        _, field_name = next(reader)
        for row in reader:
            cluster, value = row
            cluster = int(cluster)
            dic[cluster] = value
    return field_name, dic


def save_metadata(filename, field_name, metadata):
    """Save metadata in a CSV file."""
    import sys
    if sys.version_info[0] < 3:
        file = open(filename, 'wb')
    else:
        file = open(filename, 'w', newline='')
    delimiter = '\t' if filename.endswith('.tsv') else ','
    with file as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(['cluster_id', field_name])
        writer.writerows([(cluster, metadata[cluster])
                          for cluster in sorted(metadata)])


def _dat_n_samples(filename, dtype=None, n_channels=None, offset=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(filename) - offset) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def _dat_to_traces(dat_path, n_channels=None, dtype=None, offset=None):
    assert dtype is not None
    assert n_channels is not None
    n_samples = _dat_n_samples(dat_path,
                               n_channels=n_channels,
                               dtype=dtype,
                               offset=offset,
                               )
    return np.memmap(dat_path, dtype=dtype, shape=(n_samples, n_channels),
                     offset=offset)


def load_raw_data(path=None, n_channels_dat=None, dtype=None, offset=None):
    if not path:
        return
    if not op.exists(path):
        logger.warning("Error while loading data: File `%s` not found.",
                       path)
        return None
    assert op.exists(path)
    logger.debug("Loading traces at `%s`.", path)
    return _dat_to_traces(path,
                          n_channels=n_channels_dat,
                          dtype=dtype if dtype is not None else np.int16,
                          offset=offset,
                          )


def get_closest_channels(channel_positions, channel_index, n=None):
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    return out


def from_sparse(data, cols, channel_ids):
    """Convert a sparse structure into a dense one.

    Arguments:

    data : array
        A (n_spikes, n_channels_loc, ...) array with the data.
    cols : array
        A (n_spikes, n_channels_loc) array with the channel indices of
        every row in data.
    channel_ids : array
        List of requested channel ids (columns).

    """
    # The axis in the data that contains the channels.
    if len(channel_ids) != len(np.unique(channel_ids)):
        raise NotImplementedError("Multiple identical requested channels "
                                  "in from_sparse().")
    channel_axis = 1
    shape = list(data.shape)
    assert data.ndim >= 2
    assert cols.ndim == 2
    assert data.shape[:2] == cols.shape
    n_spikes, n_channels_loc = shape[:2]
    # NOTE: we ensure here that `col` contains integers.
    c = cols.flatten().astype(np.int32)
    # Remove columns that do not belong to the specified channels.
    c[~np.in1d(c, channel_ids)] = -1
    assert np.all(np.in1d(c, np.r_[channel_ids, -1]))
    # Convert column indices to relative indices given the specified
    # channel_ids.
    cols_loc = _index_of(c, np.r_[channel_ids, -1]).reshape(cols.shape)
    assert cols_loc.shape == (n_spikes, n_channels_loc)
    n_channels = len(channel_ids)
    # Shape of the output array.
    out_shape = shape
    # The channel dimension contains the number of requested channels.
    # The last column contains irrelevant values.
    out_shape[channel_axis] = n_channels + 1
    out = np.zeros(out_shape, dtype=data.dtype)
    x = np.tile(np.arange(n_spikes)[:, np.newaxis],
                (1, n_channels_loc))
    assert x.shape == cols_loc.shape == data.shape[:2]
    out[x, cols_loc, ...] = data
    # Remove the last column with values outside the specified
    # channels.
    out = out[:, :-1, ...]
    return out


class TemplateModel(object):
    n_closest_channels = 16

    def __init__(self, dat_path=None, **kwargs):
        dat_path = dat_path or ''
        dir_path = (op.dirname(op.abspath(op.expanduser(dat_path)))
                    if dat_path else os.getcwd())
        self.dat_path = dat_path
        self.dir_path = dir_path
        self.__dict__.update(kwargs)

        self.dtype = getattr(self, 'dtype', np.int16)
        self.sample_rate = float(self.sample_rate)
        assert self.sample_rate > 0
        self.offset = getattr(self, 'offset', 0)

        self.filter_order = None if getattr(self, 'hp_filtered', False) else 3

        self._load_data()
        self.waveform_loader = self._create_waveform_loader()

    def describe(self):
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))

        _print('Data file', self.dat_path)
        _print('Data shape',
               'None' if self.traces is None else str(self.traces.shape))
        _print('Number of channels', self.n_channels)
        _print('Duration', '{:.1f}s'.format(self.duration))
        _print('Number of spikes', self.n_spikes)
        _print('Number of templates', self.n_templates)
        _print('Features shape',
               'None' if self.features is None else str(self.features.shape))

    def spikes_in_template(self, template_id):
        return _spikes_in_clusters(self.spike_templates, [template_id])

    def _load_data(self):
        sr = self.sample_rate

        # Spikes.
        self.spike_samples = self._load_spike_samples()
        self.spike_times = self.spike_samples / sr
        ns, = self.n_spikes, = self.spike_times.shape

        self.amplitudes = self._load_amplitudes()
        assert self.amplitudes.shape == (ns,)

        self.spike_templates = self._load_spike_templates()
        assert self.spike_templates.shape == (ns,)

        self.spike_clusters = self._load_spike_clusters()
        assert self.spike_clusters.shape == (ns,)

        # Channels.
        self.channel_mapping = self._load_channel_map()
        self.n_channels = nc = self.channel_mapping.shape[0]
        assert np.all(self.channel_mapping <= self.n_channels_dat - 1)

        self.channel_positions = self._load_channel_positions()
        assert self.channel_positions.shape == (nc, 2)

        self.channel_vertical_order = np.argsort(self.channel_positions[:, 1],
                                                 kind='mergesort')

        # Templates.
        self.sparse_templates = self._load_templates()
        self.n_templates = self.sparse_templates.data.shape[0]
        self.n_samples_templates = self.sparse_templates.data.shape[1]
        self.n_channels_loc = self.sparse_templates.data.shape[2]
        if self.sparse_templates.cols is not None:
            assert self.sparse_templates.cols.shape == (self.n_templates,
                                                        self.n_channels_loc)

        # Whitening.
        self.wm = self._load_wm()
        assert self.wm.shape == (nc, nc)
        try:
            self.wmi = self._load_wmi()
        except IOError:
            self.wmi = self._compute_wmi(self.wm)
        assert self.wmi.shape == (nc, nc)

        self.similar_templates = self._load_similar_templates()
        assert self.similar_templates.shape == (self.n_templates,
                                                self.n_templates)

        self.traces = self._load_traces(self.channel_mapping)
        if self.traces is not None:
            self.duration = self.traces.shape[0] / float(self.sample_rate)
        else:
            self.duration = self.spike_times[-1]
        if self.spike_times[-1] > self.duration:
            logger.debug("There are %d/%d spikes after the end of "
                         "the recording.",
                         np.sum(self.spike_times > self.duration),
                         self.n_spikes,
                         )

        # Features.
        f = self._load_features()
        if f is not None:
            self.features = f.data
            self.n_features_per_channel = self.features.shape[2]
            self.features_cols = f.cols
            self.features_rows = f.rows
        else:
            self.features = None

        tf = self._load_template_features()
        if tf is not None:
            self.template_features = tf.data
            self.template_features_cols = tf.cols
            self.template_features_rows = tf.rows
        else:
            self.template_features = None

        self.metadata = self._load_metadata()

    def _create_waveform_loader(self):
        # Number of time samples in the templates.
        nsw = self.n_samples_templates
        if self.traces is not None:
            return WaveformLoader(traces=self.traces,
                                  spike_samples=self.spike_samples,
                                  n_samples_waveforms=nsw,
                                  filter_order=self.filter_order,
                                  sample_rate=self.sample_rate,
                                  )

    def _get_array_path(self, name):
        return op.join(self.dir_path, name + '.npy')

    def _read_array(self, name):
        path = self._get_array_path(name)
        return read_array(path).squeeze()

    def _write_array(self, name, arr):
        return write_array(self._get_array_path(name), arr)

    def _load_metadata(self):
        """Load cluster metadata from all CSV files in the data directory."""
        files = glob.glob(op.join(self.dir_path, '*.csv'))
        files.extend(glob.glob(op.join(self.dir_path, '*.tsv')))
        metadata = {}
        for filename in files:
            logger.debug("Load `{}`.".format(filename))
            field_name, values = load_metadata(filename)
            metadata[field_name] = values
        return metadata

    @property
    def metadata_fields(self):
        """List of metadata fields."""
        return sorted(self.metadata)

    def get_metadata(self, name):
        """Return a dictionary {cluster_id: value} for a cluster metadata
        field."""
        return self.metadata.get(name, {})

    def save_metadata(self, name, values):
        """Save a dictionary {cluster_id: value} with cluster metadata in
        a TSV file."""
        path = op.join(self.dir_path, 'cluster_%s.tsv' % name)
        logger.debug("Save cluster metadata to `%s`.", path)
        # Remove empty values.
        save_metadata(path, name,
                      {c: v for c, v in values.items() if v is not None})

    def save_spike_clusters(self, spike_clusters):
        """Save the spike clusters."""
        path = self._get_array_path('spike_clusters')
        logger.debug("Save spike clusters to `%s`.", path)
        np.save(path, spike_clusters)

    def save_mean_waveforms(self, mean_waveforms):
        """Save the mean waveforms as a single array."""
        path = self._get_array_path('mean_waveforms')
        n_clusters = len(mean_waveforms)
        out = np.zeros((n_clusters, self.n_samples_templates, self.n_channels))
        for i, cluster_id in enumerate(sorted(mean_waveforms)):
            b = mean_waveforms[cluster_id]
            out[i, :, b.channel_ids] = b.data[0, ...].T
        logger.debug("Save mean waveforms to `%s`.", path)
        np.save(path, out)

    def _load_channel_map(self):
        out = self._read_array('channel_map')
        assert out.dtype in (np.uint32, np.int32, np.int64)
        return out

    def _load_channel_positions(self):
        return self._read_array('channel_positions')

    def _load_traces(self, channel_map=None):
        traces = load_raw_data(self.dat_path,
                               n_channels_dat=self.n_channels_dat,
                               dtype=self.dtype,
                               offset=self.offset,
                               )
        if traces is not None:
            # Find the scaling factor for the traces.
            traces = _concatenate_virtual_arrays([traces],
                                                 channel_map,
                                                 )
        return traces

    def _load_amplitudes(self):
        return self._read_array('amplitudes')

    def _load_spike_templates(self):
        out = self._read_array('spike_templates')
        if out.dtype in (np.float32, np.float64):
            out = out.astype(np.int32)
        assert out.dtype in (np.uint32, np.int32, np.int64)
        return out

    def _load_spike_clusters(self):
        sc_path = self._get_array_path('spike_clusters')
        # Create spike_clusters file if it doesn't exist.
        if not op.exists(sc_path):
            st_path = self._get_array_path('spike_templates')
            shutil.copy(st_path, sc_path)
        logger.debug("Loading spike clusters.")
        # NOTE: we make a copy in memory so that we can update this array
        # during manual clustering.
        out = self._read_array('spike_clusters').astype(np.int32)
        return out

    def _load_spike_samples(self):
        # WARNING: "spike_times.npy" is in units of samples. Need to
        # divide by the sampling rate to get spike times in seconds.
        return self._read_array('spike_times')

    def _load_similar_templates(self):
        return self._read_array('similar_templates')

    def _load_templates(self):
        logger.debug("Loading templates.")

        # Sparse structure: regular array with col indices.
        try:
            data = self._read_array('templates')
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            n_templates, n_samples, n_channels_loc = data.shape
        except IOError:
            return

        try:
            cols = self._read_array('template_ind')
            logger.debug("Templates are sparse.")
            assert cols.shape == (n_templates, n_channels_loc)
        except IOError:
            cols = None

        return Bunch(data=data, cols=cols)

    def _load_wm(self):
        logger.debug("Loading the whitening matrix.")
        return self._read_array('whitening_mat')

    def _load_wmi(self):
        logger.debug("Loading the inverse of the whitening matrix.")
        return self._read_array('whitening_mat_inv')

    def _compute_wmi(self, wm):
        logger.debug("Inversing the whitening matrix %s.", wm.shape)
        wmi = np.linalg.inv(wm)
        self._write_array('whitening_mat_inv', wmi)
        return wmi

    def _unwhiten(self, x, channel_ids=None):
        mat = self.wmi
        if channel_ids is not None:
            mat = mat[np.ix_(channel_ids, channel_ids)]
            assert mat.shape == (len(channel_ids),) * 2
        assert x.shape[1] == mat.shape[0]
        return np.dot(np.ascontiguousarray(x),
                      np.ascontiguousarray(mat))

    def _load_features(self):

        # Sparse structure: regular array with row and col indices.
        try:
            data = self._read_array('pc_features').transpose((0, 2, 1))
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            n_spikes, n_channels_loc, n_pcs = data.shape
        except IOError:
            return

        try:
            cols = self._read_array('pc_feature_ind')
            assert cols.shape == (self.n_templates, n_channels_loc)
        except IOError:
            cols = None

        try:
            rows = self._read_array('pc_feature_spike_ids')
            assert rows.shape == (n_spikes,)
        except IOError:
            rows = None

        return Bunch(data=data, cols=cols, rows=rows)

    def _load_template_features(self):

        # Sparse structure: regular array with row and col indices.
        try:
            data = self._read_array('template_features')
            assert data.dtype in (np.float32, np.float64)
            assert data.ndim == 2
            n_spikes, n_channels_loc = data.shape
        except IOError:
            return

        try:
            cols = self._read_array('template_feature_ind')
            assert cols.shape == (self.n_templates, n_channels_loc)
        except IOError:
            cols = None

        try:
            rows = self._read_array('template_feature_spike_ids')
            assert rows.shape == (n_spikes,)
        except IOError:
            rows = None

        return Bunch(data=data, cols=cols, rows=rows)

    def _get_template_sparse(self, template_id):
        data, cols = self.sparse_templates.data, self.sparse_templates.cols
        assert cols is not None
        template_w, channel_ids = data[template_id], cols[template_id]
        # Remove unused channels = -1.
        used = channel_ids != -1
        template_w = template_w[:, used]
        channel_ids = channel_ids[used]
        # Unwhiten.
        template = self._unwhiten(template_w, channel_ids=channel_ids)
        template = template.astype(np.float32)
        assert template.ndim == 2
        assert template.shape[1] == len(channel_ids)
        # Compute the amplitude and the channel with max amplitude.
        amplitude = template.max(axis=0) - template.min(axis=0)
        best_channel = np.argmax(amplitude)
        b = Bunch(template=template,
                  amplitude=amplitude,
                  best_channel=best_channel,
                  channel_ids=channel_ids,
                  )
        return b

    def _get_template_dense(self, template_id):
        """Return data for one template."""
        template_w = self.sparse_templates.data[template_id, ...]
        template = self._unwhiten(template_w).astype(np.float32)
        assert template.ndim == 2
        amplitude = template.max(axis=0) - template.min(axis=0)
        best_channel = np.argmax(amplitude)
        channel_ids = get_closest_channels(self.channel_positions,
                                           best_channel,
                                           self.n_closest_channels)
        template = template[:, channel_ids]
        assert template.ndim == 2
        assert template.shape[1] == channel_ids.shape[0]
        b = Bunch(template=template,
                  amplitude=amplitude,
                  best_channel=best_channel,
                  channel_ids=channel_ids,
                  )
        return b

    def get_template(self, template_id):
        if self.sparse_templates.cols is not None:
            return self._get_template_sparse(template_id)
        else:
            return self._get_template_dense(template_id)

    def get_waveforms(self, spike_ids, channel_ids):
        """Return several waveforms on specified channels."""
        if self.waveform_loader is None:
            return
        out = self.waveform_loader.get(spike_ids, channel_ids)
        assert out.dtype in (np.float32, np.float64)
        assert out.shape[0] == len(spike_ids)
        assert out.shape[2] == len(channel_ids)
        return out

    def get_features(self, spike_ids, channel_ids):
        """Return sparse features for given spikes."""
        data = self.features
        _, n_channels_loc, n_pcs = data.shape
        ns = len(spike_ids)
        nc = len(channel_ids)

        # Initialize the output array.
        features = np.empty((ns, n_channels_loc, n_pcs))
        features[:] = np.NAN

        if self.features_rows is not None:
            s = np.intersect1d(spike_ids, self.features_rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(s, self.features_rows)
            # Relative indices of the non-null rows in the output features
            # array.
            rows_out = _index_of(s, spike_ids)
        else:
            rows = spike_ids
            rows_out = slice(None, None, None)
        features[rows_out, ...] = data[rows]

        if self.features_cols is not None:
            assert self.features_cols.shape[1] == n_channels_loc
            cols = self.features_cols[self.spike_templates[spike_ids]]
            features = from_sparse(features, cols, channel_ids)

        assert features.shape == (ns, nc, n_pcs)
        return features

    def get_template_features(self, spike_ids):
        """Return sparse template features for given spikes."""
        data = self.template_features
        _, n_templates_loc = data.shape
        ns = len(spike_ids)

        if self.template_features_rows is not None:
            spike_ids = np.intersect1d(spike_ids, self.features_rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(spike_ids, self.template_features_rows)
        else:
            rows = spike_ids
        template_features = data[rows]

        if self.template_features_cols is not None:
            assert self.template_features_cols.shape[1] == n_templates_loc
            cols = self.template_features_cols[self.spike_templates[spike_ids]]
            template_features = from_sparse(template_features,
                                            cols,
                                            np.arange(self.n_templates),
                                            )
        assert template_features.shape[0] == ns
        return template_features
