# Preparing a Template GUI dataset

The Template GUI reads a directory of NumPy (`.npy`) arrays, optional
tab-separated metadata files, and a small Python configuration file named
`params.py`. KiloSort and other exporters commonly create these files
automatically.

Before editing an export by hand, try:

```bash
phy template-describe /path/to/output/params.py
```

This loads the dataset without starting the GUI and prints its dimensions,
recording duration, channel count, and available data.

## Minimal `params.py`

For a dataset with an accessible flat binary recording:

```python
# Path relative to this params.py, or an absolute path.
dat_path = 'recording.bin'

# Total number of interleaved channels stored in recording.bin, including
# channels omitted by channel_map.npy.
n_channels_dat = 384

# NumPy data type of every sample in the binary file.
dtype = 'int16'

# Number of header bytes before the first sample.
offset = 0

# Sampling frequency in hertz.
sample_rate = 30000.0
```

The binary is interpreted as time-major, interleaved samples with shape
`(n_samples, n_channels_dat)`. Its size after `offset` must therefore be divisible
by `numpy.dtype(dtype).itemsize * n_channels_dat`.

If the raw data is unavailable, phy can still open the sorting without a Trace
View or on-demand individual waveforms. Keep the required keys but use an empty
path list:

```python
dat_path = []
n_channels_dat = 384
dtype = 'int16'
offset = 0
sample_rate = 30000.0
```

`sample_rate` remains important because `spike_times.npy` stores sample indices
and phy converts them to seconds. `dtype` and `dat_path` must be present even
when `dat_path` is empty. `n_channels_dat` should describe the original
recording, whereas `channel_map.npy` selects and orders the channels shown by
phy.

`dat_path` may also be a list of consecutive binary files:

```python
dat_path = ['recording_part1.bin', 'recording_part2.bin']
```

Relative paths are resolved from the dataset directory. Prefer forward slashes,
raw strings, or `pathlib` paths for Windows paths so that backslashes are not
interpreted as Python escapes.

## Core arrays

These files form the smallest useful Template GUI dataset:

| File | Expected contents |
| --- | --- |
| `spike_times.npy` | One nondecreasing integer sample index per spike, shape `(n_spikes,)`. |
| `spike_templates.npy` | Template ID for each spike, shape `(n_spikes,)`, with an integer dtype. |
| `channel_map.npy` | Raw-data channel index for each displayed channel, shape `(n_channels,)`, with an integer dtype. |
| `channel_positions.npy` | Physical or display coordinates, shape `(n_channels, 2)`. |
| `params.py` | At minimum, the keys shown above. |

All per-spike arrays must use the same spike order and have `n_spikes` entries.
Spike times must be sorted in nondecreasing order. Template and channel IDs are
zero-based array indices in the usual exporter output.

`spike_clusters.npy` is the editable cluster assignment, also with shape
`(n_spikes,)`. It is optional only on the first launch: if it is missing, phylib
creates it by copying `spike_templates.npy`. The dataset directory must therefore
be writable. Thereafter, merges and splits change `spike_clusters.npy` while
`spike_templates.npy` retains the sorter's original assignments.

## Files that enable views and measurements

Most real sorting exports should also contain the following files:

| File | Shape and purpose |
| --- | --- |
| `templates.npy` | `(n_templates, n_samples, n_template_channels)` floating-point template waveforms; enables template and mean-waveform displays. |
| `amplitudes.npy` | `(n_spikes,)` amplitude or template scaling per spike; enables spike-amplitude plots. |
| `similar_templates.npy` | `(n_templates, n_templates)` similarity matrix used by the Template GUI's default similarity ranking. Larger values rank first; the score's scale is defined by the exporter. |
| `whitening_mat.npy` | `(n_channels, n_channels)` whitening matrix. |
| `whitening_mat_inv.npy` | `(n_channels, n_channels)` inverse whitening matrix used to display unwhitened templates. If absent, phy computes and writes it. |
| `pc_features.npy` | `(n_spikes_subset, n_channels_loc, n_pcs)` floating-point principal-component features used by the Feature View. |
| `pc_feature_ind.npy` | `(n_templates, n_channels_loc)` channel indices corresponding to PC features. |
| `template_features.npy` | `(n_spikes_subset, n_template_features)` projections onto nearby templates. |
| `template_feature_ind.npy` | `(n_templates, n_template_features)` template indices corresponding to template features. |

Depending on the exporter, a feature array may cover a subset rather than every
spike. `pc_feature_spike_ids.npy` and `template_feature_spike_ids.npy` identify
the represented spikes in that case.

For sparse templates, `template_ind.npy` has shape
`(n_templates, n_template_channels)` and maps each local template channel to a
displayed channel. Notice that the supported filename is singular
`template_ind.npy`; KiloSort's `templates_ind.npy` is not used as the sparse
mapping and its templates are treated as dense.

`channel_shanks.npy` and `channel_probe.npy` may assign each displayed channel to
a shank or probe. If absent, phy treats all channels as belonging to one shank
and one probe.

### Raw data and waveform subsets

An accessible `dat_path` enables the Trace View and lets phy extract individual
spike waveforms on demand. Without raw data, template waveforms remain available
when `templates.npy` exists.

A sorter or preprocessing step may instead provide all three of these files:

- `_phy_spikes_subset.waveforms.npy`
- `_phy_spikes_subset.channels.npy`
- `_phy_spikes_subset.spikes.npy`

Together they let phy display stored individual waveforms without reading the
original binary. They are a coordinated set: if any member is absent, phy skips
the subset.

## Cluster labels and other metadata

Cluster metadata uses a tab-separated file named `cluster_<field>.tsv`. It has a
`cluster_id` column and a column named after the field. For example:

```text
cluster_id	group
0	noise
1	good
2	mua
```

This is `cluster_group.tsv`. The common group values are `good`, `mua`, `noise`,
and `unsorted`. Additional files such as `cluster_kslabel.tsv` or
`cluster_contamination.tsv` become columns in the Cluster View. CSV files are
also read for compatibility, but TSV is preferred.

`cluster_info.tsv` is different: phy regenerates it as a convenient export of
the complete Cluster View when saving, and does not reload it as authoritative
metadata.

Any otherwise unknown `spike_<name>.npy` whose first dimension is `n_spikes` can
be loaded as a per-spike attribute. Plugins can use these attributes even when
the standard views do not.

## Data-dependent behavior

Missing optional data changes what phy can show:

- No raw binary: no Trace View and no waveform extraction from raw samples.
- No templates and no stored waveform subset: no useful Waveform View.
- No PC features: no Feature View.
- No amplitudes: no spike-amplitude series.
- No `similar_templates.npy`: the template-similarity matrix is all zeros, so
  the default Similarity View has no informative sorter-provided ranking.
- No whitening matrices: phy assumes identity whitening and may write a computed
  inverse.

The GUI can still be useful with a reduced dataset, but missing information
limits curation decisions. In particular, do not decide that two clusters are
the same unit from one view alone.

## Validate before curating

First preserve the automatic sorter's output. phy overwrites curation files when
you save and does not make an automatic backup:

```bash
cp -a /path/to/output /path/to/output-before-phy
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse C:\path\to\output C:\path\to\output-before-phy
```

Then run:

```bash
phy template-describe /path/to/output/params.py
```

Before opening the GUI, check that:

1. the reported sample rate, channel counts, spike count, and duration are
   plausible;
2. all per-spike arrays have matching lengths;
3. `spike_times.npy` is sorted and contains sample indices, not seconds;
4. `channel_map.npy` contains indices below `n_channels_dat`;
5. `dat_path`, if set, resolves to the intended raw recording;
6. the output directory is writable.

If validation fails, rerun the sorter's phy export when possible rather than
manually reshaping arrays. Start the GUI with `--debug` for a complete terminal
trace:

```bash
phy template-gui /path/to/output/params.py --debug
```

See [Troubleshooting](troubleshooting.md) for state, cache, graphics, and reporting
advice.
