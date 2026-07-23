# Command-line reference

Run `phy --help` for the commands provided by the installed version, and
`phy <command> --help` for command-specific arguments.

## Version

```bash
phy --version
```

Development checkouts append Git information to the package development version.

## Template datasets

```bash
phy template-describe path/to/params.py
phy template-gui path/to/params.py
```

GUI commands accept:

```text
--clear-state     reset global and dataset-local GUI state
--clear-cache     remove and rebuild the dataset .phy cache
```

## Raw trace viewer

```bash
phy trace-gui recording.bin \
    --sample-rate 30000 \
    --dtype int16 \
    --n-channels 384
```

Additional options include `--offset` for a byte header and `--fortran` for Fortran-ordered raw
data. The Trace GUI is experimental.

## Legacy Kwik datasets

```bash
phy kwik-describe recording.kwik
phy kwik-gui recording.kwik
```

Both commands accept `--channel-group` and `--clustering`. The optional `klusta` and
`klustakwik2` packages are required for this legacy workflow.

## Pre-extract waveform subsets

```bash
phy extract-waveforms params.py 500 --nc 16
```

Run this command from the environment in which phy and phylib are installed,
before opening the GUI. It reads the raw binary named by `dat_path` and writes:

- `_phy_spikes_subset.waveforms.npy`
- `_phy_spikes_subset.channels.npy`
- `_phy_spikes_subset.spikes.npy`

The positional value (`500` above) is the maximum number of spikes sampled per
original template. `--nc` requests the number of best channels per waveform;
phylib may keep more when the model's minimum closest-channel count is larger.
Existing subset files with these names are overwritten. Keep all three files
together.

This is the supported way to precompute a reusable waveform **subset**. It is
useful when random access to the raw data is expensive, when the raw data will
not accompany a shared dataset, or when compressed raw data would otherwise
make on-demand extraction slow.

It does **not** pre-extract every spike in the recording. The current phylib
implementation samples from 20 representative raw-data chunks and applies the
per-template limit. There is no supported CLI option that materializes all
spike waveforms. Doing that can create a very large array; for example, one
million spikes × 82 samples × 16 channels at the extractor's default
`float64` output requires about 10.5 GB before filesystem overhead. Choose a
larger positional limit only when the resulting storage and extraction time
are acceptable.

After extraction, launch the GUI normally:

```bash
phy template-gui params.py
```

Phy detects the three files automatically. See
[Raw data and waveform subsets](dataset.md#raw-data-and-waveform-subsets) for
the dataset contract.

## Convert to ALF

```bash
phy alf-convert sorter_output converted_output
```

Multiple input directories are interpreted as probes from the same recording and are merged
before conversion. Treat conversion as a separate data-processing operation and write to a new
output directory.
