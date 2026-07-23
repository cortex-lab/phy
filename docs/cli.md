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

## Extract waveform subsets

```bash
phy extract-waveforms params.py 500 --nc 16
```

This saves a reusable subset containing at most the requested number of spikes per cluster and
the requested number of channels. It is useful when raw-data random access is expensive or the
raw data will not accompany a shared dataset.

## Convert to ALF

```bash
phy alf-convert sorter_output converted_output
```

Multiple input directories are interpreted as probes from the same recording and are merged
before conversion. Treat conversion as a separate data-processing operation and write to a new
output directory.
