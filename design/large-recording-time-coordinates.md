# Event coordinates and acquisition exposure

This document defines the narrow time-coordinate extension for the AIND export
reported against PR #1428. It supplements the
[large-recording contract](large-recording-curation.md) and its
[integration plan](large-recording-integration-plan.md). It does not authorize
changes to an upstream export or establish that this particular chronic
recording is ready for curation.

## Supported coordinates

Phy-facing event times are nonnegative integer coordinates on one monotonic
axis. Two explicitly distinguished meanings are supported:

- `acquisition_samples`: sample indices on the acquisition axis supported by the
  original contract.
- `elapsed_ticks`: elapsed seconds from the source recording's time origin,
  multiplied by a declared tick frequency and quantized to integers. Gaps remain
  visible on this axis. A tick is a time coordinate, not a recorded voltage
  sample.

The optional producer declaration is a Python dictionary in `params.py`:

```python
sample_rate = 30000.0
waveform_sample_rate = 30000.0
time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': 30000.0}
num_samples = 12188787510
# Illustrative values only; the producer must supply authoritative exposure.
valid_intervals = [[0, 100000], [200000, 12188787510]]
```

The same declaration can occur as `recording.time_base` in
`_phy_spikes_subset.manifest.json`. If both locations declare it, they must
agree. `kind` is one of the two strings above; `ticks_per_second` is positive
and finite. Unknown kinds and conflicting declarations are errors.

For this implementation, `ticks_per_second`, `params.py`'s `sample_rate`,
`waveform_sample_rate` when declared, and the waveform manifest's
`recording.sampling_frequency_hz` when declared must be equal. This keeps the
existing loader and waveform time axes correct for the reported 30 kHz export.
Supporting unequal event tick and physical waveform sample frequencies requires
separate waveform-axis plumbing and is outside this extension.

The physical and event frequencies nevertheless have different meanings. In
elapsed mode, `sample_rate` converts event ticks to seconds; the waveform
frequency describes the spacing between samples inside each stored waveform.
Clock drift or a piecewise clock map must not stretch the waveform's local
sample axis.

Without an explicit declaration, retain the original acquisition-sample
behavior and identity representation for existing sources. A source containing
`timestamps.json` must provide an explicit time-base declaration before this
bounded workflow accepts it. This is an intentional conservative admission
rule: the file's presence signals acquisition metadata whose exposure semantics
have not been established by the legacy defaults. It is not proof that every
such source actually has mixed coordinates. The error must explain how to
declare authoritative coordinates and exposure; it must not parse the JSON and
guess a conversion.

## Bounds and exposure

For both kinds, the legacy names `num_samples` and `valid_intervals` describe
the canonical event axis:

- `num_samples` is its exclusive positive integer upper bound. In elapsed mode
  it is a tick extent, not the number of acquired frames.
- `valid_intervals` contains ordered, disjoint, nonempty integer `[start, end)`
  intervals inside that extent. They declare valid acquisition exposure on the
  same event axis.
- Every spike belongs to a valid interval. Time ties and original source row
  order remain allowed and preserved.
- Exposure in seconds is the sum of interval lengths divided by the tick
  frequency, counted once across units. Gaps and filler are not zero firing.

An explicit elapsed-time source needs authoritative exposure rather than a
silent `[0, num_samples)` default. The producer must exclude missing acquisition
time and filler that does not contain usable recording data. If sorting spikes
remain in excluded intervals, reject the source with an actionable error. Do
not silently drop, move or relabel them: these actions change source row
identity or scientific interpretation. A future flagged-spike policy needs its
own full-data counting, selection and replay contract.

The upper bound must include the producer's final representable event tick.
An endpoint mapping may describe the last sample rather than an exclusive end;
do not blindly substitute its rounded clock span for `num_samples`. A physical
pipeline frame count can be retained as separately named producer provenance,
for example `recording.num_frames`, but cannot also define the elapsed extent.
`--duration` remains a consistency check and does not override conflicting
producer metadata.

Reject a first or last source spike outside the declared extent before a full
validation/hash scan. Still perform the complete batched validation for internal
reversals, out-of-range rows and invalid exposure membership. The early check
is a diagnostic improvement, not a replacement for validation.

## Waveform and window sidecars

In explicit elapsed mode, the waveform manifest must declare the coordinates of
each present time-bearing sidecar:

```json
{
  "format": "phy_spikes_subset",
  "format_version": 1,
  "recording": {
    "sampling_frequency_hz": 30000.0,
    "num_samples": 12188787510,
    "time_base": {"kind": "elapsed_ticks", "ticks_per_second": 30000.0},
    "window_bounds_coordinates": "elapsed_ticks",
    "sample_index_coordinates": "elapsed_ticks"
  }
}
```

The example omits `valid_intervals`; a real elapsed export must declare the
authoritative intervals in `params.py` or `recording`, with agreement when both
declare them. `window_bounds_coordinates` is required when window bounds are
present; `sample_index_coordinates` is required when sample indices are
present. Their values equal `time_base.kind`. A contradictory
`recording.frame_coordinates` declaration is an error, not an invitation to
reinterpret an array. Existing producer prose describing frames before
clock conversion must be corrected explicitly for an elapsed export.

The sidecars have these meanings and validation obligations:

- `_phy_spikes_subset.window_bounds.npy`: `int64[W, 2]` event-coordinate
  `[start, end)` intervals, ordered and nonoverlapping, within the declared
  extent and valid exposure. A window cannot bridge an unsupported interruption.
- `_phy_spikes_subset.window_types.npy`: `uint8[W]`, with `0` survey and `1`
  rescue.
- `_phy_spikes_subset.window_strata.npy`: `int32[W]`, nonnegative survey
  stratum IDs and `-1` for rescue. The three window arrays are aligned.
- Optional `_phy_spikes_subset.sample_index.npy`: `int64[N_waveforms]`, aligned
  with waveform rows, containing canonical event-center coordinates. For every
  waveform row `w`, it equals
  `spike_times[subset_spikes[w]]`. It is not a source spike row ID or a raw-file
  sample offset.

The established waveform triplet remains authoritative for actual waveform
availability. A window denotes potential evidence, not a waveform for every
spike in it. When sidecars are supplied as producer evidence, waveform centers
must lie in the declared window union. Reject malformed sidecars rather than
falling back to an independently generated plan that hides the error.

The existing row spaces remain unchanged: full spike row `g`, bounded spike row
`b`, waveform row `w`. Remap waveform row IDs through the selected full-to-bounded
mapping, and gather sample-index, waveform and channel rows together. Replay
preserves the original full arrays byte for byte. Supporting a sample-index
sidecar that intentionally remains in pipeline-frame coordinates is a separate
future provenance extension; this implementation does not infer its mapping.

Preparation should prefer validated producer evidence windows while keeping
whole windows, the configured spike/byte budgets, and deterministic omission
reporting. Do not enlarge memory or disk budgets merely because many waveforms
were exported. Report the source, retained and omitted waveform counts so a
small bounded package cannot appear to preserve the complete reservoir.

## Raw voltage access

Elapsed event ticks must never flow into raw voltage indexing. This first
extension rejects an elapsed source with a real raw-data path until a separate
explicit event-to-raw-frame mapping and its validation exist. Absent raw paths,
including an export's literal `None` sentinel, must be recognized consistently.
The bounded package continues to use stored waveforms without raw data.

Do not replace raw indices with elapsed ticks in an exporter and then claim
ordinary raw-trace loading works. Converting waveform centers for this contract
does not convert the physical voltage stream or establish a reversible map.

## Identity, replay and metadata preservation

Explicit time-base metadata participates in coordinate identity alongside the
canonical extent, exposure, immutable arrays, channel geometry and existing
acquisition provenance. Changing the coordinate kind, tick frequency or
exposure must invalidate a bounded package prepared against different metadata.
Keep the legacy identity shape unchanged for sources without a declaration.

Carry explicit metadata into bounded `params.py` and the bounded manifest.
Validate it when reopening and replaying, rather than trusting a free-form field
that is never consumed. Preserve it in complete replay outputs and their
lineage. Repreparing a replay output must produce the same coordinate identity.
Assignment revisions remain distinct from that identity. Coordinate changes
cannot be repaired by retaining a journal against an obsolete source identity.

Optional upstream sampling-policy windows do not by themselves change immutable
spike identity, but their validation and package hashes must bind the exact
evidence that the GUI uses. A waveform-aligned sample-index array does carry
immutable row provenance and must be included in the corresponding hashes and
copy/gather paths.

## Why Phy does not convert the supplied timestamps JSON

The supplied file contains 420 frame/clock endpoints for 210 pieces. Frame
endpoints increase strictly; clock endpoints are nondecreasing with 15 equal
adjacent pairs. Its metadata additionally reports underlying backward timestamp
steps that are not represented by those endpoint pairs. Consequently, the
endpoint map does not establish an invertible or sample-accurate raw clock.

Interpolation can reproduce the reported exported spike ticks without proving
the exposure between endpoints. Piecewise dropped samples, filler and true
gaps need producer knowledge. Mapping just the endpoints of a window that
crosses a gap would incorrectly declare the gap observed. The reported
`total_dropped_within_pieces_s` also does not locate every unsupported interval.
Neither the apparent 30 kHz unit nor monotonic exported spikes makes elapsed
ticks equivalent to physical acquisition samples.

The producer owns clock conversion and the final validity mask. Phy validates
the resulting explicit contract. Acquisition logs remain necessary to resolve
the reported interruptions and the 35 sorting spikes inside filler before
declaring this export suitable for exposure-aware statistics.

## Regression and acceptance gates

Before considering the extension complete, require concrete checks for:

1. Unchanged legacy preparation, identity, summaries, GUI behavior and replay;
   explicit acquisition-sample sources also round-trip.
2. Explicit elapsed data with tied spikes, long `int64` coordinates, gaps and
   endpoint boundaries; exact independently calculated full counts and exposure.
3. Missing/conflicting time declarations, unsupported rates, raw paths,
   contradictory frame metadata, and first/last out-of-bounds spikes; errors
   identify the conflicting coordinate meanings before the expensive scan.
4. Spikes in filler/excluded exposure, internal reversals and invalid interval
   boundaries; none are silently dropped or accepted as valid exposure.
5. Malformed or partially supplied producer sidecars, invalid dtypes/lengths,
   overlapping windows, windows crossing gaps, waveform centers outside windows,
   and sample-index values inconsistent with their source spike rows.
6. Budget-constrained producer-window selection with exact retained/omitted
   waveform reporting and deterministic complete-window selection.
7. Prepare, merge/label, save/reopen, replay and reprepare preserving explicit
   metadata, immutable rows and waveform alignment. Mutating time metadata or
   sample-index provenance must invalidate the appropriate identity/checksum.
8. Window-local ISI/CCG calculations excluding unsupported cross-boundary pairs,
   physical waveform sample spacing, and no elapsed-coordinate raw indexing.

These establish software behavior. They do not establish that endpoint-derived
exposure is scientifically precise or that the finite waveform evidence is
sufficient for chronic unit identity. Domain-curator acceptance remains required
by the existing validation plan.
