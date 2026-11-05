# AIND export handoff for elapsed-time coordinates

This note describes the producer work needed before an AIND export can use
Phy's explicit elapsed-tick contract. It is a handoff, not evidence that an
exporter change has been implemented or that a real export has passed
acceptance. The exporter source repository was not available in this worktree;
the steps below are proposed commit-sized changes, with no repository or branch
assumed.

## Proposed exporter changes

1. **Establish authoritative acquisition truth.** Obtain acquisition logs or
   equivalent source records for clock origin, piece boundaries, dropped or
   invalid spans, filler, and the physical recording extent. Resolve the
   reported interruptions and the 35 sorting spikes reported inside filler.
   Do not infer validity solely from spike times or endpoint interpolation.
2. **Emit a canonical event time axis.** For each sorting spike, emit a
   nonnegative integer elapsed tick from the source time origin. Declare
   `time_base = {'kind': 'elapsed_ticks', 'ticks_per_second': ...}`. Set
   `num_samples` to the exclusive bound that includes the last representable
   event tick; retain a physical frame count separately (for example as
   `recording.num_frames`). Do not call a tick bound a frame count.
3. **Emit authoritative exposure.** Emit ordered, disjoint integer
   `valid_intervals` in the same elapsed-tick coordinates, excluding missing
   acquisition time and unusable filler. Ensure every emitted sorting spike
   belongs to a valid interval. If any sorting rows fall outside exposure,
   resolve them using acquisition truth upstream; never silently drop, move, or
   relabel rows.
4. **Emit aligned waveform and window evidence.** Store
   `_phy_spikes_subset.sample_index.npy` in canonical elapsed ticks, with each
   value equal to `spike_times[subset_spikes[w]]`. Emit producer windows in
   elapsed ticks, with boundaries that do not bridge unsupported intervals.
   If the windows overlap, retain their original boundaries and ensure their
   union accurately represents potential evidence. Add
   `recording.sample_index_coordinates: "elapsed_ticks"` and
   `recording.window_bounds_coordinates: "elapsed_ticks"` to the manifest when
   those sidecars are present. Add `recording.time_base` and authoritative
   `valid_intervals`; if `params.py` also declares them, keep the declarations
   identical.
5. **Keep raw voltage semantics explicit.** Do not put elapsed ticks into raw
   frame-index fields or claim ordinary raw-trace loading works. An export with
   a real raw-data path is outside this contract until an explicit validated
   event-to-frame mapping is supported. Stored waveform data can be exported
   without a raw path.

## Report evidence separately from verification

The supplied export report describes 420 frame/clock endpoints for 210 pieces,
with strictly increasing frame endpoints, nondecreasing clock endpoints, and
15 equal adjacent clock pairs. It also reports backward timestamp steps that
are not represented by those endpoints. These are reported properties of the
export metadata, not independent verification of the corresponding acquisition
events.

The reported `total_dropped_within_pieces_s` was not independently confirmed
against acquisition truth. Endpoint interpolation does not establish the exact
valid-time mask, a sample-accurate raw clock, or reversibility: equal clock
endpoints make the endpoint map non-invertible. The reported 35 spikes inside
filler likewise require independent confirmation and resolution. A 30 kHz
label, monotonic spike times, or a plausible endpoint-derived extent cannot
replace those checks.

Changing Phy's sampling seed (including setting it to `42`) does not make an
export interoperable. The seed only selects among windows already defined in
Phy's coordinate system; it cannot recover producer acquisition boundaries,
map clock coordinates to frames, or determine which spans contain valid data.
Use compatible producer windows and an explicit coordinate and exposure
contract.

## Acceptance evidence to return with an export

Provide the exporter commit or reviewable diff, a short provenance note naming
the authoritative source for boundaries and filler, and machine-checkable
checks showing:

- every exported spike's elapsed tick and valid-interval membership;
- the exclusive event-axis bound and separately named physical frame count;
- waveform-row alignment of sample indices to `spike_times[subset_spikes]`;
- producer windows' coordinate kind, valid exposure, and source waveform
  coverage;
- exact source, retained, and omitted waveform counts under Phy's unchanged
  512 MiB package and 2,000,000-spike budgets.

If the exporter removes spike rows, it must apply the identical row selection to
every spike-aligned array and rewrite any row-reference arrays such as
`subset_spikes` consistently. Silent removal leaves waveform and metadata rows
misaligned and is not acceptable. Phy can validate the explicit contract and
the exported arrays; only the exporter and acquisition evidence can establish
that the chosen exposure is scientifically authoritative. A local older
recording is not acceptance evidence for a new export.
