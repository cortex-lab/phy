# Native AIND export acceptance

This checklist accepts an export for Phy software integration only. Passing
checks do not prove scientific validity. The exporter owner must provide
authoritative acquisition evidence for the clock origin, physical extent,
valid exposure, dropped spans, and filler policy. A Phy run cannot supply that
evidence by inferring or converting timestamps or exposure.

## Before validation

- [ ] Record the exact Phy revision and native exporter revision in the report.
- [ ] Attach the reviewable exporter change and provenance for acquisition
  boundaries and filler classification.
- [ ] Confirm the exporter declares one explicit time base and emits
  `num_samples`, `valid_intervals`, waveform sample indices, and producer
  windows in that same coordinate system.
- [ ] Confirm every sorting spike is inside authoritative valid exposure and
  every waveform sample index equals `spike_times[subset_spikes]`.
- [ ] Resolve spikes inside filler from acquisition evidence upstream. Any
  intentional row removal must use the same row mask for every spike-aligned
  array and update row references consistently.
- [ ] Keep elapsed ticks separate from physical frame counts. Do not route
  elapsed ticks to raw frame-index fields or claim raw-trace loading is
  supported by this contract.

## Phy engineering validation

1. Run the native export through `tools/validate_aind_export.py` with the two
   revisions above, without an output directory. Retain its JSON report,
   including input file hashes, time contract, exposure, array counts, and
   timings. Resolve every source validation failure at the exporter.
2. Run optional preparation first with Phy's default budgets: 2,000,000 spikes
   and 512 MiB. Use a new destination outside the source directory. Record
   retained spike and waveform counts, omitted evidence, output size, and
   timings. Confirm the original source file hashes are unchanged.
3. If the complete producer evidence exceeds either default cap, run larger
   spike and byte caps as distinct, explicitly labeled runs, each to a fresh
   destination. Report each cap and its result independently; never treat a
   larger-cap success as a pass under the defaults.
4. Open the default bounded output in the Phy GUI. Exercise navigation at the
   beginning, middle, and end of retained windows, select units with and
   without retained stored waveforms, and verify displayed waveform/channel
   mapping and feature behavior.
5. Exercise the three-round SLAY carryforward workflow on the bounded output:
   save each round, carry assignments and labels forward, reopen the result,
   and compare final assignments and metadata with an independent expected
   result. Preserve the round-by-round artifacts and logs.
6. Recheck the native source hashes after GUI and carryforward work. All curation
   products must be separate from the source export.

## Decision record

- [ ] Phy contract validation passes with no timestamp or exposure inference.
- [ ] Default-budget preparation and output counts are recorded.
- [ ] Larger-cap runs, if needed, are separately reported.
- [ ] GUI and three-round SLAY carryforward evidence is attached.
- [ ] Acquisition authority has approved the clock, exposure, and filler
  policy independently of the software checks.

Record the outcome as **software integration accepted** only when the Phy
checks pass. Record **acquisition truth accepted** separately and only when the
authoritative producer evidence has been reviewed. The first outcome alone is
not scientific acceptance of the export.
