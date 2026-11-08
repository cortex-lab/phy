# Real AIND software diagnostic

## Conclusion

The downloaded export does **not** work unchanged with PR #1428: its time
coordinates and exposure declarations still need upstream correction. A
separately labeled diagnostic derivative exercised the complete Phy workflow
over 469,725,534 rows and exposed one additional Phy defect. That defect is
fixed, with portable regression coverage. Preparation, GUI curation,
save/reopen, full replay, independent verification, and repreparation now pass
on the derivative.

This is software integration evidence, not scientific acceptance of the
original export. The derivative's exposure is inferred from the JSON endpoint
map, and 33 filler-region spikes were explicitly quarantined. Those choices
require acquisition evidence and producer review before use for curation.

## Source and unchanged-export checks

The supplied download script selected computation
`b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`, rather than the earlier reported
`85640506-601d-4826-aa61-d3b822e53c84`. Its recording is
`experiment1_AmplifierData_recording1_group0` from session
`854393_2026-06-09_19-34-26`.

All 67 downloaded files matched the server's listed sizes, totaling
15,381,599,263 bytes. NumPy headers and payload sizes were checked. The export
contains 469,725,567 spikes, 705 declared units, 140,240 stored waveforms and
177 producer windows. Spike times are monotonic across the complete array.

The unchanged source fails `prepare-large` with the intended early
`timestamps.json requires an explicit time_base` diagnostic. It declares
10,800,000,000 pipeline frames, while its last spike is elapsed tick
12,188,787,421. Exactly 54,808,765 spikes are beyond the declared extent.
`params.py` and the waveform manifest lack an explicit elapsed-time contract
and valid acquisition intervals.

All 140,240 waveform centers match the JSON's frame-to-clock conversion within
0.5 tick, while their stored sample indices remain pipeline frames. This
reproduces producer interpolation; it does not establish sample-accurate
acquisition timestamps or exposure. Under that same map, filler regions contain
23, 4 and 6 spikes. These 33 rows differ from the 35 reported for the other
computation. Neither explicit large gap contains spikes.

## Diagnostic derivative and limits

The original export was retained unchanged. A separate source was constructed
solely to isolate software behavior:

- Declare the elapsed-tick contract at 30 kHz, an exclusive extent of
  12,188,787,510 ticks, and canonical sidecar coordinates.
- Copy exact `spike_times[waveform_ids]` into the waveform sample-index sidecar.
  Convert each producer `[a, b)` window to
  `[round(map(a)), round(map(b - 1)) + 1)` on the elapsed axis.
- Infer diagnostic exposure by excluding only the mapped explicit large gaps
  and filler regions. This does not locate unknown dropped frames within
  pieces or establish acquisition truth.
- Confirm that coordinate normalization alone still rejects the 33 filler
  spikes through the strict exposure-membership check.
- Quarantine those source rows explicitly, apply the identical row selection
  to every spike-aligned array, and remap waveform row references. No stored
  waveform references a quarantined row. Keep a provenance file listing every
  excluded source row and spike tick.

No conversion or filtering behavior was added to the production importer.
The diagnostic source, outputs, SHA-256 inventory, assumptions and one-off
reproduction scripts are retained beneath the downloaded computation's
`diagnostic-validation/` directory on the validation workstation.

## Observed Phy defect and fix

With the actual bounded data, merge, undo/redo, amplitude split and undo/redo
left NumPy integer child IDs as keys in split-event metadata. Projected session
metadata had been normalized, but the retained event had not. Saving failed
while serializing `journal.json` with `TypeError: keys must be str, int, float,
bool or None, not numpy.int64`.

Both split-event construction paths now normalize metadata before retaining
the event. This preserves IDs and the remainder/selected child order. Two
session API regressions reproduced the failure before the fix; a GUI regression
also exercises merge undo/redo, split, save and reopen. The focused
curation/lifecycle run passed 43 tests. Independent Sol-high review confirmed
the root cause and semantics of the fix.

## Full workflow results

Default preparation retained 1,486,361 spikes in 36 windows and 27,865 of
140,240 waveforms. Every unit has spike evidence; 702 of 705 have stored
waveform evidence. The package occupies 532,625,401 bytes, within the
536,870,912-byte cap. The complete producer-window plan is estimated at
2,554,394,546 bytes and 7,298,318 spikes, exceeding both default budgets.

All default bounded views opened, including waveform-derived features without
exported PCs. Selection included units 574, 578 and 579, which have no retained
stored waveforms. Navigation covered the beginning, middle and final retained
windows. Artificial merge and amplitude-split actions passed undo/redo, an
independent bounded assignment comparison, save and reopen, without logged
application errors. These actions are test operations, not biological curation.

Full replay wrote a separate output. An independent streamed oracle compared
**all 469,725,534 assignments**, using explicit merge masks and amplitude
comparisons rather than the production replay functions. Edge searches and
weighted/unweighted histograms independently checked the original per-bin
counts, finite amplitude counts and amplitude sums. Exposure arithmetic was
checked against the declared diagnostic intervals, without claiming their
scientific validity.

Every bounded spike row was checked against its source row. All 27,865 bounded
waveform rows, channel lists and center indices matched their mapped source
rows exactly. Replay preserved immutable array bytes and timestamp provenance.
Repreparation's complete unit counts matched the replay oracle. SHA-256 checks
confirmed that every file in the original download remained unchanged.

## Measurements and acceptance limits

Single workstation runs used Linux, Python 3.12.3, phylib 2.7.1, 100,000-row
batches, Xvfb/software OpenGL and Qt's offscreen platform. The workstation has
125 GiB RAM; source files were warm in the filesystem cache. Some later phases
ran concurrently with tests or another verification phase. These are observed
integration costs, not controlled throughput benchmarks or laptop predictions.

| Phase | Wall time | Peak process RSS |
| --- | ---: | ---: |
| Preparation | 50.1 s | 14.2 GiB |
| GUI actions and reopen | 9.1 s | 0.68 GiB |
| Full replay | 37.0 s | 25.0 GiB |
| Repreparation | 51.0 s | 14.1 GiB |

Offline RSS includes resident memory-mapped source/output pages. It is not
limited by the bounded package's byte budget. These measurements do not prove a
small process-RSS bound on a much longer source.

The original export remains blocked until its producer supplies consistent
time coordinates, authoritative exposure and a justified filler-spike policy.
See the [exporter handoff](aind-export-handoff.md). No acquisition logs or
verified exporter source repository were available, and this diagnostic does
not resolve timestamp precision within pieces or chronic unit identity.
