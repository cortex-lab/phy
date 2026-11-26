# SLAY persistence validation

The follow-up to PR #1428 preserves producer `curation.json` bytes and the
complete source-bound review ledger through bounded preparation, committed
save, full replay, and re-preparation. Suggestions retain original IDs; pending
suggestions affected by merges or threshold splits become stale. Historical
accept/reject decisions take precedence over that derived status.

## Integrity and regression coverage

Preparation hashes the producer document and an immutable inherited review
snapshot. Session loading uses committed checkpoint reviews before loose review
copies. Direct saves and saves with the proposition feature disabled preserve
the ledger; suggestions without decisions receive an empty bound ledger.
Generation extra documents must belong to the checkpoint hash inventory.
Replay stages both documents, checks copied bytes, and publishes hashes and
cumulative merge/split retirement records in lineage.

Focused regressions cover three no-op rounds in direct, enabled and disabled
modes; accepted, rejected and orphan history; cumulative merge/split retirement;
malformed, conflicting or absent loose copies; mutated bound documents; missing
or null checkpoint ledgers; and mutations during copying. Old bounded packages
with unbound proposition artifacts fail explicitly instead of dropping history.
Ordinary unbounded loading keeps its existing compatibility behavior.

A separate Sol-high read-only audit identified copy-time mutation, legacy
silent-drop and null-ledger bypasses. Each received a fix and regression; the
final audit reported no remaining blocking findings.

## Real-data fixture and limits

The fixture is Carl's converted overlay of computation
`b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`, with 469,725,567 spikes,
705 original units, 140,240 waveform rows and 93 suggestions. It preserves the
33 filler-edge spikes using Carl's test-only exposure intervals. These intervals
have not been established as acquisition truth; this is an engineering test,
not acceptance of Brandon's native exporter. See
`aind-real-data-validation.md` for the original coordinate mismatch and exposure
limitations.

The roundtrip uses Carl's explicit 8,000,000-spike / 3,000,000,000-byte caps to
retain all waveform rows. Through the ordinary proposition-review and merge
path, it accepts `[40,44,48] -> 705` and `[49,50,51,53] -> 706` and marks both
results good. Subsequent rounds perform no additional curation.

The acceptance oracle requires two accepted and 91 pending suggestions in each
round, identical suggestion bytes and review records, save/reopen consistency,
and an independent streamed comparison of every full-source assignment against
the two original merge groups. Small regression fixtures provide overlapping
stale-suggestion coverage, because Carl's remaining 91 suggestions are disjoint
from these groups.

The separate validator run on this same test overlay passed with default
2,000,000-spike / 536,870,912-byte caps: 1,486,361 evidence spikes, 27,865 of
140,240 waveform rows retained, and 532,625,749 output bytes. It records that
the exporter revision is Carl's test conversion and that acquisition truth is
unestablished. A real native export and authoritative gap/filler policy are
still required; see `aind-native-export-acceptance.md`.

Local reports and reproduction scripts are retained under
`/home/cyrille/GIT/IBL/phy-data/carl/aind-b1b0782a-b3ea-4f34-81d7-b3c6e36452b3/slay-validation`.
Large arrays and access credentials are not repository artifacts.

## Results

- All three rounds retained exactly two accepted and 91 pending suggestions,
  identical producer bytes and review records, and matching assignments for
  all 469,725,567 source rows. Each bounded package retained all 140,240
  waveform rows and 7,298,318 evidence spikes at Carl's larger caps.
- The independent hash oracle verified all 67 original files were unchanged
  and 12 immutable files per full replay matched the converted source. Both
  good labels and the seven retired parent records survived every round.
- A separate final-code controller with proposition loading disabled saved and
  reopened the real third-round checkpoint with both accepted reviews intact.
- `make test-full` passed: 854 tests (499 core, 355 applications), with one
  existing benchmark skip. Lint and formatting passed. The corrected disabled
  controller tests subsequently passed all 22 persistence cases. `make doc-check`
  passed; packaging was unchanged.
- The combined three-round script completed every functional assertion and
  wrote its success report, then exited with signal 11 during process shutdown.
  A separate repeated six-controller/three-GUI lifecycle run with event
  processing and reference cleanup exited successfully. The combined harness's
  shutdown fault is retained as a limitation; its cause is not established as
  a product defect or a data-integrity failure.
