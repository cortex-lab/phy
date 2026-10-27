# Executable curation equivalence contract

Carl's acceptance requirement is that applying supported decisions to retained
evidence and replaying them over the full source produces the same complete
curation as applying those decisions to the full recording directly. A useful
regression must exercise actual controllers and saved files, including spikes
that were neither retained nor drawn.

This follow-up stays on PR #1428's `feature/long-recording-curation` branch.
The separate empty-waveform feature projection fix belongs to
[phylib PR #70](https://github.com/cortex-lab/phylib/pull/70); bounded mode does
not require a dependency bump for that guard.

## Permanent controller test

`phy/apps/large_recording/tests/test_controller_equivalence.py` creates a finite
660-spike, six-unit sorting with an explicit acquisition gap, imported producer
metadata and sparse stored waveforms. Three window/budget/seed configurations
retain genuine subsets; every active unit has unseen source spikes.

One shared action runner drives ordinary TemplateController/Supervisor and the
manifest-selected bounded controller/Supervisor. It never substitutes manual
`BoundedSession.record_*()` calls for controller actions. The amplitude panel
receives real Qt Alt-right-drag events and the normal split QAction. Only five
points may be drawn while the parent contains more spikes. The source includes
amplitudes exactly equal to the requested threshold; they stay in the remainder.
Viewport calibration accommodates floating pixel transforms without changing
the spike values or bypassing the split action.

The same ledger exercises:

1. Merge, undo/redo, amplitude split, undo/redo, and merge one child with another
   source unit; explicit GOOD/MUA, noise, cleared classification and custom notes;
   undo/redo of reviewed labels, a new edit discarding the redo branch, save,
   close, reopen and save again.
2. Prepare the full replay for another round, merge and split again, review both
   new memberships, edit/undo/redo an explicitly seeded note, save and reopen.
3. Prepare/save/reopen/apply without edits. Complete NPY files and manual group
   and notes tables must be byte-identical to the preceding replay.
4. Prepare again and leave a separate merge unreviewed. The checker verifies
   bounded clearing and explicitly admits only supported native ancestor
   inheritance, rather than pretending these imported values must agree.

Expected reviewed labels are written independently in `EXPECTED_ONE/TWO`.
Each round compares full saved ordinary data with full bounded replay using the
original source and the actual final bounded package. Original inputs are hashed
before and after; GUI callbacks and mapped resources are explicitly cleaned up.
The GUI layout is scoped to amplitude and tables, not an all-view rendering test.

An early extra snapshot assertion exposed ordinary inherited-note undo behavior:
editing an inherited parent note on a derived unit and undoing it restores an
empty value, rather than the displayed inherited note. The ledger seeds an
explicit reviewed note before testing its edit/undo, matching the explicit-label
contract. That initial diagnostic is retained locally; it is not claimed as a
successful inherited-note undo check or fixed by this PR.

## File-by-file contract

The opt-in checker is `tools/curation_output_contract.py`, called by
`tools/compare_curation_modes.py compare --source ...`. The existing basic
comparison remains available for complete partition/selected-label comparisons.

| Output class | Required check |
| --- | --- |
| `spike_clusters.npy` | Every source row belongs to the same final partition through a two-way ID bijection. An independent chunked oracle also executes cumulative merges/split predicates on original source rows and checks exact replay IDs. Assignment dtype/vector-column representation and native allocated IDs may differ. |
| Immutable NPY arrays and producer sidecars | Every full array/file matches the original bytes, including times, amplitudes, features, templates, channels, waveform rows/channels/samples, producer window plans, timestamps and suggestion bytes when present. |
| Imported and curator metadata | All fields are checked. Untouched units retain source values. New identities clear bounded producer fields unless explicitly reviewed. Every surviving label event requires an independent final expectation, including clear/null. Native differences must be supported ancestor values or empty values and are reported individually. |
| Metadata files/schema | Replay preserves producer basenames, delimiters and columns. Native canonical field tables may supersede byte-identical source aliases. Entirely blank native source columns may retain retired rows only when their original file is byte-identical. Missing blank tables still fail. Typed sidecars and tables must agree. |
| Parameters | Original Python structure outside managed extent/exposure/raw-path overrides is preserved; evaluated parameter values and effective recording clock/exposure/raw reference are checked. This is not a general proof of executable Python equivalence. |
| Journal/lineage | Exact operation/version/clearing-payload/predicate schemas, journal hash, current parent/preparation binding, final identity/revision, cumulative allocated ancestry, retired identities, metadata snapshots and explicit label intent are checked. |
| Saved generation | Pointer, descriptor and all checkpoint file hashes bind the export's journal, metadata, schema, allocator, provenance and reviews to the saved session. Every retained row's full output assignment must equal its saved assignment. |
| Suggestions/reviews | Suggestion bytes and source-hash bindings survive; inherited decisions match semantically. This action ledger makes no new proposition decisions. Their separate acceptance procedure remains in force. |
| Native derived whitening inverse | When absent from source but materialized by ordinary Phy, it must equal the inverse of the source whitening matrix, or the channel-count identity when no matrix exists. It is not an arbitrary extra-file exemption. |
| GUI state/derived summaries | `.phy` caches and logs may differ. Optional generated cluster summaries must have the complete active catalog and agree with their manual labels. Other optional source snapshots are byte-checked. Unknown or missing assets fail. |

The checker does not call the bounded assignment projector or event metadata
interpreter. It uses the source format reader for admission and independently
executes cumulative membership intent and tracks clearing and label intent.
The assignment oracle reads original scalar attributes for every source spike,
including spikes omitted from every bounded package. It applies earlier-round
actions before later actions, skips history already incorporated in the supplied
original source, and compares the resulting IDs exactly with the replay. Split
child order, comparator equality and parent-local nonfinite policy are checked.
Memory use is bounded by the requested chunk size and action history.
Native producer inheritance
is constrained to supported ancestor values/empty values; it does not reconstruct
every intermediate largest-parent inheritance choice. Historical parent revisions
are recorded ancestry, not fetched historical files. The current parent and
saved generation are verified against the supplied final package.

`--source` without a package supports first-round checks but explicitly reports
unverified preparation/saved-generation bindings. The full acceptance command is:

```bash
uv run --no-sync python tools/compare_curation_modes.py compare \
  /data/ordinary-final /data/bounded-full-output \
  --source /data/original-source --bounded-dir /data/final-bounded \
  --expectations /data/expected-labels.json --chunk-size 1000000 \
  --output /data/full-output-comparison.json
```

The report path must be fresh and outside every input folder. Expectations map
replay unit IDs to actual planned final decisions, not values copied from the
export under test. Use the original source across repeated rounds. Immutable
input preservation is checked by the test's pinned before/after inventory; a
post-hoc CLI source path alone cannot establish that its caller never edited it.

## Corruption regressions and scope

`test_output_contract.py` includes positive canonical/native-alias/blank-column
and implicit-group cases, and deliberate corruptions of arrays, memberships,
manual labels, new/untouched producer values, source aliases, metadata layout,
parameters, missing/extra files, malformed provenance, journal payloads,
predicates, parent bindings, allocator state, retired identities, review binding
and generated catalog coverage. Corrupt metadata tests repair table/JSON/revision
consistency so policy checks must catch them. Journal tests repair the correct
canonical JSON hash including its trailing newline, including source-only checks.
A standalone CLI regression catches import/path problems outside pytest.

A two-merge case swaps derived unit IDs while preserving the same partitions
and identical labels. The basic comparison accepts that rename; the strict
assignment oracle rejects its now-incorrect recorded source-parent relationships.

A subsequent independent audit exposed a checker false acceptance: a split
threshold could be changed in both journal and provenance, with repaired
checkpoint hashes, while leaving the old assignments in place. Schema, ancestry
and saved-row agreement did not establish predicate semantics. The independent
all-source oracle closes that gap; the corruption is now a regression, including
source-only and cumulative-round checks. This was a validation gap, not evidence
that the normal curation producer generated an incorrect split.

These tests guard the supported workflows and declared exceptions. They do not
prove equivalence for every possible action sequence or arbitrary lasso/feature
cuts, which cannot assign unseen spikes in bounded mode. Controller tests use
finite source amplitudes; ordinary display sanitization of nonfinite values can
differ from raw bounded predicates. Existing raw predicate regressions test
nonfinite/equality policy separately.

## Real-data validation

The existing main-GUI versus PR comparison uses Carl's converted engineering
dataset: **469,725,567 spikes**, a 112.859-hour elapsed clock, and unverified
inferred exposure. It is not the newer 500-hour recording or native-export
acceptance. The stronger file stage passed against the hash-pinned all-row
mapping, checking **46 metadata fields**, **20 immutable files**, the actual
saved generation and full file inventory. It recognized the unchanged plural
SI-ID alias, three byte-identical blank source columns, and a valid native
derived whitening inverse. These are explicit file-format artifacts.

The initial strict run correctly surfaced a checker admission gap: ordinary
loading skips entirely blank imported columns, leaving their original files
unchanged after source IDs retire. A narrow byte-pinned rule and a forged-value
regression were added. The first standalone run also found a tool import-path
error, now covered by an actual subprocess CLI regression. Both diagnostic logs
are retained; neither is described as a product curation mismatch.

The final standalone CLI run passed over all **469,725,567 rows**, including
all 46 metadata fields, 20 immutable files, the exact saved generation and full
file inventories. It reported zero unexpected partition or manual-label
mismatches and zero producer-value differences for this particular saved pair.
The subsequent predicate-oracle run independently checked all 469,725,567
original rows through two merges and one threshold split, with no assignment
contradictions. The runtime checker is commit `eb7983f0`; the complete local
report is `contract-comparison-predicate-oracle-committed.json` in the acceptance
directory below.
The [committed report summary](curation-output-contract-result.json) records the
runtime revision, input hashes, checked fields/files and admitted native artifacts.

After the predicate-oracle fix, `make test-full` passed **1,041 tests** (499 core
and 542 application) with one existing benchmark skip, including lint and
formatting. The complete **52-case file-contract target** includes coherent
current/inherited threshold forgeries, an unseen-row-only contradiction, and
an already-curated source whose incorporated history must be skipped.
The **26 literal oracle cases** cover all four operators, equality, child
ordering, scalar vector/column attributes, wide allocated IDs, nonfinite policy
and mixed cumulative actions across chunk boundaries. Independent targeted
verification also passed all three controller plans and 18 comparison cases.
The Sol 6.1 high auditor independently reran the original forged-checkpoint
reproduction: it is now rejected at source spike 4, and the implementation review
found no further issue.
Documentation and cross-platform CI links are recorded at handoff. Local
reports and reproducibility scripts remain in the dataset's
`main-versus-pr-gui-acceptance/` directory; no source or prior baseline output was
modified by the checker.

The independent high-effort audit found and regression-tested missing blank
table coverage, journal/history agreement, new-unit ancestry, malformed history
and invalid event payloads. Its final read found no remaining must-fix issue.

## Live post-split summaries

Commits through `6a5c9824` add automatic background computation of exact counts and
available finite amplitude/depth summaries for split descendants. The service
uses native preparation bins and acquisition exposure. Immutable membership
expressions identify cached results independently of labels and allocated IDs;
merges add counts and scalar sums/counts. Source readers are validated once and
reused while their file and metadata fingerprints remain unchanged. A separate
bounded derived cache cannot change or prevent curation saves.

The controller schedules work after transaction completion, delivers results on
the GUI thread and maps results to current memberships. Undo/redo, rollback,
branching, hidden views, source relocation, accepted GUI closure and reopening
have regressions. Scalar full counts remain available if plot arrays are evicted.
Unavailable source/cache capacity produces an explicit status rather than a
sampled rate. Finite-sum comparisons allow floating-point reduction tolerance;
spike and finite-value counts must match exactly.

The actual GUI trial ran on a copy of Carl's earlier converted 112.859-hour
engineering dataset, with **469,725,567 spikes**, **113 native bins** and
unverified exposure. It split unit 49 at source amplitude **-341.797119140625**,
producing units 709 and 710 with **292,990** and **346,764** full spikes.
Unrelated control unit 50 retained **541,925** spikes and its rate curve during
the computation. Three earlier saved membership actions were present in the
copied package; this run added the new split and resolved the missing summaries.

First-job source validation, computation and GUI delivery took **8.331 seconds**;
the checked source arrays totaled **9,394,511,724 bytes**. A completed summary
lookup took **0.000603 seconds**. Hash preflight warmed the filesystem cache:
these timings are not cold-storage measurements or a latency guarantee for the
newer 500-hour recording. No responsiveness timing threshold is claimed.

An independent chunked replay checked every original source row and compared
the two children and control against the displayed counts and finite scalar
summaries. Counts matched exactly; float64 sums and means matched at the
declared tolerance. All required source-array hashes and all **99 original
package-file hashes** were unchanged. The [committed result](live-summary-result.json)
records revision, source bindings, thresholds, timings and comparisons. Full
local artifacts remain in `live-summary-acceptance-2026-10-27-final/` under the dataset.

These are actual application screenshots of that trial:

- [Before splitting](../docs/images/live-summaries/before-split.png): parent 49
  and control 50 have their original full-recording rate curves.
- [Recomputing](../docs/images/live-summaries/recomputing.png): both children
  explicitly show pending computation while the control curve remains.
- [After computation](../docs/images/live-summaries/after-split.png): both child
  curves and the unchanged control are available together.

Reproduce on a fresh output directory using:

```bash
xvfb-run -a -s '-screen 0 2400x1500x24' \
  env QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  uv run --no-sync python tools/capture_live_summaries.py \
  --bounded /data/existing-bounded --output /data/fresh-live-summary-trial \
  --unit 49 --control 50 --rate-only
```

The tool copies the package, makes the split only in that copy, captures the
three stages and verifies full-source summaries. Existing valid cache entries
are reusable offline; new computations need matching original source files.
The high Sol 6.1 review found and regression-tested unsafe cache payload
allocation, redundant source hashing, hidden-view refresh and cache-eviction
rescan hazards. Its final read found no remaining correctness blocker, and an
independent run of the 40 engine/service cases passed.

The full local suite passed **1,086 tests** (500 core and 586 application) with
one existing benchmark skip, including lint and format checks. After the final
cache-key performance change, all **40 engine/service cases** and **four live
GUI integration cases** passed again. Strict documentation checks and final CI
results are recorded at handoff.

## Parallel phylib follow-up

[phylib PR #70](https://github.com/cortex-lab/phylib/pull/70) returns the existing
zero-filled feature tensor when a request has no stored waveform overlap or
no channels, instead of attempting PCA on an empty array. Five real on-disk
regressions cover empty requests, no coverage, empty channels, partial coverage
and complete coverage. Three fail before the guard; **308 full-suite tests** and
cross-platform/minimum-SpikeInterface CI pass afterward.

A separate copied real sorting was reopened in unmodified main Phy with only
the isolated patched phylib on `PYTHONPATH`. It exited zero with **zero
tracebacks**, versus five Feature View errors in the original equivalent reopen.
Complete assignments and group/notes files remained byte-identical. Original
validation data and installed environments were preserved. This addresses the
recorded baseline renderer failure and does not change PR #1428's dependency pin.
