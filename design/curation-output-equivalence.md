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

Windows 3.12 CI exposed differing `stat()` and `fstat()` timestamp semantics,
consistent with [CPython issue 157671](https://github.com/python/cpython/issues/157671).
The source reader now captures path and descriptor fingerprints separately,
compares each against its own later samples, and retains common identity/size/
modification-time binding plus the original file hashes. Three regressions cover
different timestamps, changes reported by either provider and replacement
between sampling the path and opening it. All **44 engine/service cases** and
the **four live GUI cases** passed after this fix. Source-edit fault injection
also writes the mapped payload in place, since Windows prevents truncation of
an open mapping. Summary arithmetic and the recorded real-data results are
unchanged.

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


## Unit-indexed summary acceleration

The 1,000-hour feedback exposed full-source scan and first-session hashing costs.
The optional summary store groups exact source scalars and native bin IDs by
original unit. It is independently budgeted, atomically published outside the
bounded package, and attached through a small `.phy` reference. Existing package
manifests and saved curation generations are unchanged. Payload checksums are
verified only for consumed units before any result is published. This certifies
the prepared source snapshot, not the present contents of original files;
full-source replay and the unindexed fallback retain source validation.

At revision `9e4ac004`, the earlier converted 112.859-hour engineering dataset
contained **469,725,567 spikes** and 113 native bins. The reusable store occupied
**2,348,718,604 bytes**, took **47.389 seconds** to build (including required
source validation), and peaked at **335,151,104 bytes RSS** in a separate process.
The input arrays were read in bounded batches, without touching every page of a
recording-sized memory mapping or allocating a full permutation.

Splitting unit 49 at **-341.797119140625** read only its **639,754 original spikes**
(approximately 3.2 MB). The first indexed request took **0.01795 seconds**, and a
repeated uncached request **0.01683 seconds**. The updated source fallback took
**6.470 seconds** including required-source hashing, and **2.176 seconds** with
validated readers reused. Result-cache lookups and publication were disabled for
both paths, so these are computation timings rather than result-cache hits.
These measurements exclude GUI delivery and use uncontrolled OS cache state;
they are not cold-storage or 1,000-hour acceptance measurements.

Counts were **346,764** and **292,990**. Integer counts and finite counts matched
exactly. Float64 sums matched at `rtol=1e-9, atol=1e-6`. Besides comparing indexed
and full-scan engines, independent cumulative assignment replay traversed **all
469,725,567 original rows** and used histogram accumulation to verify the indexed
outputs. Top-level bounded-package file hashes were unchanged. The
[committed result](summary-store-result.json) records the measurements; full
artifacts remain under `summary-store-acceptance-2026-10-28/` in the dataset.

Regressions cover native-bin boundaries and acquisition gaps, every comparator,
nonfinite rejection and remainder membership, nested split/merge ancestry,
source-offline computation, template-predicate fallback, corrupt stores,
cancellation, interrupted and racing publication, disk failure, lost references,
Fortran-order depth coordinates, wide scalar dtypes, and actual sparse-file
seeks past `2**32` rows. The real GUI split/merge/undo/redo/save/replay/reopen test
runs with and without an index; its indexed variant forbids source access.
Progress ignores cancelled or undone memberships. The high-effort Sol audit's
identified issues were corrected and regression-tested; its final read found no
remaining must-fix issue.

Reproduce on a fresh copy of the bounded package:

```bash
uv run --no-sync python tools/benchmark_summary_store.py \
  --bounded /data/bounded-copy --store /data/new-summary-store \
  --output /data/summary-store-result.json --unit 49 \
  --threshold -341.797119140625 --dataset-label 'engineering dataset'
```

Cost scales with original ancestry, not surviving child size: a split of a merge
of every original unit can read the entire store. The compact 1,000-bin,
float32-amplitude layout requires roughly 29.2 GB for 4.87 billion spikes, before
optional depth. Carl's actual 1,000-hour dataset still needs owner-run cold/warm
split and build measurements. No universal one-second bound is claimed.


## Large-dataset scaling and GUI background preparation

The fixed implementation at `f2c53f69` was benchmarked serially on the local
Linux host (134,698,618,880 bytes physical RAM). The real 100-hour sorting has
418,167,522 spikes. The 200-hour and 1,000-hour engineering datasets repeat its
arrays two and ten times, offsetting time by 100 hours per segment. They test
scale, not independent biology or Carl's native 4.873-billion-spike export.
The assumed continuous exposure on the real legacy 100-hour input is also an
engineering declaration, not a new validation of acquisition timing.

Each case resolves both children together. Result-cache lookup and publication
are disabled, and each scenario gets a fresh engine. These are first-request
computation timings, excluding GUI delivery. OS page caches are uncontrolled;
repeated requests are recorded in the linked JSON. They are not cold-storage
measurements or latency guarantees on a laptop with less RAM.

| Dataset | Source spikes | Median unit | Largest unit | Nested median split | Merge two largest, then split | Merge ten typical, then split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real 100 h | 418,167,522 | 0.020 s | 0.215 s | 0.019 s | 0.447 s | 0.113 s |
| Synthetic 200 h | 836,335,044 | 0.038 s | 0.429 s | 0.038 s | 0.880 s | 0.248 s |
| Synthetic 1,000 h | 4,181,675,220 | 0.170 s | 2.137 s | 0.174 s | 4.738 s | 1.224 s |

The 1,000-hour median unit contains 4,481,760 spikes; the largest contains
77,538,160. The two largest ancestors together contain 154,940,400 spikes.
Merging ten typical ancestors produces 44,726,720 spikes. Query time still
scales with this original ancestry, including nested descendants.

One-time store creation took approximately **41 s / 2.09 GB** for 100 hours,
**83 s / 4.18 GB** for 200 hours, and **420 s / 25.09 GB** for 1,000 hours.
Peak builder RSS stayed between **347 and 353 MB**. Build measurements came
from separate earlier child processes; the frozen runtime rerun reused their
completed compatible stores. The 1,000-hour format uses two bytes per native
bin ID instead of one, explaining its larger bytes-per-spike footprint.

All cases matched a combined fully validated source scan and a separate
cumulative assignment replay using independent histogram accumulation. Every
original source row was traversed in each dataset: **5,436,177,786 rows** across
the three runs. Integer spike and finite-value counts matched exactly; trend
sums matched at `rtol=1e-9, atol=1e-6`. The combined source scan computes all five
scenario expressions in one pass; its duration must not be compared as though
it measured a single split. Top-level bounded input hashes were unchanged.
Reports record runtime source-component hashes, recipes, byte counts, and timings:
[100 h](summary-scaling/100h.json), [200 h](summary-scaling/200h.json),
[1,000 h](summary-scaling/1000h.json).

Reproduce on a fresh bounded benchmark package with:

```bash
uv run --no-sync python tools/benchmark_summary_scenarios.py \
  --bounded /data/bounded-copy --store /data/summary-store \
  --output /data/result.json --label 'identified dataset and provenance'
```

The GUI trial at `0da082b1` used a fresh copy of the earlier converted
469,725,567-spike package. Building, attaching and restoring the two split-child
summaries took **49.61 seconds** while curation remained unsaved. A 10 ms Qt
heartbeat had **9.62 ms median**, **10.64 ms p95**, and **333.68 ms maximum** across
4,659 intervals, excluding screenshot capture intervals. This supports normal
responsiveness; it does not establish a hard maximum event-loop latency. The
children contained **292,990** and **346,764** spikes. The store was adopted
without reopening. [GUI report](summary-store-gui-result.json) and actual
[prompt](../docs/images/summary-store-task/prompt.png),
[progress/ETA](../docs/images/summary-store-task/building.png), and
[completion](../docs/images/summary-store-task/ready.png) captures are committed.

The task runner has bounded latest-only progress, staged ETA, shared recording
I/O exclusion, cooperative cancellation and a short publication boundary.
Regressions cover cancellation/publication races, failed thread startup,
worker exits, detached QObject ownership, GUI deletion before controller close,
vetoed close, slow destination checks, newer attachment references, persisted
budgets, and automatic prompts retaining keyboard-shortcut ownership. The
ordinary/bounded controller equivalence and graphical K-split tests pass with
the new prompt behavior.

### Next interactive-latency improvement

One-time preparation cost is secondary to interactive split latency. The current
row store greatly reduces source I/O, but the 2.1-second largest-unit and
4.7-second largest-parent-merge cases remain targets for further improvement.
The high-effort Sol architecture audit recommends an optional amplitude range
accelerator, not merely more batching or a faster builder:

- Sort exact finite amplitudes within each original-unit/native-bin segment.
  Keep aligned raw depth when needed; track nonfinite membership separately.
- Compile supported merge/split expressions into disjoint amplitude intervals
  per original unit. Preserve inherited parent rejection obligations.
- Use small endpoint blocks and balanced aggregate range trees, avoiding
  cancellation-prone subtraction of large floating prefixes. Benchmark block
  sizes 64, 128 and 256 before selecting a default.
- Authenticate only consumed metadata pages, aggregate nodes and data blocks.
  Retaining a whole-unit hash scan would erase the latency improvement.
- Reproduce canonical evaluator comparison semantics. Raw `searchsorted` can
  disagree with float32 weak-scalar comparisons near a threshold. Counts must
  remain exact; document and test trend arithmetic tolerances.
- Keep the existing verified row store as fallback. Preflight extra final and
  scratch disk space; perform bounded-memory sorting and atomic publication in
  another adapter for the same GUI task runner.

This is an audited proposal, not an implemented accelerator or an asserted speed
result. Initial prototype targets are below 50 ms for typical warm queries and
below 200 ms for the largest-parent cases, including selective integrity work.
Cold-query behavior needs separate device measurements. The key acceptance
criterion is that query work no longer grows linearly with source spike count
when bins, ancestry and predicates are fixed.


## Authenticated amplitude-range acceleration (28 October 2026)

Implemented on the same PR1428 branch. Preparation now optionally sorts finite
raw amplitudes per original unit/native bin and builds bounded-block aggregate
trees. Queries compile split/merge membership into rank intervals and verify only
consumed pages against pinned SHA-256 roots. Original dtype comparisons, nonfinite
policies and rejection obligations remain unchanged. Counts/finite counts are
exact; conservative error bounds require existing float64 sum tolerance (1e-9
relative, 1e-6 absolute), otherwise verified rows are used. Replay/assignments are
unchanged. No recording-sized index array is built in memory.

The GUI reuses the generic background task manager: inspect → rows if missing →
attach rows → amplitude index → attach index. Existing rows can be upgraded while
the original source is offline. Late cancellation stops subsequent long stages;
completed rows remain usable. Each automatic attachment is fenced against newer
choices. Independent indexes can be reattached from File or CLI. Combined disk
budgets and temporary sorting space are checked before expensive preparation.

### Reproducible measurements

`tools/benchmark_summary_scenarios.py --amplitude-index PATH` uses the same five
scenarios as the earlier row-store benchmark. Every scenario uses a fresh engine;
a second query reuses authenticated pages but no completed summary result. OS page
cache is uncontrolled. These timings include summary computation and selective
integrity checks, and exclude GUI delivery/rendering. The 1000h source is ten
repetitions of Carl's real 100h source, **not** an independently acquired 1000h
recording; legacy 100h acquisition/exposure construction is not certified here.

| Scenario | Real 100h: first / repeat | Synthetic 1000h: first / repeat | Earlier 1000h row-only first |
| --- | ---: | ---: | ---: |
| median unit | 6.5 / 4.4 ms | 55.1 / 40.2 ms | 169.9 ms |
| largest unit | 11.4 / 5.9 ms | 115.5 / 57.2 ms | 2137.3 ms |
| nested split median | 8.1 / 5.7 ms | 73.6 / 54.2 ms | 174.4 ms |
| merge two largest then split | 24.3 / 12.3 ms | 244.2 / 120.1 ms | 4737.7 ms |
| merge ten typical then split | 53.7 / 38.8 ms | 524.2 / 540.1 ms | 1223.5 ms |

Both runs compared every native-bin count, finite-value count and scalar sum with
one validated full-source scan and a separate independent sequential semantic
curation replay over all source spikes. All comparisons passed; bounded-package
files were unchanged. The earlier row-only results are historical measurements,
not an interleaved controlled A/B trial. Machine details, runtime component hashes,
cache conditions and complete recipes are in
[100h results](summary-scaling/amplitude-index-100h.json) and
[1000h results](summary-scaling/amplitude-index-1000h.json). Runtime hashes identify
the tested implementation even where the recorded HEAD predates its commit.

The 4.18-billion-spike index added 20.27 GB to the existing 25.09 GB row store;
construction took 264.3 seconds at 314 MB peak RSS. The real 418-million-spike
index added 2.04 GB, built in 26.0 seconds at 235 MB peak RSS. Construction is an
offline cost; these figures are not interactive latency guarantees. A ten-unit
query exceeds the 64 MiB page cache in the 1000h run and takes about half a second
on repeats; broad merged ancestry still increases work.

The refreshed real converted 469,725,567-spike GUI trial completed row build,
index build, attachment and current child summaries in 84.9 seconds. Curation
remained unsaved. Its 10 ms heartbeat measured p95 10.70 ms and maximum 322.84 ms
excluding screenshot intervals: responsive overall, with occasional larger pauses.
See [GUI results](amplitude-index-gui-result.json) and the actual
[prompt](../docs/images/amplitude-index-task/prompt.png),
[row preparation](../docs/images/amplitude-index-task/building.png),
[indexing with ETA](../docs/images/amplitude-index-task/indexing.png), and
[restored child rates](../docs/images/amplitude-index-task/ready.png).

Focused regressions cover all comparators, weak float32 boundaries, int64/uint64
extremes, nested/merged memberships, nonfinite-only bins, depth, stable external
merge sorting, real ill-conditioned arithmetic fallback, selective corruption,
reattachment invalidation, cancellation, injected sort/hash/publication failures,
newer-choice conflicts, offline upgrade and GUI lifecycle/adoption. High-sol
read-only audits checked reader correctness, resource bounds and task publication
boundaries; their concrete findings were fixed and regression-tested.
