# Carl's curation requests: validation and limits

Implementation branch: `feature/carl-curation-feedback`, based on the long-recording
curation branch with current master merged before implementation. The original
request attachments were copied from MBA to local storage before implementation.
No request attachments or user data are committed here.

## Behavioral checks

- Bounded correlogram row transfer uses the same Supervisor operation as ordinary
  correlograms. Tests cover reference promotion, protecting the final staged unit,
  and close/reopen without duplicate callbacks.
- Ctrl-clicking SLAY suggestions changes only the comparison selection. The unit
  union is stable and deduplicated. Explicit G records all selected acceptances
  in one undoable operation; unrelated suggestions are never autoaccepted.
  Tests cover invalid/stale suggestions, failing preflight, cancel, assignment
  and review undo/redo, bounded save/reload, unchanged producer JSON, and replay
  against every source row.
- Parent inspection uses exact immediate parent membership. Its views have no
  curation actions and do not publish channel/time/transfer click events. Tests
  cover nested merges, splits, undo/redo, persistence, stale history, bounded
  predicate history, exact partition checks, and callback cleanup. See
  [the parent implementation note](parent-review-implementation.md) for limits.
- Performance tests compare serial/parallel arrays, row bytes, index bytes and
  Merkle data; sparse large IDs, NaNs, threshold ties, split/merge metadata,
  cancellation, hash mutation and deliberately corrupted clones have dedicated
  cases. Cloning never uses hardlinks and preserves existing destinations.

## Synthetic measurements

Measurements are single synthetic runs on this Linux machine (Intel i9-14900K,
CPython 3.12.13, NumPy 2.4.3), with cached files and `tracemalloc` active. They do
not measure Carl's 185 GB dataset, his machine, or cold-storage behavior. Traced
scratch is not whole-process RSS. Times are illustrative, not acceptance limits.

The lookup/histogram microbenchmark uses 10 million spikes, 1173 units, 2050
bins, and batches of 100,000. Replacing per-batch search and scatter with a
capped unit lookup and tiled histogram took approximately 0.115 s versus 1.344 s
(11.7×), with exact counts and sums. This isolates a kernel, not preparation as
an end-to-end command.

The dense pipeline fixture uses 8 million spikes, 16 units, two bins and 192 MB
of source arrays. The row/index configured global scratch budget is 256 MiB.

| Pass | One worker | Four workers | Ratio |
| --- | ---: | ---: | ---: |
| Source-file hashes | 0.1437 s | 0.0746 s | 1.93× |
| Row snapshot | 0.8706 s | 0.4717 s | 1.85× |
| Amplitude index | 1.5101 s | 0.7256 s | 2.08× |

Corresponding traced peak scratch bytes (one → four workers): hash 2,105,230 →
7,391,690; row snapshot 94,201,452 → 99,773,006; index 16,848,615 → 52,436,623.
Every compared output and Merkle byte was equal. In an earlier smaller fixture
(2 million spikes, 128 units, eight bins), four-worker row/index construction
was slower than serial. Consequently the default remains one worker.

Reproduce from a synced development checkout:

```bash
uv run --no-sync python design/benchmarks/summary_histograms.py
uv run --no-sync python design/benchmarks/bounded_parallel.py
```

Native stable integer grouping is retained. No claim about NumPy's internal
indirect sorting algorithm is required for correctness or performance.

## Remaining empirical limits

Headless tests establish view providers and curation state, not physical display
presentation. Ordinary merges made before lineage recording remain unavailable;
no template-based guess is substituted. Parent comparison is limited to 16
immediate parents, and plots obey existing evidence budgets. Bounded parent
views describe retained windows; full split-parent counts can be unavailable.

The follow-up [rendered real-data validation](carl-feedback-captures/README.md)
now covers Carl's earlier 469.7-million-spike fixture, including full replay and
matched baseline timings. His newer 5.24-billion-spike recording remains
unavailable locally, so its end-to-end improvement and worker choice still need
measurement there.

## Final integrated checks

On the combined implementation (including the cancellation and output-contract
follow-ups), the standard component targets completed with exit status zero:

- `make test UV_RUN='uv run --no-sync'`: 682 passed, 13 skipped.
- `make test-apps UV_RUN='uv run --no-sync'`: 788 passed, 6 skipped.
- Together these cover the `make test-full` test partitions: 1470 passed,
  19 skipped. Both targets include their required lint/format checks.
- `make doc-check UV_RUN='uv run --no-sync'` and `uv build` passed.
- Both checked-in benchmark scripts ran successfully and compared equal
  serial/parallel outputs. The wheel contains every new runtime module.

Two earlier coverage-enabled core runs completed all assertions and then exited
with signal 11 during interpreter shutdown. Fault tracing reported no Python
frame; the kernel log confirmed a native fault. The final standard core run
exited normally, as did debugger runs with and without coverage and with normal
address randomization. No native stack identified the cause, so no speculative
renderer or ownership change was made. A similar unresolved shutdown limitation
was already recorded in `slay-roundtrip-validation.md` before these changes.
This remains an intermittent harness/runtime limitation, not a claim that native
shutdown has been fixed or that the cause is established as harmless.


## Follow-up on real data

See [screenshots, complete recordings, replay and timing reports](carl-feedback-captures/README.md).
The rendered walkthrough found and fixed hidden parent controls and excessive
legend height. Application checks now pass 791 tests (6 skipped); the core
partition passes 682 assertions (13 skipped), but the native shutdown fault
recurred after that summary in the latest full-target attempt. We do not report
that invocation as a clean full-suite pass. Both real GUI processes and both
469,725,567-row replay pipelines exited normally, with unchanged source hashes.


## November 19: growing-journal curation follow-up

Carl and Andrew's logs contained two recoverable Python exceptions: an absent
optional parent-color configuration directory, and stale Similarity selection
after the final primary reference disappeared. Focused regressions cover both,
plus backward batch selection and reordering the blue reference independently
of its identity and color.

The performance PDF described 20,216,806 evidence spikes and 213 journal events
on a newer 6.75-billion-spike source. That exact checkpoint/source was unavailable.
The following engineering measurements use disposable fixtures on this Linux
machine, Python 3.12.13, NumPy 2.4.3 and phylib 2.7.1. OS caches were uncontrolled.
Raw profiles and user attachments remain outside Git; compact measurements and
revision attribution are in [the result JSON](benchmarks/carl-nov19-results.json).

### Isolated metadata actions

The synthetic fixture retains 20,216,806 rows across 859 source units. A fixed
100-merge prefix holds the live catalogue constant; labels extend the journal
to 100, 213 and 1,000 events. Three unprofiled label actions per case give:

| Initial events | Before median | After median | Full projections per label |
| --- | ---: | ---: | --- |
| 100 | 29.745 s | 0.802 ms | 3 → 0 |
| 213 | 29.717 s | 0.809 ms | 3 → 0 |
| 1,000 | 29.500 s | 0.855 ms | 3 → 0 |

These timings isolate supervisor actions without GUI tables/views. They show
that repeated assignment projection caused the dominant cost; they do not prove
all work is independent of journal/catalogue size. Event-reference and metadata
bookkeeping still scale with their respective small collections. Each measured
case also compares final assignments against the sequential reference projector.
Whole-process peak RSS, including fixture construction, was 4.00 GB before and
3.96 GB after; this is not an action-specific memory measurement.

### GUI actions

A guarded copy of the existing real-source bounded fixture retains 7,298,318
rows from 469,725,567 source spikes. Both runs seed 213 events including the
same 100 merges, use Evidence Amplitude and Unit Stability views, and measure
two unprofiled samples under private Xvfb/software Mesa. Both controller
threading settings are recorded. Readiness means the supervisor/view queues
plus one Qt timer turn, rather than physical frame presentation.

| Action | Before median | Final optimized median |
| --- | ---: | ---: |
| Move to good and next selection | 16.523 s | 0.794 s |
| Label | 15.844 s | 0.194 s |
| Merge | 7.912 s | 2.687 s |
| Undo | 7.966 s | 2.857 s |

Every GUI worker exited normally and all input package hashes were unchanged.
The final follow-up refreshes affected metadata rows and preserves unchanged
stability artists. Its focused tests retain filters, sorting, selection, custom
metrics and plugin events; membership edits and asynchronous summaries redraw.
Unknown or replaced metric providers refresh across all live units because their
values can depend on other clusters. Only unchanged built-in provider identities
use the narrower metadata refresh. Array-valued metrics retain table support.
Merge/undo still include metric refresh and selected-unit plot reconstruction.

Reproduce on a synced checkout with new output directories:

```bash
uv run --no-sync python tools/profile_bounded_actions.py --help
uv run --no-sync python tools/profile_bounded_gui.py --help
```

### Full-source correctness

Replay of the previously validated saved checkpoint on runtime code at
`22eed44f` completed in 52.99 seconds with one worker; complete validation took
98.62 seconds. This is a single warm-cache run, not a baseline speed comparison.
All **469,725,567 assignments** matched both earlier one-/four-worker outputs;
all 52 compared metadata/JSON files, lineage and rewritten parameters matched.
A separate original-label/raw-amplitude oracle also passed, including split
counts 1,691,273 and 1,597,610, with 700 final active units. All 156 named input
files retained their hashes and identity statistics. Runtime module hashes were
unchanged during the run. Peak process RSS was 26.68 GiB including mapped pages.

Incremental native edits, bounded inverse-history retention and compiled export
writing retain full validation on Save/load and an independent sequential
interpreter for export verification. Cache misses replay validated history;
the new fast path does not remove those fallback and integrity checks.


### Final local gates

On the integrated runtime (`c2d541f0`, followed by documentation/results only),
`make test-full UV_RUN='uv run --no-sync'` exited normally with **739 common and
893 application tests passed**, 19 skipped in total. It includes lint and
formatting checks. `make doc-check` and `uv build` passed, and the wheel contains
all new runtime modules. The locked environment uses PyQt5-sip 12.17.0.

The independently scoped shared `master` changes at `306667e4` passed their own
full suite (**673 common + 168 application tests**, 19 skipped), documentation,
lint and formatting checks with normal exits. The PR1428 branch incorporates
those shared changes without merging PR1428 into master.

The previous remote CI run at `8c232422` failed after successful pytest summaries
on four matrix jobs (Unix exit 139; Windows exit 1). These are recorded separately
from assertion failures. Local normal exits do not establish that this existing
remote runtime problem is fixed; the separate dependency-fix PR is unchanged.


### Sampled near-zero pair exclusion (PR1428 only)

The follow-up implements pair exclusion in Sampled Correlogram View only. It
filters raw integer-sample delays before histogram binning, independently for
auto and cross pairs, with inclusive boundaries and a default half-width of
0.3 ms. Both switches start off. No spike is removed from the retained trains;
all longer-lag pair contributions remain eligible. Window boundaries and the
existing central-bin conventions are preserved. The conditional-uniform chance
line uses the same pair-specific exclusion when evaluating off-center exposure.

Independent all-pairs and exhaustive discrete-clock oracles cover all four
switch combinations, partial bins, sample boundaries, ties, sparse/reversed IDs,
late clocks and observation gaps. View regressions cover legacy providers,
keyboard scope/customization, unchanged input arrays and curation history,
GUI-state restoration, and live merge/split/undo membership changes. Actual
canvas-click keyboard tests also pass under Xvfb/xcb with software OpenGL. The
visible status prefix retains active flags/width when the dock truncates text.

On the same 7,298,318-row fixture (175 windows), fixed selected units 40/44/48
contain 18,605 retained spikes. Three unprofiled repeats with 0.5 ms bins and a
50 ms correlogram window produced the following medians on this machine:

| Half-width | Disabled | Cross only | Auto only | Both |
| --- | ---: | ---: | ---: | ---: |
| 0.3 ms | 37.75 ms | 34.64 ms | 33.63 ms | 32.57 ms |
| 2 ms | 36.24 ms | 34.14 ms | 33.98 ms | 31.70 ms |

These are warm-mapping provider timings, not complete GUI/presentation timings
or a general speedup claim. All calls reuse the same read-only integer arrays;
input hashes are unchanged. Peak process RSS was 337 MiB, including mapped
pages and imports, rather than a per-case allocation measurement. The compact
benchmark JSON records the exact statistics module SHA256 and case results;
that module is committed in `d1018a63`. Existing ordinary Correlogram View,
source assignments, replay and master remain unchanged by this feature.


The integrated exclusion runtime at `d1018a63` passed `make test-full` with a
normal process exit: **739 common + 972 application tests passed**, 19 skipped.
The target includes lint and formatting checks. Strict `make doc-check` and
`uv build` passed. The eight UI/provider tests present during the real
Xvfb/xcb run passed; the additional live-membership oracle test also passes in
the full suite. Master remains at `306667e4`; this follow-up is PR1428 only.


### November 22 requests: recovery, annotations and RP investigation

The three PDFs downloaded on MBA were fetched read-only and retained with
SHA256 provenance under `phy-data/carl/feedback-2026-11-22`. The deduplication
request matches the existing sampled pair-exclusion feature. SLAY's requested
axis-label grouping is implemented separately from trace color semantics.

The RP investigation is an external prototype, not a shipped estimator. Its
12 tests establish aligned uint32 sample-offset encoding, authenticated paired
chunks, independent iterator cursors, 64-bit absolute clocks, and rejection of
inconsistent metadata. A five-million-row synthetic warm-cache trial added
20 MB of offsets to 30 MB of base records. It built in 0.153 seconds and read,
authenticated, decoded and oracle-compared in 0.088 seconds. These timings do
not measure deduplication, RP estimation, cold storage, or Carl's 6.75-billion-
spike recording. The real fixture's base record is five bytes rather than the
synthetic fixture's six; neither existing store was rewritten.

Two small counterexamples demonstrate unresolved producer conventions. With
spikes at samples 995 and 1005 and window [0,1000), attributing a pair to its
earlier bin counts a violation that a both-endpoints-inside rule excludes.
With spikes at 996 and 1004, amplitudes 1 and 2, and exclusion radius 9,
global amplitude-greedy deduplication leaves no spike inside that window,
whereas window-local deduplication retains one. Merged-parent violation counts
are also nonadditive. A display-only pair mask cannot implement this metric.

The next RP gate is the exact `phytastic` repository and commit used for the
figures, or its `export.py` and `rp_windows.py`, together with SpikeInterface
and DARTsort versions. Scoped local/MBA and connected-source searches did not
locate these files; no request was sent. The proposed next step is an independent
small-data oracle against that producer before implementing an optional exact-
time sidecar, full-train deduplication and a background RP panel. Existing raw
rate/count summaries must retain their definitions.

Draft protection captures immutable active-journal/catalog references only after
the outer curation transaction settles. A single worker writes coalesced
snapshots; two independently verified heads rotate by successful writes rather
than revision parity. Private files/directories, durable ancestor creation,
live-owner locks, checksum fallback, canonical-pointer conflicts, allocator
high-water marks, and explicit discard/retirement have focused regressions.
Checkpoint publication remains separate. Recovered work starts a new undo
baseline and stays dirty; saving retires the recovered source draft without
deleting a newer foreign head. Failed or uncertain Save outcomes do not invent
a successful protection acknowledgement.

The close-path review found that existing callbacks disconnected the workspace
before a later save prompt could veto closure. GUI state capture now runs in
`close_accepted`, after every veto and before geometry persistence; resource
cleanup runs in `closed`. Regression tests preserve Merge mode, staged selection,
parent inspection, controller selection hooks and logging after cancellation.

Native Xvfb/xcb screenshots on a new 12-spike synthetic fixture cover neutral
and amber states at 1100×700 and 1600×900. Actual checkpoint/draft acknowledgements
were used; only the reminder widget's clock advanced by 902 seconds. The fixed
status area was 520×26 pixels and its Save button 171×26 in all four captures.
The indicator remained outside the plots and the colored SLAY axes were visible.
Artifacts and exact code hashes are in
`phy-data/carl/recovery-visual-review-2026-11-22/final-002`.

An external 5 ms Qt heartbeat probe measured two consecutive worker writes per
case, including previous-head verification. At 10,000 valid label events, writes
took 28.4/41.0 ms with maximum observed UI gaps of 15.4 ms. At 100,000 events,
writes took 173.5/318.7 ms with maximum gaps of 76.5/75.8 ms; none exceeded
100 ms. This is a small warm-cache experiment, not a latency guarantee. Python
JSON encoding can briefly hold the GIL even on a worker; protection avoids
spike-array scans/copies on the GUI thread but is not strictly pause-free for
arbitrarily large journals. The probe script, raw gaps and code hashes are in
`phy-data/carl/recovery-validation-2026-11-22/heartbeat_probe.json`. No original
Carl data was read or modified by either recovery experiment.

The integrated changes passed `make test-full UV_RUN='uv run --no-sync'` with
the CI-matching Qt offscreen backend under Xvfb: **744 common + 1,016 application
tests passed**, 19 skipped, with normal process exits. This includes lint and
formatting. Two earlier full-native probes without a window manager encountered
shortcut/focus assumptions; they are not recorded as successful full-native
runs. The proposition text-focus assertion now explicitly waits for focus.
Native recovery, close-cancellation and screenshot checks passed separately.

A final indicator-only refinement distinguishes retained startup warnings from
actual protection failures: a warning shows “Draft needs attention” while a
genuine error shows “Protection failed.” All nine reminder/status tests were
rerun under native xcb after that refinement and passed; lint, formatting and
the package build passed again. Strict `make doc-check` passed for the published
documentation. The wheel includes all recovery and proposition-annotation
modules. Master remains `306667e4`; these changes target the existing PR1428.
