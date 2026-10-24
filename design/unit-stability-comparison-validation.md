# Unit Stability comparison validation

Carl's PR #1428 feedback requested independent metric visibility and comparison
of multiple clusters on long recordings. The view now overlays up to eight
selected units using authoritative selection colors, with saved firing-rate,
amplitude and depth toggles. The display limit does not change merge membership.
Existing preparation, export, journal and replay formats are unchanged.

## Data and screenshots

The real-data gallery uses the existing converted AIND test export from
computation `b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`. Its clock spans
112.859 hours including gaps, with 113 summary bins and 705 source units.
This is the earlier local export, not Carl's newly mentioned 500-hour recording.
The conversion's acquisition mask remains unverified; these are software
integration checks, not scientific validation of exposure.

Before and after captures select the same units, 40, 44 and 48, in a
2100 × 1200 Qt window. The baseline at `6ac9010d` displays only unit 40.
The new view displays all three, permits a full-height rate-only panel, and
preserves the same IDs and colors in Merge View. The authoritative table
presentation order is 40, 48, 44; colors follow that shared selection mapping.
Rate and amplitude are present; depth remains explicitly unavailable.

The screenshot tool opens a disposable copy of the bounded package and checked
all 72 input file hashes remained unchanged. Full counts read from the source
summaries are 612,536 (40), 207,903 (44), and 336,413 (48).

A separate generated 500-hour fixture contains 120,000 spikes, four units,
rate/amplitude/depth summaries and 1,000 bins. Each selected unit has exactly
30,000 source spikes. It supplies duration and missing-real-depth test coverage;
it must not be presented as Carl's 500-hour recording.

Both capture processes exited normally after explicitly draining deferred Qt
deletions. Captures use Xvfb, Qt's X11 platform and software Mesa. Single runs,
concurrent with the test suite, measured selection/block at 0.372 s on the AIND
package and 0.623 s on the synthetic fixture. The two rate-only redraws together
took 0.094 s and 0.529 s respectively. These are observations, not benchmark
guarantees or CI timing thresholds.

The gallery is in `docs/images/unit-stability-comparison/`. Reproduction options
are documented in `docs/long-recordings.md`; local capture manifests, source
hashes, scripts and logs are retained under
`/home/cyrille/GIT/IBL/phy-data/carl/stability-comparison/`.

## Regression coverage and review

Focused tests exercise exact per-unit rates with partial/zero exposure and a
short final bin, finite-value weighted merged means, pending split children
beside valid units, units without sampled evidence, row-at-a-time aggregation,
and absence of complete-spike-array reads in bounded mode. Rate-only loading
does not open trend arrays; toggle redraws reuse cached summaries.

Selection tests cover normal/Similarity and Merge ordering, authoritative
colors, initial attachment during Merge mode, deferred hidden-view updates,
empty selection, frozen updates, display truncation, and merge/split undo/redo.
All eight metric visibility combinations retain stable axes/callback counts;
saved flags and QAction checks survive closing and reopening the view.
Ordinary optional Unit Stability retains its continuous-acquisition assumption
and refreshes cached counts on cluster edits.

The implementation and existing adapter tests passed 25 focused cases.
An independent Sol 6.1-high audit found and reproduced a hidden-view palette
reset; the scoped fix and regression were included. Final review found no
remaining product blocker. A minimal standalone ordinary-GUI probe completed
all count assertions but exited 139 during interpreter teardown; repeating it
with explicit deferred Qt deletion exited normally. This remains a harness
cleanup limitation consistent with the earlier roundtrip record, not a verified
new product defect.

Final local validation: `make test-full` passed 864 tests (499 core, 365
application), with one existing benchmark skip; its lint and formatting checks
passed. `make doc-check` passed, including strict site generation and current
generated documentation. No packaging or dependency changes were made.
