# Unit metadata and display feedback: seven-item acceptance

This follow-up implements Carl's three-page **PR1428 - unit metadata across
curation rounds.pdf**, received 25 October 2026. Changes remain in PR #1428's
`feature/long-recording-curation` branch. No phylib or exporter change is needed.

| PDF item | Behavior and evidence |
| --- | --- |
| 1. Source IDs and filenames | Schema descriptors preserve source basenames, CSV/TSV delimiters and column ordering. `cluster_si_unit_ids.tsv` remains plural with header `si_unit_id`; new merge/split IDs have blank scalar source IDs. Complete parent metadata is retained in history. |
| 2. Stale pipeline metrics/classifications | Membership-policy events clear imported fields and quality on new membership. Explicit new-unit labels survive unchanged rounds; prior decisions remain historical. GUI, journal, checkpoint and export use the same projection. |
| 3. Parameter-file growth | One final managed override block replaces itself; only recognizable old generated suffixes are removed. Original Python code, comments, CRLF, encoding cookies and BOMs survive. Raw-data sentinel strings normalize to Python `None`. |
| 4. Eight-unit cap | Default 32, saved nonnegative setting; zero shows all and any positive integer is accepted. Display IDs and cache capacity are independent; truncation never changes merge membership. |
| 5. Legend | Removed outright from every metric axis. Authoritative cluster/Merge colors are retained. |
| 6. Camera action | Unit Stability captures its actual Qt FigureCanvas, retaining the usual screenshot path/naming/return contract. Regression tests trigger the screenshot action and inspect the PNG. Real capture also calls that same view method. |
| 7. Rate readability | Default saved width 0.6 points. Dataset-local display bins default to native resolution and permit aligned coarsening. Counts/exposure and finite measurement sums/counts add before division; zero exposure remains unavailable and final short edges are retained. |

## Metadata compatibility and policy

Imported values remain unchanged for untouched units, including legacy empty
strings and typed JSON values. New producer values are blank even when parent
values agree. New memberships are unreviewed; an explicit `705 = good` applies
to unit 705 until it is transformed again. No producer SNR, classifier, or
violation metric is recomputed or copied from a preferred parent.

Layout descriptors validate filenames, columns, duplicate fields and collisions.
Typed JSON/table disagreement fails with file, field and unit context. Read-only
source aliases may be symlinks; output writes refuse symlink targets. Legacy
omitted-null rows and Python scalar representations remain readable, and old
blank values retain their canonical revision hashes.

Version-1 event meaning is preserved. `BoundedSession.migrate_metadata()` records
an explicit policy transition, recovering source names from the exact original
source identity and revision. Saving creates a new immutable generation without
changing the base manifest, assignments, allocation IDs or review ledger. It
clears ambiguous resolved values on active derived units while retaining explicit
label journal intent. Already exported values without their earlier journal
cannot establish curator intent; they remain imported values rather than being
retroactively called explicit decisions. The guide includes migration usage.

## Real-data acceptance

The fixture is Carl's earlier converted engineering test export, computation
`b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`: 469,725,567 spikes, 705 original
units and a 112.859-hour elapsed clock. Its exposure declaration remains a test
conversion, not independently verified acquisition truth. Carl's newly mentioned
500-hour export was not available locally.

Fresh outputs exercise:

1. Accept `[40,44,48] -> 705` and `[49,50,51,53] -> 706` through the GUI
   proposition workflow; assert producer fields and quality clear before explicit
   good labels and a custom curator note are recorded.
2. Re-prepare, split 705 into 707/708 by a source-amplitude predicate, assert new
   membership values clear, then explicitly label both children.
3. Re-prepare, merge 707/708 into 709, assert clearing again, then label 709.

The independent Qt-free oracle checked every one of the 469,725,567 assignments
in **each round**, all unchanged source metadata cells, blank producer cells,
explicit group labels, and the two accepted review records. All 45 original
table filenames survive. The 91 remaining propositions remain pending. Params
are exactly **892 bytes in all three outputs**, with one managed block.
History grows from 5 to 8 to 11 records rather than losing parent decisions.

The same oracle verified all **67 original download files** were unchanged and
**12 immutable files per output** matched the converted source. A separate copy
of the earlier reviewed bounded package was migrated: its manifest, assignments
and accepted reviews were unchanged; ambiguous producer values became blank and
explicit good labels on 705/706 survived. The data-only process exited zero.

The combined GUI/data harness completed all assertions and printed PASS, then
exited 143 (SIGTERM). The signal's source is unknown; no traceback or verified
product termination path was found. This is retained as a combined-harness
limitation, not described as a clean process exit. Independent data verification,
the test suites, and both standalone capture processes exited normally. The
original independent checker also needed a harness-only fix to flatten column
cluster arrays before combining them with scalar amplitude masks; saved outputs
were unchanged and all rows were subsequently rechecked.

## Display and regression checks

Refreshed real captures use units 24, 26, 27 and 32, with no legend and 0.6-point
rate lines. The actual camera image is included in
`docs/images/unit-metadata-refinements/`. A generated **513-hour, twelve-unit**
fixture uses 6156 native five-minute bins and exact ten-minute display coarsening
to 3078 bins. It is synthetic stress coverage, not Carl's 500-hour data.

Focused coverage includes metadata aliases/multi-column CSVs, typed values,
safe names and conflicts, blank legacy revisions, immutable old generation
reopen, migration identity, snapshot saves, policy event versions, provenance,
undo/redo, encoding/BOM/CRLF preservation, all display settings, actual camera
capture, cache limits, nonuniform-grid rejection, zero-exposure groups, merged
weighted means and final partial bins.

Local verification ran `make test-full` (499 core + 431 application tests, one
existing benchmark skip), then the application target after two added symlink
regressions (433 passed). The final parameter/replay run passed 23 cases,
including two added encoding regressions. This covers 934 distinct current
tests; full CI runs the final tree together. Lint and format checks passed.
Documentation and wheel checks are recorded with the final handoff.

Local reports/scripts are under
`/home/cyrille/GIT/IBL/phy-data/carl/aind-b1b0782a-b3ea-4f34-81d7-b3c6e36452b3/pdf-metadata-validation/`.
Captures are in sibling `pdf-refinements-screenshots` and
`pdf-refinements-synthetic-513h` directories. Arrays and credentials are not
repository artifacts.
