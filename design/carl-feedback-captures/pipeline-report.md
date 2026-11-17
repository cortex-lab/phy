# Real Carl feedback pipeline validation — 17 November 2026

PASS on code `9e7607059fdd4271e733acfc0d42bcc957728958`. This is an engineering validation of Carl's
converted AIND overlay, not acceptance of the original exporter or acquisition
exposure policy. The unmodified export's clock/exposure problem remains outside
this result.

The source has 469,725,567 spikes, 705 original units, 140,240 waveform rows and
93 SLAY suggestions. Both runs used 8,000,000 evidence-spike / 3,000,000,000-byte
caps, 1,800-second bins, 100,000 requested source rows per batch, 32 MiB total
row-packing/write allowance and 64 MiB total index memory allowance. Both retained
7,298,318 evidence rows and all 140,240 waveform rows.

| Operation | 1 worker seconds | 4 workers seconds | 1 worker peak RSS GiB | 4 workers peak RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Preparation | 31.75 | 26.92 | 15.16 | 15.18 |
| Summary rows | 61.97 | 59.17 | 0.23 | 0.26 |
| Amplitude index | 67.46 | 102.65 | 0.24 | 0.27 |
| Full replay | 53.86 | 52.51 | 26.67 | 26.59 |

These are single warm-cache sequential worker comparisons while independent GUI
validation ran concurrently. A 10.5-second independent evidence/exposure check
also overlapped the four-worker index. They are observed integration costs, not
controlled throughput benchmarks. RSS includes resident mapped source/output
pages and is not a scratch-budget measurement. Detailed callback-stage timings
and stage RSS are in `result.json`; the final callback stage can also include
publication and cleanup work. Four workers were slower for this real index
layout; keeping one worker as the default is supported by this run.

The prepared packages' 23 NumPy files, unit/count catalogs and producer document
matched exactly between worker counts. Row-store payloads and all three index
payloads plus their packed Merkle hash files were byte-identical. Independent
source-row comparison checked every bounded spike field, all waveform values,
channels and mapped center coordinates in both packages. Bin edges matched an
independent construction; exposure agreed within 1e-10 seconds against the
**declared test intervals only**.

A copied package exercised the ordinary controller's new multi-suggestion path:
`_activate_merge_propositions(view, (keyA, keyB))`, then the ordinary merge action.
The groups `[40,44,48]` and `[49,50,51,53]` formed one ordered seven-unit union and
merged to 705. Both accepted decisions retain all seven applied units and result
705; their displayed status is `accepted_modified`, as each applied union differs
from its individual suggestion. Selecting the rows alone changed neither
assignments nor reviews. A source-amplitude split at
`-378.4182434082031` created selected child 706 and remainder 707, both
labeled good. Save/reopen preserved assignments, events, labels and both review
records. There are two accepted-modified and 91 pending suggestions. This ran in
a private Xvfb display, without desktop windows, and the child exited cleanly.
Its peak process RSS was 1.41 GiB.

The independent oracle used only original seven-unit membership and raw amplitude
`< threshold` comparisons with nonfinite values retained on the remainder. It
checked all **469,725,567 assignments in each of the two full replays**, without
calling production projection helpers. It found 1,691,273
selected and 1,597,610 remainder spikes and 700 active
units. Independent dense histograms verified exact per-unit/bin counts and exact
finite amplitude counts; amplitude sums agreed at rtol 1e-9, atol 1e-6. Both outputs
preserved every unchanged canonical NumPy file's bytes and producer/review
records. Repreparation took 28.43 seconds
and preserved full counts, labels and reviews.

Independent SHA-256 inventories before and after matched all 134 named source
entries: 67 original files and 67 overlay entries, resolving to 71 unique files.
The inventory also binds resolved paths, symlink targets, inode/device, lengths,
mtime and ctime. All 18 replayed canonical arrays per output have distinct source
inodes and link count one. The mutable working assignment file is independent of
the pristine prepared package. Original source files were unchanged.

Artifacts are all local to this directory: `validate_pipeline.py` (resumable
harness), `curate_offscreen.py`, `check_evidence_exposure.py`,
`check_output_inodes.py`, `summarize.py`, `result.json`, `progress.jsonl`,
`stdout.log`, `curation-offscreen.log`, both source inventories and supplemental
result JSON files. Preparation, row/index stores, isolated curation, both full
replays and repreparation are retained in separate new subdirectories. No runtime
repository edits, commits or GitHub writes were made by this validation task.
