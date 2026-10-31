# Curation-mode user acceptance protocol

Status: protocol for the graphical bounded curation workflow; run both GUI
branches against the same full-source data before accepting the feature.

Permanent controller-to-controller tests and the stronger whole-output checker
are now described in [the executable equivalence contract](curation-output-equivalence.md).
They supplement the manual protocol and the pinned main-branch GUI trial below.

## Purpose and evidence boundary

This is an end-to-end acceptance check for ordinary and bounded curation. It
compares their complete final spike assignments and explicitly reviewed manual
labels after applying the same user action ledger. It does not compare bounded
evidence arrays with full-source arrays, nor does it treat a streamed reference
as ordinary GUI evidence.
The harness comparison tests live in
`phy/apps/large_recording/tests/test_curation_comparison.py`.

The ordinary GUI run exercises the usual complete-data workflow. The bounded
GUI run exercises retained evidence, semantic action journaling, and full-source
replay. Separately, `tools/compare_curation_modes.py reference --engine
ordinary` executes full Supervisor commands without GUI views attached; it is
an ordinary engine reference, not evidence from the ordinary GUI. The
`--engine streamed` reference is a further independent streamed baseline, also
not GUI evidence.

## Freeze inputs and outputs

1. Pin the exact ordinary and bounded code revisions under review. Record both
   revisions and the comparison tool revision.
2. Preserve an immutable, complete snapshot of the source sorting. Create a
   separate writable working copy for the ordinary GUI, which saves in place.
   Record hashes for `spike_times.npy`, `spike_clusters.npy`, and
   `amplitudes.npy`. Do not use the bounded evidence package as the ordinary
   input.
3. Prepare the bounded package from the immutable source and retain its manifest,
   window plan, journal, and preparation report. Use fresh output directories
   outside the immutable source copies.
4. Use the same selected source units, threshold, action order, and explicit
   final manual labels in both GUI runs. Save each result and preserve its
   final full output. Keep screenshots or a short screen recording of the
   relevant preview and actions alongside the ledger.

For real data, record the dataset's actual extent and time axis from metadata.
Do not use a generated 500-hour example as evidence about a different real
recording. In particular, Carl's currently available converted dataset is
469 million spikes and 112.9 hours on its declared clock, with a frame value
near 100 hours; it is not a 500-hour dataset. Treat any newer source revision
as a separate input and recapture its metadata.

## Graphical action ledger

Record one shared `ledger.json`. The following schema illustrates the order and
the required alias mapping; replace example aliases, IDs, threshold, and labels
with those actually used. The actions describe user intent, including history
and save boundaries.

```json
{
  "aliases": {"unit_a": 40, "unit_b": 44, "unit_c": 48},
  "actions": [
    {"op": "merge", "parents": ["unit_a", "unit_b"], "as": "merged"},
    {
      "op": "split",
      "parent": "merged",
      "threshold": 1.5,
      "selected": "low",
      "remainder": "rest"
    },
    {"op": "undo"},
    {"op": "redo"},
    {"op": "label", "ids": ["low"], "field": "group", "value": "good"},
    {"op": "label", "ids": ["low"], "field": "notes", "value": "reviewed low side"},
    {"op": "label", "ids": ["rest"], "field": "group", "value": "mua"},
    {"op": "label", "ids": ["rest"], "field": "notes", "value": "reviewed remainder"},
    {"op": "label", "ids": ["unit_c"], "field": "group", "value": "mua"},
    {"op": "label", "ids": ["unit_c"], "field": "notes", "value": "reviewed unchanged unit"},
    {"op": "save"},
    {"op": "reopen"},
    {"op": "save"}
  ],
  "expected_labels": {
    "low": {"group": "good"},
    "rest": {"group": "mua"},
    "unit_c": {"group": "mua"}
  },
  "fields": ["group", "notes"]
}
```

Use the actual final unit identities returned by each run to fill the aliases.
In ordinary Amplitude View, select the **template** amplitude type, which reads
`amplitudes.npy`; feature and template-feature amplitudes define different
quantities. Run ordinary GUI acceptance with the graphical threshold interaction
(Alt plus right-drag, then `K`) and bounded GUI acceptance with the same threshold. Use
source units whose amplitudes are all finite for this graphical equivalence
check. Bounded splitting assigns finite values strictly below the threshold to
the selected side; equal and nonfinite values stay in the remainder. Require
evidence on both sides before committing. Test nonfinite values separately
through the raw source predicate engine: ordinary TemplateModel displays
nonfinite amplitudes as zero, so its graphical selection need not match that
raw predicate. The ordinary comparison engine deliberately uses raw source
`amplitudes.npy` values rather than displayed values.

Record intentional no-ops and state boundaries in the acceptance log alongside
the shared action ledger when relevant: for example an invalid threshold
rejected before mutation, an undo with nothing available, a no-change label, a
save/reopen/save cycle without edits, or a second save without edits. The
reference harness ledger currently supports merge, split, label, undo, redo,
save, and reopen; do not put unsupported no-op operations into its `actions`
array. Record the expected result (no mutation, rejection, or unchanged output)
so both GUI branches are checked consistently.

After the final redo and before concluding, assign explicit final `group` and
`notes` values to every transformed or deliberately reviewed final unit, plus
an unchanged control. Other untouched units retain and compare their source
labels. This keeps the comparison about reviewed manual labels, not
legacy producer-label inheritance. The protocol compares only the requested
manual fields. Producer metrics such as SNR may intentionally be cleared on
transformed memberships in bounded mode. Membership-scoped quality automation
may also give transformed units different automatic status; record and review
that behavior separately instead of treating it as a manual-label mismatch.

## Complete-output comparison

Export the ordinary GUI result and run `phy apply-curation` on the bounded
session to produce its complete full-source result. The bounded GUI assignment
at `K` covers retained evidence rows; replay applies its recorded predicate to
the unseen source spikes. Save both results to separate complete output
directories.
Put the JSON report in a fresh path outside both compared directories and both
immutable source copies. Compare the complete output directories (not the
bounded package or its retained rows):

```bash
uv run python tools/compare_curation_modes.py compare \
  /data/ordinary-final /data/bounded-replay-final \
  --fields group notes --output /data/curation-comparison.json
```

The report checks equal source row counts and exact row-aligned `spike_times`,
then proves partition equivalence using a two-way raw cluster-ID mapping over
all assignment rows in chunks. It checks the requested manual labels through
that mapping. A label attached to an ID with no assigned spikes is reported as
unmapped and fails acceptance; empty or unmapped label tables are not silently
ignored. `--fields` is an explicit allowlist: include the manual fields reviewed
in the ledger and do not add producer metrics or automatic quality fields whose
bounded membership policy intentionally differs. Review the full report,
including input hashes, ID mapping, examples, and label diagnostics. A zero exit
status and `ok: true` are required.

This check establishes final partition and chosen manual-label equivalence for
the pinned inputs and ledger. It does not prove identical intermediate UI
behavior, producer metadata, automatic quality calculations, or scientific
validity of the reviewed labels. Preserve GUI evidence and compare reports with
the pinned revisions and immutable source hashes.

## Practical user-test matrix

Start with one shared ledger, then add small variants rather than attempting
every combination in one session. These are proposed user checks; the observed
automated run is recorded separately below.

| Check | What to verify |
| --- | --- |
| Merge and review | Merge three units, split the result, then merge one child with another original unit. Explicitly assign GOOD, MUA, noise, a cleared/unsorted group and custom notes on final units. Try reversing merge-candidate order in a separate run. |
| History | Undo and redo a merge, split and label. Undo, make a different edit and confirm redo is discarded. Track the actual new IDs rather than assuming allocation reuse. |
| Thresholds | Compare graphical `K` with the exact recorded numeric threshold. Test equality, a rejected one-sided threshold, and selection/context changes that clear a preview. Test raw nonfinite values separately from ordinary graphical equivalence. |
| Persistence | Save, close, reopen and save without edits. Apply, prepare again, save without edits and apply again. Compare complete assignments and explicit manual labels after both rounds. |
| Sampling and gaps | Repeat a small ledger with a different seed, window plan or budget while both children retain evidence. Keep the same authoritative gap/exposure declaration and source predicate. The complete result should be independent of the displayed sample. |
| SLAY decisions | Separately accept and reject suggestions, transform an overlapping unit and check stale reasons. Preserve decisions through further rounds using the existing `design/slay-roundtrip-validation.md` procedure. The comparison reference ledger does not implement proposition operations. |

## Observed engineering acceptance, 26 October 2026

Runtime code and comparison tool were pinned at
`06637b24aeca9b0984d7928d459187df46406444` on PR #1428's existing
`feature/long-recording-curation` branch. The source was Carl's converted test
overlay for computation `b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`: **469,725,567
spikes**, 705 original units and a 112.859-hour elapsed clock. Its inferred
exposure remains unverified acquisition truth; this is engineering integration
evidence, not native-export acceptance or a run on Carl's newer 500-hour data.

The bounded package contained 1,486,361 evidence spikes. A native X11 Qt GUI
performed the real `Alt`-right drag and `K` action, then saved and reopened.
The shared [recorded ledger](curation-mode-acceptance-ledger.json) includes:

1. Merge 40, 44 and 48; undo and redo.
2. Graphically split the merged unit using the exact preview threshold
   **-424.0037077454673** in raw source amplitude units; undo and redo.
3. Merge the low-amplitude child with original unit 24.
4. Explicitly label that new unit GOOD, the remainder MUA and unchanged
   control 49 MUA, with custom notes. Change a label to noise, undo it, then
   branch with a different label and restore the intended final MUA value.
5. Save, reopen and save again without edits; apply to the full source.

The independent reference ran the same ledger through the complete ordinary
Supervisor/Clustering and ClusterMeta implementations without attaching GUI
views. It used raw source amplitudes for the predicate. Both paths used the
same PR revision: this was **not a main-branch versus PR ordinary GUI trial**.
The ordinary run finished in 3:31 with 29.8 GiB peak RSS and no swap, under a
96 GiB virtual-memory ceiling. The bounded GUI plus first replay took 53.3 s.
These are single warm-cache observations, not controlled benchmarks.

The independent complete-output comparison passed over **all 469,725,567 rows**:
exact row-aligned sample times, **zero partition mismatches**, a bijection over
703 final units, and **zero group/notes mismatches or unmapped label IDs**.
The numerical unit IDs also happened to agree; that is not required by the
partition test. Complete NPY file hashes for assignments differ because the
ordinary reference writes a vector and replay retains the source column shape;
the comparator normalizes both shapes and checks every assignment exactly.
The [committed report summary](curation-mode-acceptance-result.json) retains
the comparison method, revision, counts, diagnostics and input hashes.

A further preparation from that full output, direct session save, reopen,
no-edit save and replay completed in 121.8 s. All **18 complete NPY files**,
including assignments and spike times, and both explicit group/notes tables
matched the preceding replay byte-for-byte by SHA-256. This second cycle tests
persistence without edits; it is separate from the native GUI save/reopen test.
An independent hash verifier also confirmed that all **67 original download
files** remained unchanged, and that **19 immutable source files in each full
output** matched the converted input.

### Screenshots and reproduction

The three images in `docs/images/amplitude-threshold-workflow/` show a separate
disposable demonstration on the newly prepared real output, selecting unit 708.
They show its sampled overview before the drag, the orange threshold preview
with matching waveform highlights, and the two children after actual `K`.
The exact threshold is **-180.09097605065006**, with retained children of 3721
and 3643 spikes and 34 highlighted waveform traces. This is a separate split
from the shared comparison ledger. All 88 files in the input bounded package
retained their hashes; the capture operates on a temporary copy.

```bash
QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -s '-screen 0 2600x1600x24 +extension GLX' \
  uv run --no-sync python tools/capture_long_recording_views.py \
    --amplitude-split --bounded-dir /data/fresh-bounded \
    --cluster-ids 708 --output-dir /data/amplitude-screenshots

QT_QPA_PLATFORM=offscreen uv run --no-sync python \
  tools/compare_curation_modes.py reference /data/full-source \
  design/curation-mode-acceptance-ledger.json /data/ordinary-reference \
  --engine ordinary --chunk-size 1000000

uv run --no-sync python tools/compare_curation_modes.py compare \
  /data/ordinary-reference /data/bounded-full-output --fields group notes \
  --chunk-size 1000000 --output /data/comparison.json
```

The reference command uses full ordinary in-memory curation and needs memory
appropriate for the source. Its comparison artifact is assignments, times,
manual-label tables and provenance, not a complete sorting export. Choose the
actual active unit in a fresh bounded package when reproducing the screenshots.
Pixel quantization can move the requested threshold slightly; `capture.json`
records the exact committed value.

Local reports, scripts and logs are archived in
`/home/cyrille/GIT/IBL/phy-data/carl/aind-b1b0782a-b3ea-4f34-81d7-b3c6e36452b3/curation-mode-acceptance/`.
The initial one-off native GUI script needed a harness correction from QWidget
focus handling to QWindow activation; the corrected GUI and reference processes
both exited zero. The final capture tool also exits zero.

`make test-full` passed **960 tests** (499 core and 461 application), with one
existing benchmark skip, including lint and formatting. The 24 added tests cover
six graphical workflows and eighteen independent comparisons, including raw
nonfinite semantics, ID renumbering, metadata differences, callback cleanup,
history and output protection. The independent Sol 6.1-high audit found no
remaining concrete blocker. Documentation checks and CI are recorded at handoff.

The later [main-GUI comparison](curation-main-gui-validation.md) completed this
same ledger on unmodified main and matched every assignment and manual label.
It retains baseline Feature View rendering errors as an explicit limitation.
Carl's manual usability and scientific review remain useful, together with the
small practical variants in the matrix. Neither automated run establishes the
scientific validity of splits or of the acquisition exposure declaration.
