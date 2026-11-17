# Carl feedback: rendered real-data validation

These are actual Qt/OpenGL screenshots and uncut virtual-display recordings,
not generated UI mockups. The ordinary run used Carl's earlier Sway sorting
(4,143,311 spikes, 614 original units, 181,947 stored waveforms). The bounded run
used a fresh package from the converted AIND engineering fixture (469,725,567
source spikes, 705 original units, 93 suggestions), retaining 1,486,361 evidence
spikes and 27,865 stored waveforms under the default 2M-spike / 512MiB caps.
The source fixtures were copied before curation; input-file hashes were
unchanged after each complete run. No windows or recordings reached the desktop.

The converted AIND clock/exposure assumptions remain engineering assumptions;
these captures do not validate acquisition timing, rates or merge quality.
See [the source investigation](../aind-real-data-validation.md).

## Recordings and checkpoints

Scaled 4fps preview; the complete recordings below retain the original frames:

![Bounded workflow preview](bounded-preview.gif)

- [Ordinary recording, 51 seconds](ordinary-walkthrough.mp4) · [captions](ordinary-walkthrough.srt)
- [Bounded recording, 54 seconds](bounded-walkthrough.mp4) · [captions](bounded-walkthrough.srt)

The MP4s contain optional English soft subtitles. Video packets are identical to
the original captures; only a subtitle track was added. The display was
1920×1200 at 12fps, Xvfb with software Mesa OpenGL. Renderer inventories and
[packet/dimension checks](artifact-validation.json) accompany the videos.
The recorder limits BLAS and software-renderer threads within its own process.
These videos are visual/state evidence, not interactive latency benchmarks.

Each run exercised ten checkpoints: initial comparison, actual correlogram
right-click transfer, one suggestion, Ctrl-click addition/removal, explicit G
merge, context-menu Show parents, visible one-click return, undo, redo/save,
and reopen. Qt mouse/key events drive native widgets; this does not establish
physical input routing. Screenshot settlement pumps events and resizes the GUI,
so this does not prove spontaneous repaint correctness.

The bounded run explicitly chooses evidence window 7 to show the selected units'
stored waveforms/features. Reopening initially selects window 0; the final
recording explicitly uses the **Inspect evidence window** dialog to select 7
again. This is a user action, not automatic persistence of the inspected window.
The final amplitude plot consequently shows that window rather than the initial
all-windows overview.

| Checkpoint | Ordinary | Bounded |
| --- | --- | --- |
| Ctrl-click combined suggestions | [Screenshot](ordinary-04-control-click-union.png) | [Screenshot](bounded-04-control-click-union.png) |
| Exact immediate parents | [Screenshot](ordinary-06-show-parents.png) | [Screenshot](bounded-06-show-parents.png) |
| Saved state reopened | [Screenshot](ordinary-10-reopened.png) | [Screenshot](bounded-10-reopened.png) |
| Correlogram row transferred | In video | [Screenshot](bounded-02-right-click-transfer.png) |
| Undo restores both suggestions | In video | [Screenshot](bounded-08-undo.png) |

The ordinary union was `[9,15,10,13] → 614`; the bounded union was
`[40,44,48,49,50,51,53] → 705`. Selection alone changed neither assignments nor
reviews. Both selected reviews were accepted-modified by the explicit combined
merge. Parent inspection changed no assignments, metadata, dirty state or history.
Undo restored assignments, review records and the exact proposition workspace;
redo and save/reopen preserved the complete expected assignment vector and both
review records. All returned parent groups were exact after reopening.

The images revealed two real UI defects that state-only tests missed: the generic
GUI toolbar started hidden, and a wrapped label could consume excessive header
height. Commits `f2102295` and `4a2345b2` fix visible/reachable return controls,
a compact untruncated legend, and a 2×2 scientific layout with correlogram priority.
Regression coverage checks visibility, actual click geometry and wrapping for
2, 7 and 16 parents at two window widths.

The ordinary capture is at `40c5d2bd`; the final bounded capture is at
`ce019119`. Only the recorder's explicit reopen-window step differs between
those revisions; production runtime files are identical. Sanitized checkpoint
reports: [ordinary](ordinary-capture.json), [bounded](bounded-capture.json).
All twenty original PNGs and complete local logs remain in the separate validation
workspace; selected checkpoints are published here with both complete recordings.

## Full-source replay and performance

[The pipeline report](pipeline-report.md) covers independently verified assignments
for every one of the 469,725,567 source rows in **both** serial and four-worker
replays, following a combined SLAY merge and source-amplitude split. It also
covers byte-identical prepared/store/index payloads, all 140,240 waveform rows,
save/reopen, repreparation, exact counts, finite-amplitude counts and sums, labels,
review records, independent source inventories and output inode separation.

[The matched baseline comparison](baseline-comparison.md) uses identical dependencies,
parameters and checkpoint with unmodified `3ee67e5` versus `40c5d2bd`:

| Serial operation | Old PR head | Current | Observed result |
| --- | ---: | ---: | --- |
| Preparation | 88.42 s | 33.49 s | 2.64× faster |
| Full replay | 54.93 s | 54.99 s | Essentially unchanged |

The separate one-versus-four-worker run found preparation 31.75 → 26.92s,
row snapshots 61.97 → 59.18s, indexes 67.46 → 102.65s, and replay 53.86 → 52.51s.
These are single warm-cache workstation observations with possible concurrent
validation work, not controlled causal estimates or predictions for Carl's newer
5.24-billion-spike recording. The default stays at one worker; more workers can
slow index construction on this layout.

## Reproduction

From a synced checkout, using the existing real fixture paths on your machine:

```bash
uv run --no-sync python tools/capture_carl_feedback.py \
  --mode ordinary --dataset SWAY_SOURCE --output NEW_ORDINARY_OUTPUT
uv run --no-sync python tools/capture_carl_feedback.py \
  --mode bounded --dataset FRESH_BOUNDED_PACKAGE --output NEW_BOUNDED_OUTPUT
```

The recorder creates its own Xvfb server, configuration and copied dataset.
It needs Xvfb, GLX/Mesa, glxinfo and FFmpeg. Its bounded example expects the
specific AIND proposition groups named above. New output directories are required.
It mutates only its newly created working copy and checks source hashes afterward.

A standalone verifier independently rechecked both full replay assignment arrays
without importing Phy's projection or journal implementation:

```bash
uv run --no-sync python tools/verify_feedback_replay.py \
  --source CONVERTED_SOURCE --outputs REPLAY_ONE REPLAY_FOUR \
  --parents 40 44 48 49 50 51 53 --children 706 707 \
  --threshold -378.4182434082031
```

It verified all rows, 700 active units, and child counts 1,691,273 / 1,597,610.

## Test-process limitation

The updated application target passed **791 tests, 6 skipped**, with exit status
zero; both final real GUI recording processes also exited normally. The core
partition passed **682 assertions, 13 skipped**, then the previously documented
intermittent native interpreter-shutdown fault recurred (exit139, no Python frame).
Thus the latest `make test-full` invocation did **not** complete cleanly.
CI is excluded from this assessment at the maintainer's request; these results
are local Linux validation. Prior
standard and debugger core runs have exited normally; the cause remains
unlocalized. The current evidence supports further beta testing, not a claim
that native shutdown or final release readiness is established.
