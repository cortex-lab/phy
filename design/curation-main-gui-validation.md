# Main GUI versus bounded PR curation: real-data comparison

On 26 October 2026, the actual ordinary GUI on current `master`
(`ad64e9138a6556f4785a359c3ee1258864d60f1d`, verified against GitHub) completed
the same action ledger as the bounded PR GUI. Its complete saved output matches
bounded full-source replay on **all 469,725,567 spikes** and requested manual
labels. This extends the earlier ordinary-engine-only reference with an actual
main-branch TemplateModel, controller, graphical split and GUI save/reopen.

The result is **curation completed with baseline Feature View errors**. It is
not an error-free all-view GUI acceptance claim. The renderer failure and other
limits are retained below.

## Inputs and isolation

- Source: Carl's earlier converted engineering export, computation
  `b1b0782a-b3ea-4f34-81d7-b3c6e36452b3`, 705 original units, 112.859 elapsed
  clock hours. This is not Carl's newer 500-hour dataset, and its inferred
  exposure remains unverified.
- Main code: a clean detached worktree at `ad64e913`; the script asserts the
  imported `phy` module comes from that worktree. No main code was changed.
- PR runtime: `06637b24`, unchanged by the subsequent tool/documentation commits
  through reviewed head `c79666c6`. The previously completed native bounded GUI
  run and full replay were reused, with their pinned inputs and ledger.
- Main working data: a separate 15,381,601,942-byte deep copy with symlinks
  dereferenced, never hard-linked. Its `params.py` has one final `dat_path = None`
  override so ordinary phylib treats the exporter's string sentinel as absent raw
  data. No event times, source rows, amplitudes or exposure declarations were
  converted for this run. Both parameter-file hashes are recorded locally.
- Environment: Python 3.12.3, phylib 2.7.1, native X11 Qt under Xvfb and software
  OpenGL. Each full ordinary GUI process ran under a 96 GiB virtual-memory limit.

The [shared ledger](curation-mode-acceptance-ledger.json) specifies merge
40/44/48, undo/redo, source-amplitude split, undo/redo, merge the low child with
24, explicit GOOD/MUA and notes, a label-history branch, save/reopen and a no-edit
save. The ordinary GUI used **template** amplitudes, matching `amplitudes.npy`.

## Native graphical split and save/reopen

The script ran the real ordinary controller and Supervisor. A real Qt
Alt-right-drag placed the preview; a real `K` key event triggered the standard
split action and its all-spike provider. The resulting two children were checked
against the full parent membership and source predicate, rather than inferred
from child counts alone.

The viewport was panned so the shared threshold landed on an integer pixel.
Requested and actual thresholds were exactly **-424.0037077454673**. An
independent chunked scan of raw source arrays confirmed all **1,156,852 parent
spikes** have finite amplitudes, with none equal to the threshold. Thus ordinary
display sanitization of nonfinite values does not affect this comparison.

After the ledger's final save, the process closed the GUI/controller and exited
zero. A fresh process loaded the saved full sorting through the main GUI,
verified the final groups/notes for all three reviewed units, selected them,
saved without edits and exited zero. Using a separate process establishes real
reload and avoids retaining an earlier model through global callbacks.

Observed single warm-cache runs:

| Run | Wall time in harness | Peak RSS |
| --- | --- | --- |
| Main GUI ledger and saves | 116.1 s | 33.5 GiB |
| Fresh-process GUI reopen and no-edit save | 44.5 s | 26.5 GiB |

These are integration observations, not controlled performance benchmarks.

The caller requested Amplitude and Waveform views, but ordinary saved/default
state restored **11 view types**, including Feature View, Firing Rate, ISI,
Correlogram, Probe, Cluster Scatter and the tables. The original run JSON's
`views` field lists requested views only. The committed summary records the
actual restored inventory and the rendering errors instead of treating that
field as the observed layout.

## Exact complete-output comparison and integrity

`tools/compare_curation_modes.py compare` compared the saved main full sorting
after reopen against the existing complete bounded replay. It passed:

- **469,725,567** rows, with exact row-aligned integer spike times.
- **Zero partition mismatches**, using a two-way raw unit-ID mapping over every
  row, spanning 703 final units.
- **Zero group/notes differences** and no unmapped manual-label IDs.

Ordinary main saved int32 assignments; bounded replay retained int64. Assignment
dtype and numeric IDs need not agree for partition equality. The comparison
checks exact integer memberships, with input hashes recorded in the
[report summary](curation-main-gui-result.json).

An independent verifier checked all **67 original download files** against the
archived SHA-256 inventory. All **19 immutable working-copy files** (every NPY
except assignments, plus timestamps and suggestions) match the converted source.
The source parameters still match the original hash recorded during preparation.

An optional working-copy hash helper was started concurrently with the first
GUI run and lost a race with the intentional GUI save of `spike_clusters.npy`.
Its failed log is retained; it is not claimed as a successful input preflight.
The source copy operation, unchanged original hashes, final immutable-file check
and complete-output comparison provide the stated evidence. When reproducing,
run the copy hash helper before launching the GUI.

## Baseline Feature View failure

Main emitted **7 Feature View errors during curation and 5 during reopen**:
background feature sampling sometimes requests no IDs in the stored waveform
reservoir. Unmodified phylib 2.7.1 `TemplateModel.get_features()` then invokes PCA
on zero waveforms, causing `ZeroDivisionError` in `_compute_pcs()`.

A minimal independent reproduction calls the installed `get_features()` with
requested row 2 and stored row 1; their empty intersection raises the same
exception. This is a baseline feature-rendering failure, not an assignment
mutation. No exception was discarded when classifying this acceptance result.

Bounded `_get_feature_view_spike_ids()` intersects selected and background IDs
with stored waveform rows before projection when dense features are unavailable.
Its inherited feature getter returns an empty bunch for an empty request, so
this particular phylib empty-PCA path differs from ordinary main. No PR #1428
change is needed to establish the curation equivalence above. An upstream phylib
guard and focused regression would be a separate follow-up; changing the
ordinary baseline during this comparison would hide the observed behavior.

## Artifacts and remaining acceptance

Actual main captures are in `docs/images/amplitude-main-gui/`: the native
threshold preview and the reopened full sorting. Empty feature panels in these
captures reflect the recorded baseline issue. They are not a controlled layout
comparison with the separately captured bounded views.

Reports, exact scripts, screenshots and logs are archived under
`/home/cyrille/GIT/IBL/phy-data/carl/aind-b1b0782a-b3ea-4f34-81d7-b3c6e36452b3/main-versus-pr-gui-acceptance/`.
The separate high-effort audit independently confirmed raw parent finiteness,
the exact comparator result, both process exits, restored inventory and the
rendering-error scope. No additional product-code regression was identified.

This closes the automated main-GUI versus bounded-GUI comparison for this pinned
dataset and ledger. Carl's manual usability and scientific assessment remain
valuable, particularly on his 500-hour recording. The practical variants in
`curation-user-acceptance.md` remain proposed user tests, not claims that every
variant was executed. Acceptance of Brandon's native exporter still needs that
export and authoritative acquisition exposure.
