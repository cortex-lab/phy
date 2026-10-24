# Curating long recordings

Phy's long-recording workflow prepares a bounded evidence dataset for interactive
review, then replays supported curation actions against the complete local
sorting. Preparation reads source arrays in batches; it does not load the full
sorting into the GUI. The source dataset is retained unchanged, and replay
writes a separate output dataset.

This workflow supports a single monotonic event axis. Existing sources without
an explicit time-base declaration use `acquisition_samples`, the legacy
fixed-rate sample-coordinate contract. Sources with `time_base` may instead
declare `elapsed_ticks`: nonnegative integer elapsed-time coordinates with a
declared `ticks_per_second`. In either case, `num_samples` is the exclusive
upper bound of the canonical coordinate axis, and `valid_intervals` are
half-open `[start, end)` intervals on that same axis. For elapsed ticks,
`num_samples` is a tick bound, not a physical frame count. Explicit elapsed
sources must provide authoritative valid intervals; Phy cannot infer
acquisition gaps or filler from periods with no spikes. Multi-segment recordings
are not supported.

The `time_base` declaration is a dictionary such as
`{'kind': 'elapsed_ticks', 'ticks_per_second': 30000.0}` in `params.py` (or
`recording.time_base` in the waveform manifest). For this workflow,
`ticks_per_second`, `sample_rate`, and the physical waveform sample rate must
be equal. Unequal event and waveform rates need separate waveform-axis support.
A source containing `timestamps.json` must declare its coordinate contract
explicitly; Phy does not parse that file to guess a conversion. Elapsed ticks
cannot index raw voltage, so an elapsed source with a real `dat_path` is not
supported. Stored waveforms can be used without raw data.

For explicit elapsed exports, the waveform manifest must declare
`recording.sample_index_coordinates` and
`recording.window_bounds_coordinates` as `elapsed_ticks` when those sidecars are
present. The waveform-aligned `sample_index[w]` must equal
`spike_times[subset_spikes[w]]`; it is neither a source row ID nor a raw frame
offset. The manifest's `sample_index_coordinates` and
`window_bounds_coordinates` must each equal the declared `time_base.kind`.
Producer window bounds describe canonical elapsed coordinates and must not
cross unsupported gaps. The 512 MiB output cap applies to package bytes, not
process RSS; waveform source, retained, and omitted counts are reported in the
CLI and manifest.

The screenshots below show the actual bounded GUI views using a deterministic
synthetic eight-hour sorting. The overview plots full-duration rates and
available source means; gray marks the synthetic acquisition gap and cyan marks
the retained evidence windows. The evidence view places windowed amplitudes at
top left, sampled ISI at top right, sampled correlograms at middle right, and
features below. They illustrate the interface and its sampling context; they
are not biological data or scientific examples.

![Full-duration Unit Stability with a synthetic acquisition gap](images/long-recording-overview.png)

![Windowed source amplitudes, features, sampled ISI and correlogram](images/long-recording-evidence.png)

Recreate these images from source with Qt's X11 platform and software OpenGL
(the script writes a temporary synthetic source and bounded package):

```bash
QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -s '-screen 0 2400x1600x24 +extension GLX' \
  uv run python tools/capture_long_recording_views.py
```

## Prepare bounded evidence

Run `prepare-large` with the source `params.py` and a new output directory:

```bash
phy prepare-large /data/recording/params.py /data/recording-bounded \
  --duration 3600000 \
  --valid-intervals /data/recording/valid-intervals.npy \
  --max-spikes 2000000 \
  --max-bytes 536870912 \
  --window-seconds 30 \
  --stratum-seconds 3600 \
  --seed 0 \
  --bin-seconds 3600 \
  --batch-size 100000
```

`--duration` is in seconds and must agree with declared acquisition metadata.
The `.npy` valid-interval file contains an integer array of shape `[W, 2]` with
ordered, nonoverlapping coordinate pairs. Without it, legacy acquisition-sample
sources treat the full declared extent as valid exposure. Explicit elapsed
sources require producer-declared `valid_intervals`; there is no full-extent
default for them.

`--max-bytes` limits the bounded output package size, including its summaries and
window plan. It is not a process-memory or peak-RSS limit. The source unit IDs
must fit nonnegative `int32`; sample coordinates and full-source spike row IDs
use `int64`.

Preparation uses `--window-policy auto` by default. When validated producer
windows are available, it selects among those whole windows and preserves their
acquisition boundaries. Producer-window mode requires valid producer windows;
it does not generate rescue windows to compensate for missing or malformed
producer evidence. Without producer windows, the original shared-window planner
uses the configured window length, strata, and seed. Inspect the saved window
plan in the output package when documenting or repeating an analysis.
Windows that exceed the budget are omitted, so a high-density interval can
reduce retained evidence. Preparation does not shorten an over-budget window or
silently thin spikes inside it. The generated planner supplements its survey of
shared windows with deterministic rescue evidence for units missed by the
survey. This improves the chance of seeing rare units, but makes the evidence
selection unequal: rescue windows must not be treated as a probability sample.
The explicit `producer` policy uses only producer windows. The `auto` policy
can supplement producer windows with generated rescue evidence for units still
missing from the selected evidence.

Overlapping producer windows are allowed; selection uses their union while
retaining their acquisition boundaries. Preparation records exact
full-recording spike counts and rates using valid exposure. If source scalar
amplitudes or depths are available, it
also records per-bin finite-value means and availability counts. It does not
invent missing measurements or convert source values to physical units. Stored
waveforms are included only where available in the source; raw traces are not
required when stored waveform data are sufficient for the views being used.
The default spike and byte budgets remain 2,000,000 spikes and 512 MiB. Output
byte budgets describe package bytes, not resident memory. Waveform availability
can exceed the bounded output budget: preparation reports waveform counts
retained, omitted, and available in the source on the CLI, and stores the counts
in the manifest so truncation is visible.
Producer-supplied per-unit metrics and annotations remain inherited metadata;
Phy does not recalculate those values for merged or split units. Use the
full-count, rate, amplitude-mean, and depth-mean summaries produced by this
workflow for the measurements they explicitly cover.

## Review and curate

Open the bounded output with the normal GUI command:

```bash
phy template-gui /data/recording-bounded/params.py
```

The bounded manifest explicitly selects this mode. Bounded mode has its own
configuration namespace and default view layout; it does not replace the normal
Phy layout or restore ordinary saved dock geometry. The GUI keeps full-recording
counts and rates distinct from counts in selected evidence rows. Unit Stability
shows time-binned counts/rates, available source scalar-amplitude means,
evidence windows, and acquisition gaps. Missing measurements and units without
evidence remain visible as unavailable.

Evidence Amplitude shows raw source scalar amplitudes from the currently
selected evidence window. The Feature view also uses window-relative seconds.
The amplitude plot labels this axis as **Seconds from window start** and shows
the window's recording-time range above the plot.
Both views label the current window and reset their time origin when it changes.
To inspect another retained window, choose **View > Inspect evidence window**
and enter its zero-based index. Unit Stability keeps the full-recording time
context visible while you inspect individual windows.

The Sampled ISI view excludes intervals crossing retained-window boundaries.
The Sampled Correlogram view shows raw pair counts from retained windows; it
does not apply a full-recording rate baseline. Both views are explicitly labeled
as sampled evidence. Its central bin counts unordered within-unit pairs and
combines both temporal orientations for cross-unit pairs whose absolute delay
is less than one bin, so tied sample times do not depend on source row order.
For exact pair counts, each unit may have at most 46,340 spikes in an evidence
window; the view reports an error for a larger per-unit window. Prepare shorter
windows if this limit is reached.

Bounded curation supports whole-unit merges, cluster labels, and a source
amplitude threshold action. Use the GUI action **Split by source amplitude**
with one unit selected and a threshold in the source amplitude units. The action
uses the source `amplitudes.npy` scalar for every spike during replay; nonfinite
values stay with the remainder. Child units show full-data metrics as pending
until the replayed sorting is re-prepared, because their complete counts and
trends have not yet been calculated. Merge and label actions are journaled with
the bounded assignments and metadata in an atomic checkpoint generation.
Reopening loads the last complete generation. A reopened session starts a new
undo baseline. Each save checks that the checkpoint has not changed since this
session opened. If another session has saved first, the stale save fails; reopen
the bounded dataset before continuing to avoid overwriting that generation.

Lasso, feature, and waveform splits remain unavailable in bounded mode because
the selected rows do not define how unseen source spikes should be assigned.
Standard, unbounded datasets retain their existing curation behavior.

## Unit Stability on ordinary recordings

Unit Stability can also be opened as an optional view for an ordinary dataset.
It evaluates only the selected unit and bins its loaded spikes over the recording
duration. Since ordinary datasets do not supply valid acquisition intervals to
this view, its rate assumes continuous acquisition; the plot labels that
assumption. The ordinary default view layout remains unchanged. Use the bounded
manifest workflow above when known acquisition gaps must be represented
explicitly.

## Apply to the complete sorting

After saving the bounded session, replay its journal against the exact source
assignment revision used during preparation:

```bash
phy apply-curation /data/recording/params.py /data/recording-bounded \
  /data/recording-curated --batch-size 100000
```

Replay checks the source identity and parent assignment revision, streams the
complete assignment array in batches, and writes a new full sorting. It rejects
a journal prepared from a different source revision. Keep the original source
and bounded package; they establish the lineage of the curated output. To review
the curated full sorting with bounded evidence again, run `prepare-large` on
the curated output to produce a fresh package and revision-specific summaries.
This replay and re-preparation step also resolves full-data counts and trends
for threshold-split children.

## Interpreting evidence

Full counts and exposure-based rates describe the complete sorting. Window plots,
waveforms, feature scatterplots, and other bounded measurements describe only
the retained evidence. Survey and rescue selection can overrepresent some
periods or units, and waveform availability can vary across units and time.
These views do not establish full-recording quality, contamination, refractory
violations, or continuous biological identity. Use them to guide review, not as
scientific claims about unseen spikes. Long-duration coordinates preserve the
source sample clock; silence alone is never evidence that the acquisition was
inactive.

For reproducible synthetic capacity measurements, run
`uv run python tools/benchmark_long_recording.py`. The benchmark reports
preparation and replay time and peak memory in separate processes and checks
replay against a streamed oracle. It is capped at two million synthetic spikes;
its results are not a substitute for measurements on real chronic datasets.
