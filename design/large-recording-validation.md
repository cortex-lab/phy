# Long-recording implementation validation

This record summarizes the current implementation checks and their limits. The
results validate the measured code paths and fixtures described here. They do
not establish capacity for every source layout or scientific suitability for
chronic curation.

## Automated checks

On 2026-10-24, the AIND time-coordinate follow-up passed `make test-full`:
**818 passed, 1 existing benchmark skip** (496 core tests and 322 application
tests). The environment used Python 3.12.3, released phylib 2.7.1, Linux,
Xvfb, software OpenGL, and Qt's offscreen platform. The target included lint
and formatting checks. The strict documentation check and `uv build` also
passed; the wheel contains all 12 long-recording runtime modules.

The follow-up has a 96-test focused source/preparation/replay run and an
additional GUI regression for explicit elapsed clocks. Its independent
discontinuous-clock oracle verifies per-bin exposure and unit counts, exact
waveform sample-index alignment, merge assignments, save/reopen, replay,
repreparation, timestamp provenance, and unchanged source bytes. The GUI test
opens all default bounded views, checks exposure-based firing rate and window
navigation, and confirms that elapsed ticks do not enable raw traces. All of
these tests are included in the full-suite result above.

Separate architecture, correctness, and final audit reviews used Sol at high
reasoning effort. The correctness review found a legacy coordinate-name
mismatch and a generated-policy bypass of producer waveform-window membership
validation; both received regression coverage and were corrected. The final
independent audit found no remaining blocking defect in the reviewed changes.

These results validate the [explicit coordinate contract](large-recording-time-coordinates.md),
not the supplied export's acquisition truth. The complete new Code Ocean export
and its acquisition logs were unavailable. The supplied `timestamps.json` was
inspected directly; spike-array counts and filler-spike claims remain attributed
to Carl's report. No exporter repository was identified or modified. The
[exporter handoff](aind-export-handoff.md) records the upstream work and real-data
acceptance evidence still needed.

On 2026-10-13, `make test-full` completed with **756 passed and 1
skipped** (496 non-application tests and 260 application tests). Final channel
geometry and nonidentity waveform-mapping additions passed a further focused
source/preparation run of **33 tests**. Lint, format checking, package build,
and the strict documentation build also passed. The dedicated bounded GUI lifecycle file passed all three
regressions: reentrant observer rollback with the GUI attached, merge-proposal
review through undo/redo/save/reopen, and pending full metrics for threshold
children in the default bounded views.

The reference long-recording fixture is deliberately sparse: 32 spikes spread
over a 1000-hour sample clock. It checks long `int64` coordinates, tied times,
source-row mapping, waveform remapping, and loader compatibility. It is not a
billion-spike capacity test, a realistic chronic recording, or evidence of
biological identity across a recording.

## Synthetic preparation and replay runs

The reproducible benchmark is `uv run python
tools/benchmark_long_recording.py`. It writes synthetic source arrays in
batches, with no raw voltage data, and runs preparation and replay in separate
processes. Both runs below used Python 3.13.3, NumPy 2.4.3, Linux
`x86_64`/glibc 2.39, 30 kHz sampling, 1000 units, a 1000-hour sample axis,
100,000-row batches, 30-second windows, 3600-second strata, a 100,000-spike
budget, a 512 MiB output budget, seed 0, and 3600-second summary bins. The
oracle independently streamed every source and output assignment row and
matched the expected merge in both cases.

| Synthetic source spikes | Source files | Bounded package | Bounded spikes | Curated output | Prepare time / peak RSS | Replay time / peak RSS | Oracle rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250,000 | 4,032,892 B | 8,625,460 B | 2,389 | 5,038,086 B | 1.902 s / 215,773,184 B | 0.114 s / 214,847,488 B | 250,000 |
| 2,000,000 | 32,032,892 B | 9,157,120 B | 16,676 | 40,038,086 B | 1.227 s / 214,433,792 B | 0.292 s / 215,285,760 B | 2,000,000 |

Each measurement is a single run, not a throughput curve. Preparation and
replay durations exclude child-process startup; peak RSS is the process high
water mark. The source is a regular synthetic distribution with tiny templates
and no stored waveforms, raw data, irregular acquisition gaps, or real feature
loads. These figures therefore do not predict every preparation cost or GUI
interaction on a chronic dataset. The full JSON reports are retained at
`/tmp/phy-long-recording-benchmark-250k.json` and
`/tmp/phy-long-recording-benchmark-2m.json` on the validation workstation.

## Ordinary-dataset smoke comparison

A separate three-run comparison used fresh subprocesses, configs, and caches in
the same Python environment. It compared baseline `a30d5c65` with current
`f212f069e5b5b19469cfa73781dac08de093d9a1` on a 12,000-spike, two-cluster
dataset with 1,024 stored waveform rows of shape `[40, 4]` and dense template
features. Counts and output dimensions matched exactly.

| Measure (median of 3) | Baseline | Current |
| --- | ---: | ---: |
| Controller startup | 0.03879 s | 0.02822 s |
| Waveform provider | 0.00353 s | 0.00293 s |
| Feature provider | 0.00372 s | 0.00302 s |

This small case showed no regression; the apparent gains are within the range
where run-to-run noise matters. The comparison did not measure Qt rendering,
raw-voltage I/O, or large-dataset selection churn. Its report is retained at
`/tmp/phy-standard-recording-comparison.json` on the validation workstation.

## Remaining acceptance work

Real chronic-recording review by domain curators remains outstanding. It must
cover drift, rare and bursting units, artifacts, gaps, waveform coverage,
intentionally incorrect merges, withheld contiguous windows, and full-data
diagnostics where available. Software integrity checks and synthetic throughput
do not establish that the evidence is scientifically sufficient or that a
cluster represents one biological unit throughout 1000 hours.
