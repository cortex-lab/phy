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
