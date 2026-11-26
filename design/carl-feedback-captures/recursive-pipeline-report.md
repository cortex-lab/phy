# Full source recursive restoration validation

Engineering validation of a converted fixture only: original acquisition clock, exposure and producer export have not been verified. Timings are observations, not benchmarks; other GUI and test workloads run concurrently.

Code: `03377575cadc4cfd010d1977b869375e5078d515`. Complete: **True**. Input SHA-256/stat inventories unchanged: **True**.

The independent oracle checked **469,725,567 rows in each of 2 outputs**, using original labels and raw finite amplitude comparisons.

| Phase | Seconds |
|---|---:|
| immutable-inventory-before | 20.42 |
| copy-prepared-input | 0.15 |
| public-gui-supervisor-curation-save-reopen | 46.97 |
| full-source-replay-1 | 93.92 |
| full-source-replay-4 | 80.54 |
| independent-original-row-oracle | 14.51 |
| replay-canonical-bytes-counts-labels-reviews-lineage | 84.68 |
| reprepare-one-replayed-full-source | 25.90 |
| reprepared-contract-verification | 0.56 |

See `result.json`, `scenario.json`, `curation-state.json`, `independent-oracle.json`, and input inventories for exact provenance and checks.

All **45 metadata fields (31,545 unit cells)** match exactly in both full outputs and the reprepared package, including JSON types. Cumulative metadata provenance also matches. Both assignment payload SHA-256 hashes match; timestamps, subset manifest and producer propositions retain their bytes. See `final-checks.json`.

The scientific pipeline is attributed to the launch commit above. During validation, a subsequent UI-only SLAY containment proof was added in `live_summaries.py` and its caller in `large_recording/supervisor.py`. AST comparison confirms the existing membership-expression, assignment, journal and replay code stayed unchanged. Before/after file hashes are recorded in `final-checks.json`; the followup has separate focused UI regressions.

Copy method: 73 conservative full-copy fallbacks, 0 explicit reflinks; all copies use distinct inodes.
