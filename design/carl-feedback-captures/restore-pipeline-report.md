# Full source restoration validation

Engineering validation of a converted fixture only: original acquisition clock, exposure and producer export have not been verified. Timings are observations, not benchmarks; other GUI and test workloads run concurrently.

Code: `e13906ac68510d24df71205d96ca6d5994724c56`. Complete: **True**. Input SHA-256/stat inventories unchanged: **True**.

The independent oracle checked **469,725,567 rows in each of 2 outputs**, using original labels and raw finite amplitude comparisons.

| Phase | Seconds |
|---|---:|
| immutable-inventory-before | 11.97 |
| copy-prepared-input | 0.12 |
| public-gui-supervisor-curation-save-reopen | 37.97 |
| full-source-replay-1 | 86.76 |
| full-source-replay-4 | 85.29 |
| independent-original-row-oracle | 12.81 |
| replay-canonical-bytes-counts-labels-reviews-lineage | 89.46 |
| reprepare-one-replayed-full-source | 25.42 |
| reprepared-contract-verification | 0.56 |

See `result.json`, `scenario.json`, `curation-state.json`, `independent-oracle.json`, and input inventories for exact provenance and checks.

All **45 metadata fields (31,590 unit cells)** match exactly in both full outputs and the reprepared package, including JSON types. Cumulative metadata provenance also matches. Both assignment payload SHA-256 hashes match; timestamps, subset manifest and producer propositions retain their bytes. See `final-checks.json`.

Copy method:73 conservative full-copy fallbacks,0 explicit reflinks; all copies use distinct inodes.
