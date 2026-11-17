# Matched real-data baseline/current comparison

Single-worker passes on the same converted Carl AIND overlay: 469,725,567 source
spikes, 8,000,000-spike / 3,000,000,000-byte evidence caps, 1,800-second bins,
100,000-row batches. Replay uses the identical committed combined-SLAY-merge and
threshold-split checkpoint. This is engineering validation; acquisition exposure
truth remains unestablished.

| Operation | Baseline seconds | Current seconds | Observed ratio | Baseline RSS GiB | Current RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| prepare | 88.42 | 33.49 | 2.64x | 15.16 | 15.16 |
| replay | 54.93 | 54.99 | 1.00x | 26.62 | 26.62 |

The baseline is unmodified revision 3ee67e5 in a new isolated detached checkout.
Both processes assert their imported phy path and git revision and use the same
Python executable, NumPy and phylib versions. Exact revisions, dependency versions,
host load and individual logs are retained in comparison.json and case JSON files.
No GUI was launched by this comparison. Root GUI/media/application validation may
have run concurrently. These are single warm-cache observed ratios, not controlled
causal estimates; RSS includes mapped source/output pages.

All prepared NumPy files and both replay assignment files matched the previously
independently verified current outputs byte for byte. An independent SHA-256 and
file-stat recheck confirmed all 134 original/overlay entries unchanged. The old
checkout and runtime code were not patched. The earlier pipeline report is retained
unchanged; this is a separate comparison.
