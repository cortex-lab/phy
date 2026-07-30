# Cluster similarity

The Similarity View ranks clusters that may be useful to compare with the
cluster selected in the Cluster View. It is a navigation aid: a high score is
not, by itself, evidence that two clusters should be merged. Always inspect the
waveforms, correlograms, amplitudes, firing stability, and refractory-period
violations before merging.

## Default in the Template GUI

The Template GUI uses **template similarity** by default. The sorter normally
writes an `n_templates × n_templates` matrix to
`similar_templates.npy`; larger values mean that the sorter considered two
templates more similar. The numerical scale and the way the matrix was computed
depend on the sorter, so phy does not define a universal score threshold for a
merge.

For a cluster containing one original template, phy reads the corresponding
row of this matrix. Manual merges and splits may make a cluster contain spikes
from several original templates. In that case, the score between clusters \(A\)
and \(B\) is:

```text
max(similar_templates[i, j] for i in templates(A) for j in templates(B))
```

In other words, one strongly matching pair of constituent templates determines
the cluster-to-cluster score. The template membership is computed from the live
spike clustering, so it follows merges and splits made during the current
session.

The default function sorts by decreasing score and returns at most 100
clusters. Sorting and filtering in the Similarity View can further change which
rows are visible.

If `similar_templates.npy` is absent, phylib supplies an all-zero matrix. The
Similarity View can still open, but the tied zero scores provide no meaningful
ranking. Generate the file with the sorter or install a custom metric if you
need useful similarity navigation.

## Other controllers

The base controller's fallback is **peak-channel similarity**. It returns every
cluster whose best-channel set includes the reference cluster's peak channel,
with a score of `1.0`. The Template GUI overrides this fallback with template
similarity.

## Changing the metric

A plugin can add a function to `controller.similarity_functions` and select it
through `controller.similarity` before the supervisor is created. The function
takes one cluster ID and returns `(cluster_id, score)` pairs in decreasing score
order.

See [Writing a custom cluster similarity metric](plugins.md#writing-a-custom-cluster-similarity-metrics)
for a complete mean-waveform example. Keep custom metric code in a plugin
rather than editing phy: it remains separate from the curation workflow and is
easier to reuse across datasets.

When designing a metric:

- define whether larger values always mean “more similar”;
- document its scale and whether a threshold has a scientific meaning;
- use the live clustering if the result should follow manual merges and splits;
- return a bounded candidate list if computing or displaying every pair is
  expensive;
- invalidate any plugin cache when the metric or its inputs change.
