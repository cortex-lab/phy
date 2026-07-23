# Saving and output files

phy keeps clustering changes in memory until you save. Press Control-S or choose **File > Save**
regularly, and save before using the curated results in another program.

## Files written by the Template GUI

Saving writes:

* `spike_clusters.npy`: the current cluster ID for every spike;
* `cluster_group.tsv`: the `good`, `mua`, `noise`, or unsorted group assignments;
* `cluster_<label>.tsv`: one file for each additional cluster label;
* `cluster_info.tsv`: a convenient snapshot of the columns currently known to the Cluster View.

`spike_templates.npy` is not changed. It continues to record the original template assignment
made by the spike sorter.

Merges and splits create new cluster IDs. Old IDs are not reused, so downstream analysis should
always reload `spike_clusters.npy` and the cluster metadata after curation.

## Backups

The Template GUI overwrites its output files when saving; it does not create a versioned backup
of every save. Before beginning a new curation pass, copy at least:

```text
spike_clusters.npy
cluster_*.tsv
cluster_info.tsv
```

For a complete reproducible backup, retain the original sorter output as well. Version-control
small TSV files if useful, but avoid putting large raw binary or NumPy arrays into an ordinary Git
repository.

If you close a session with unsaved clustering changes, phy prompts you to save, discard, or
cancel closing. Undo and redo apply within the active editing history; they are not a substitute
for file backups.

## Cache and state files

The dataset's `.phy/` directory contains cached computations and local GUI state, not the primary
curated spike assignments. It can be removed and reconstructed, although the first selections
afterward will be slower.

The global `~/.phy/` directory contains plugins, configuration, screenshots, and GUI state. Do not
delete it as a general troubleshooting step unless you have backed up custom plugins and
configuration.

See [Troubleshooting](troubleshooting.md) for targeted reset commands.
