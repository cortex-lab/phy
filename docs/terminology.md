# Terminology

We give some terminology used in phy (especially in the Template GUI).

## Probe

A **probe** is a multielectrode array used for an electrophysiological recording session.


## Shank

A **shank** is one of the different physical sets of electrode sites used in a recording.

Every shank has a unique identifier, the `shank_id`, ranging from `0` to `n_shanks-1`.

**Note**: a general principle in phy is that every id is unique and fixed. When a cluster is modified, it is assigned a new id.


## Channel

In phy, a **channel** corresponds to the digital signal recorded on a given **electrode site**.

There is a range of indirections and mappings between the physical electrode sites and the channels shown in phy.

Phy makes the following assumptions:

* The `channel_map.npy` file contains a 1D integer array, called `channel_map`, of size `(n_channels,)`.
* The raw data array has `n_channels_dat` columns (which may be different from `n_channels` if there are dead channels).
* The raw data columns are immediately and virtually swapped using the channel map, as follows: `X = raw_data[:, channel_map]`. This applies to all views that show multiple channels: trace view, waveform view, template view, etc. In other words, phy always sees the raw data as a NumPy-like array (in fact, a virtual memmapped and column-swapped array) with `n_channels` columns, which correspond to the `channel_map`-swapped columns in the binary file.
* The `channel_ids` used internally in phy are always ranging from 0 to `n_channels - 1`.
* The channel label displayed in phy of channel `channel_id` is `channel_map[channel_id]`.


## Spike

A **spike** is an action potential emitted by a given neuron, at a given time. It is recorded on a specific set of **channels**. It has a specific shape (waveform) on each of these channels. It belongs to a given **template** and **cluster**. The spike-cluster assignment is the main output of a spike sorting session.

Every spike has a unique identifier, `spike_id`, ranging from `0` to `n_spikes-1`.

Spikes themselves cannot be modified or deleted. Only the spike-cluster assignments, and cluster attributes, can be modified in phy.


## Template

A **template** is defined by a set of waveforms (**template waveforms**) on specific channels. It is obtained by a spike sorting algorithm based on template matching. The algorithm attributes a template for every spike, along with an **amplitude**. The waveform of every spike is expected to be the template waveform multiplied by the amplitude.

Every template has a unique identifier, `template_id`, ranging from `0` to `n_templates-1`.

The spike-template assignments are saved in `spike_templates.npy`. This 1D array has `n_spikes` elements, it gives the template id of every spike.

A template's **"best channels"** correspond to the channels where the template waveform has been detected (based on the waveform amplitude). The "best channel" (or peak channel) is the channel with the maximum template waveform amplitude.


## Cluster

A **cluster** is a set of spikes, supposed to have been emitted by a single neuron.

Every cluster has a unique integer, `cluster_id`, ranging from `0` to `n_clusters-1`.

The cluster id is unique: when the cluster changes (i.e. spikes are removed or added), the cluster id changes. This simplifies the implementation of phy, which uses an internal cache on disk for performance.

The spike-cluster assignments are saved in `spike_clusters.npy`. This 1D array has `n_spikes` elements, it gives the cluster id of every spike.

### Cluster vs templates

**Initially, before running phy, the spike-cluster and spike-template assignments are identical**. If `spike_clusters.npy` does not exist, it is automatically copied from `spike_templates.npy`. When modifying the spike-cluster assignments in phy, only `spike_clusters.npy` is modified, while `spike_templates.npy` remains unchanged.

As clusters are merged and split, new clusters are created, old ones are deleted. Therefore, whereas the template ids and clusters ids match initially, they no longer do as soon as the user performs manual clustering actions.


## Amplitude

There are several slightly different definitions for the amplitude:

* Per template:
    * **Template amplitude**: for every template, the maximum amplitude of the template waveforms across all channels.
* Per spike:
    * **Spike raw amplitude**: for every spike, the maximum amplitude of the raw waveforms across all channels (extracted from the raw data file).
    * **Spike template amplitude**: for every spike, the value found in `amplitudes.npy`, saved by the spike sorting algorithm.
* Per cluster:
    * **Mean spike template amplitude**: for every cluster, the average of the spike template amplitudes.
    * **Mean spike raw amplitude**: for every cluster, the average of the spike raw amplitudes.


## Event

(Upcoming feature). Time behavioral events, like stimulus onsets, that will be supported in a future version of phy.
