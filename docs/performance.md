# Spike sampling and performance

Several views use a subset of spikes to stay responsive on long recordings.
Increasing a limit gives a denser display or a more complete calculation, at
the cost of CPU time, memory, raw-data reads, and sometimes GPU work. These
limits do not discard spikes from the dataset or change the saved clustering.

## Current defaults

| Controller setting | Default | Applied to |
| --- | ---: | --- |
| `n_spikes_waveforms` | 100 | Per-cluster ceiling in the Waveform View |
| `n_spikes_waveforms_total` | `None` | Optional shared ceiling for displayed Waveform View clusters |
| `batch_size_waveforms` | 10 | Persisted compatibility setting; the current selector does not read it |
| `n_spikes_features` | 2,500 | Each displayed cluster in the Feature View |
| `n_spikes_features_background` | 2,500 | Background points across the recording |
| `n_spikes_amplitudes` | 10,000 | Per-cluster ceiling in the Amplitude View |
| `n_spikes_amplitudes_total` | `None` | Optional shared ceiling for selected Amplitude View clusters |
| `n_spikes_correlograms` | 100,000 | Per-cluster ceiling for ACG/CCG computation |
| `n_spikes_correlograms_total` | `None` | Optional shared ceiling for one ACG/CCG computation |

The default `None` total settings give every cluster the corresponding
per-cluster budget, so selecting more clusters does not reduce the sampling
accuracy of any one cluster. For example, eight large clusters in the
Waveform View may each display 100 waveforms. `None` disables only the
secondary shared cap; it does not disable the per-cluster limit or load all
spikes.

Set a total setting to an integer to opt into a fixed shared budget when
predictable runtime or memory use matters more. For example,
`n_spikes_waveforms_total = 400` divides 400 waveforms across the displayed
clusters, subject to the 100-per-cluster ceiling. Small clusters keep all
their available spikes and leave their unused shares for larger clusters.

You can change these values without a plugin: open the Waveform, Amplitude, or
Correlogram view menu and choose **View settings**. **Use per-cluster budget**
controls the corresponding `n_spikes_*` value, and **Use shared total budget**
controls `n_spikes_*_total`. Disabling either checkbox stores `None` for that
limit. The Amplitude dialog also exposes
`n_spikes_amplitudes_background`. Changes apply immediately and are saved in
the global Template GUI state when phy closes normally.

Other common cluster display limits are eight clusters for the Waveform,
Feature, Amplitude, and scatter views, and 20 for histogram and Probe views.
These limits control what a view plots; they do not change the table selection.

Waveform, Amplitude, and Correlogram views also defer selection plotting while
their dock is hidden or in an inactive tab. Their public cluster selection is
updated immediately, but the canvas is redrawn only when it becomes visible.
Only the latest pending selection is retained.

## How spikes are chosen

The standard selector samples up to the configured number independently from
each requested cluster, without replacement, and returns the chosen spikes in
time order.

`batch_size_waveforms` remains in controller and GUI state for compatibility,
but the current Waveform View does not use it. Raw-data chunk selection is
controlled separately by the controller's `n_chunks_kept` implementation.

- Waveforms are restricted to exported subset waveforms when those are
  available. Otherwise phy samples from a limited set of raw-data chunks to
  avoid expensive random reads, especially with compressed data.
- Feature and amplitude points are normally random subsets of each cluster.
  Raw amplitudes use the same exported-waveform or chunk-aware restrictions as
  raw waveforms.
- Background feature points are regularly spaced through the full spike list,
  independently of cluster identity.
- Correlograms use a random subset bounded by
  `n_spikes_correlograms` per cluster. If
  `n_spikes_correlograms_total` is an integer, it also bounds the entire
  view.

The Firing Rate View is different: it uses every spike in each displayed
cluster, with no spike-sampling budget. Its bin count and visible time range
control the histogram. The time range is saved in the dataset's `.phy/state.json`
and is not reused for fresh datasets.

The Correlogram, Firing Rate, and ISI view menus expose their bin and range
parameters together under **View settings**. These display parameters are
dataset-local; changing them does not affect the controller spike budgets.

Some actions explicitly request `load_all=True` and bypass a display limit.
That does not turn the corresponding view setting into a global analysis
setting; check the action or plugin doing the calculation.

### Pre-extract waveforms before opening phy

If on-demand waveform reads are the bottleneck, pre-generate the individual
spike data used by the Waveform View by creating phylib's reusable waveform
subset first:

```bash
phy extract-waveforms params.py 500 --nc 16
```

Phy automatically uses the three `_phy_spikes_subset.*.npy` files produced
beside `params.py`. This pre-extracts up to 500 spikes per original template on
the relevant channels, sampled from 20 representative raw-data chunks. The
Waveform View then draws its current per-cluster sample from this saved pool
instead of reading those waveforms from the raw binary.

Despite the command's historical description, this feature does not
materialize every spike waveform in the recording, and there is currently no
supported “extract all” CLI switch. Increasing `500` increases the ceiling but
does not remove the representative-chunk sampling. See
[Pre-extract waveform subsets](cli.md#pre-extract-waveform-subsets) for output
files, overwrite behavior, sizing guidance, and limitations.

## Change the waveform count for this session

In the Waveform View, type the snippet:

```text
:wn 500
```

and press Enter. This sets `controller.n_spikes_waveforms` to 500 and replots
that view. The value is saved in the GUI state when phy closes normally.

## Change several limits with a plugin

Put the following in `~/.phy/plugins/n_spikes_views.py`. Applying the values in
`on_gui_ready` is important: this event runs after phy has restored packaged,
global, and dataset-local GUI state.

```python
from phy import IPlugin, connect

class ExampleNspikesViewsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            controller.n_spikes_waveforms = 500
            # Optional shared totals: remove these three lines, or use None,
            # to retain the same per-cluster accuracy as selections grow.
            controller.n_spikes_waveforms_total = 2000
            controller.n_spikes_features = 5000
            controller.n_spikes_features_background = 5000
            controller.n_spikes_amplitudes = 20000
            controller.n_spikes_amplitudes_total = 80000
            controller.n_spikes_correlograms = 250000
            controller.n_spikes_correlograms_total = 1000000
```

Enable it in `~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['ExampleNspikesViewsPlugin']
```

Correlogram cache keys include both correlogram spike limits, so changing
either limit does not require `--clear-cache`. Previously computed entries for
other limits remain available until normal cache-size maintenance removes
them.

The post-state `gui_ready` hook reliably wins over saved controller values, so
`--clear-state` is not required. If you do use `--clear-state`, be aware that it
removes both `~/.phy/TemplateGUI/state.json` and this dataset's local state,
resetting saved layouts and view options along with controller values.

The repository also contains a
[spike-count plugin example](https://github.com/cortex-lab/phy/blob/master/plugins/n_spikes_views.py).
Whichever plugin you use, keep only the settings you intend to own.

## State and cache precedence

At startup, effective GUI values are assembled in this order:

```text
controller defaults
→ early plugin assignments
→ existing global GUI state (or packaged state on the first launch)
→ dataset-local state
→ on_gui_ready plugin assignments
```

Later sources override earlier ones when they contain the same key. Global state is stored in
`~/.phy/TemplateGUI/state.json`; dataset-local state and computation caches live
under `<dataset>/.phy/`. Only keys explicitly designated as local are written
to the local state. The spike-count controller settings above are global.

Runtime changes, such as `:wn`, take effect after startup and are saved when the
GUI closes normally. A plugin's `on_gui_ready` callback runs after state
restoration but before normal user interaction, and can deliberately override
restored state on every launch.

## Choosing practical values

Increase one setting at a time and test selection latency on a representative
large cluster. Waveform and raw-amplitude increases can cause substantial disk
I/O. Correlogram cost grows with both spike count and the number of displayed
clusters, because the view computes auto- and cross-correlograms for the
selection. A high correlogram spike limit combined with the 20-cluster display
limit can be much more expensive than increasing an individual scatter view.
