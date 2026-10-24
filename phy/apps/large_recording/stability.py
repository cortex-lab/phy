"""Selected-unit full-duration evidence with explicit acquisition exposure."""

import numpy as np
from phylib.utils import Bunch, connect, unconnect

from phy.cluster.views import ManualClusteringView
from phy.plot.plot import PlotCanvasMpl
from phy.utils.color import selected_cluster_color


class UnitStabilityView(ManualClusteringView):
    """Compare exact rates and available source trends for up to eight selected units."""

    plot_canvas_class = PlotCanvasMpl
    # Apply the display limit locally, retaining the full presentation selection.
    max_n_clusters = 0
    display_limit = 8
    _default_position = 'right'
    ex_status = 'Full-duration evidence; sampling cannot establish chronic unit identity.'
    _metrics = ('firing_rate', 'amplitude', 'depth')

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.canvas.subplots(3, 1, sharex=True)
        self.show_firing_rate = self.show_amplitude = self.show_depth = True
        self.state_attrs += tuple(f'show_{metric}' for metric in self._metrics)
        self._summary_arrays = {}
        self._selected_summaries = {}
        self._common_data = None
        self._attaching = False
        self._footer = self.canvas.figure.text(0.12, 0.025, '', fontsize=8)

    def _array(self, name):
        if name not in self._summary_arrays:
            self._summary_arrays[name] = np.load(self.controller.dir_path / name, mmap_mode='r')
        return self._summary_arrays[name]

    def _common(self):
        if self._common_data is not None:
            return self._common_data
        controller = self.controller
        session = getattr(controller, 'bounded_session', None)
        if session is None:
            edges = np.linspace(0, controller.model.duration, 101)
            self._common_data = Bunch(
                edges=edges,
                exposure=np.diff(edges),
                windows=[],
                intervals=[[0, controller.model.duration]],
                footer='',
            )
        else:
            manifest = session.manifest
            summary = manifest.get('summaries', manifest)
            edges = self._array(summary['edges_file']) / controller.model.sample_rate
            missing = [c for c, n in zip(session.unit_ids, manifest['bounded_counts']) if not n]
            footer = ''
            if missing:
                footer = (
                    f'{len(missing)} source units without evidence: '
                    f'{", ".join(map(str, missing[:12]))}' + (' …' if len(missing) > 12 else '')
                )
            self._common_data = Bunch(
                edges=edges,
                exposure=self._array(summary['exposure_file']),
                windows=controller.evidence_windows / controller.model.sample_rate,
                intervals=np.asarray(manifest['valid_intervals']) / controller.model.sample_rate,
                summary=summary,
                lookup={int(c): i for i, c in enumerate(session.unit_ids)},
                footer=footer,
            )
        return self._common_data

    def get_data(self, cluster_id, metrics=None):
        """Load only requested metrics, caching bounded selected-unit summary rows.

        Counts share the manifest's exposure. Trend sums are combined with their
        finite-value counts, never averaged as equally weighted unit means.
        """
        metrics = self._metrics if metrics is None else metrics
        common = self._common()
        data = self._selected_summaries.get(cluster_id)
        controller = self.controller
        session = getattr(controller, 'bounded_session', None)
        if data is None:
            data = Bunch(common.copy())
            data.update(counts=None, trends={}, loaded=set(), pending=False)
            if session is None:
                data.provenance = 'Loaded recording; continuous acquisition assumed.'
            else:
                units = controller._source_units(cluster_id)
                data.pending = units is None
                data.rows = [] if units is None else [common.lookup[c] for c in units]
                data.provenance = (
                    'Split evidence pending full-data replay.'
                    if data.pending
                    else 'Complete source sorting; exact counts and finite-value means.'
                )
            # Keep at most eight summaries, including direct source-unit inspection.
            if len(self._selected_summaries) >= self.display_limit:
                self._selected_summaries.pop(next(iter(self._selected_summaries)))
            self._selected_summaries[cluster_id] = data
        if data.pending:
            return data
        for metric in metrics:
            if metric in data.loaded:
                continue
            if metric == 'firing_rate':
                if session is None:
                    data.counts = np.histogram(controller.get_spike_times(cluster_id), data.edges)[
                        0
                    ]
                else:
                    counts = np.zeros(len(data.exposure), dtype=np.int64)
                    grid = self._array(common.summary['counts_file'])
                    for row in data.rows:
                        counts += grid[row]
                    data.counts = counts
                data.rates = np.full(len(data.exposure), np.nan)
                np.divide(data.counts, data.exposure, out=data.rates, where=data.exposure > 0)
            elif session is not None:
                sum_file, count_file = f'{metric}_sum_file', f'{metric}_count_file'
                if sum_file in common.summary and count_file in common.summary:
                    total = np.zeros(len(data.exposure))
                    valid = np.zeros(len(data.exposure), dtype=np.int64)
                    sums = self._array(common.summary[sum_file])
                    counts = self._array(common.summary[count_file])
                    for row in data.rows:
                        total += sums[row]
                        valid += counts[row]
                    mean = np.full(len(data.exposure), np.nan)
                    np.divide(total, valid, out=mean, where=valid > 0)
                    data.trends[metric] = mean
            data.loaded.add(metric)
        return data

    @property
    def status(self):
        n = len(self.cluster_ids)
        return (
            f'Showing first {self.display_limit} of {n} selected units.'
            if n > self.display_limit
            else ''
        )

    def _layout_axes(self, visible):
        axes = list(self.canvas.iter_ax())
        for metric, ax in zip(self._metrics, axes):
            ax.clear()
            ax.set_visible(metric in visible)
        for index, metric in enumerate(visible):
            ax = axes[self._metrics.index(metric)]
            height = 0.76 / len(visible)
            ax.set_position(
                [
                    0.12,
                    0.15 + (len(visible) - 1 - index) * height,
                    0.84,
                    height * (0.84 if len(visible) > 1 else 1),
                ]
            )
            ax.tick_params(axis='x', labelbottom=index == len(visible) - 1)
            if index == len(visible) - 1:
                ax.set_xlabel('Elapsed acquisition clock (hours)')
        return axes

    def plot(self, **kwargs):
        visible = [metric for metric in self._metrics if getattr(self, f'show_{metric}')]
        axes = self._layout_axes(visible)
        self._footer.set_text('')
        # Reuse an existing axis for the explicitly empty panel.
        if not visible:
            axes[0].set_visible(True)
            axes[0].set_position([0.12, 0.15, 0.84, 0.76])
            axes[0].set_axis_off()
            axes[0].text(
                0.02,
                0.6,
                'No metrics enabled. Use Show firing rate, amplitude or depth.',
                transform=axes[0].transAxes,
            )
        else:
            common = self._common()
            self._footer.set_text(common.footer)
            first_ax = axes[self._metrics.index(visible[0])]
            notes = []
            unavailable = {metric: [] for metric in visible}
            if not self.cluster_ids:
                notes.append('Select a unit to inspect full-duration evidence.')
            for index, cluster_id in enumerate(self.cluster_ids[: self.display_limit]):
                data = self.get_data(cluster_id, visible)
                if data.pending:
                    notes.append(f'Unit {cluster_id}: {data.provenance}')
                    continue
                color = selected_cluster_color(self.cluster_color_index(cluster_id, index))
                x = (data.edges[:-1] + data.edges[1:]) / 7200
                for metric in visible:
                    ax = axes[self._metrics.index(metric)]
                    values = data.rates if metric == 'firing_rate' else data.trends.get(metric)
                    if values is not None and np.any(np.isfinite(values)):
                        ax.plot(x, values, color=color, label=str(cluster_id))
                    elif metric != 'firing_rate':
                        unavailable[metric].append(cluster_id)
            labels = ('Rate (Hz)', 'Mean amplitude\n(source units)', 'Mean depth\n(source units)')
            for metric in visible:
                ax = axes[self._metrics.index(metric)]
                ax.set_axis_on()
                ax.set_ylabel(labels[self._metrics.index(metric)])
                if unavailable[metric]:
                    missing = ', '.join(map(str, unavailable[metric]))
                    message = (
                        f'{metric.title()} measurement unavailable.'
                        if not ax.lines
                        else f'{metric.title()} measurement unavailable for units: {missing}.'
                    )
                    ax.text(0.02, 0.96, message, transform=ax.transAxes, va='top', fontsize=8)
                for start, end in common.windows:
                    ax.axvspan(start / 3600, end / 3600, color='cyan', alpha=0.15)
                previous = 0
                for start, end in common.intervals:
                    if start > previous:
                        ax.axvspan(previous / 3600, start / 3600, color='gray', alpha=0.4)
                    previous = end
                if previous < common.edges[-1]:
                    ax.axvspan(previous / 3600, common.edges[-1] / 3600, color='gray', alpha=0.4)
                ax.set_xlim(common.edges[0] / 3600, common.edges[-1] / 3600)
                if ax.lines:
                    ax.legend(
                        title='Unit', fontsize=8, loc='upper right', ncol=min(4, len(ax.lines))
                    )
            provenance = (
                'Complete source sorting; exact counts and finite-value means.'
                if getattr(self.controller, 'bounded_session', None) is not None
                else 'Loaded recording; continuous acquisition assumed.'
            )
            first_ax.set_title(provenance, fontsize=9)
            if notes:
                first_ax.text(
                    0.02,
                    0.96,
                    '\n'.join(notes),
                    transform=first_ax.transAxes,
                    va='top',
                    fontsize=8,
                )
        self.update_status()
        self.canvas.update()

    def on_select(self, cluster_ids=None, **kwargs):
        if self._attaching:
            supervisor = self.controller.supervisor
            cluster_ids = supervisor.selected
            self._update_cluster_color_indices(supervisor, cluster_ids)
        self.cluster_ids = tuple(cluster_ids or ())
        selected = set(self.cluster_ids[: self.display_limit])
        self._selected_summaries = {
            c: d for c, d in self._selected_summaries.items() if c in selected
        }
        self.plot(**kwargs)

    def _update_cluster_color_indices(self, sender, cluster_ids):
        # Hidden-view replay already adopted the authoritative palette at receipt.
        if sender is not self:
            super()._update_cluster_color_indices(sender, cluster_ids)

    def on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs):
        # The base intentionally ignores empty selections; this evidence panel must clear.
        if not cluster_ids:
            if (
                not self.auto_update
                or self._closed
                or sender.__class__.__name__ in ('ClusterView', 'SimilarityView')
            ):
                return
            self._update_cluster_color_indices(sender, cluster_ids)
            if self.defer_hidden_updates and not self._dock_visible:
                self.cluster_ids = ()
                self._pending_selection = ([], dict(kwargs))
                return
            self._pending_selection = None
            self.on_select([], **kwargs)
            return
        return super().on_select_threaded(sender, cluster_ids, gui=gui, **kwargs)

    def _toggle_metric(self, metric, checked):
        setattr(self, f'show_{metric}', bool(checked))
        if hasattr(self, 'actions'):
            self.actions.get(f'toggle_{metric}').setChecked(bool(checked))
        if not self._closed:
            self.plot()

    def toggle_firing_rate(self, checked):
        """Show firing rate."""
        self._toggle_metric('firing_rate', checked)

    def toggle_amplitude(self, checked):
        """Show amplitude."""
        self._toggle_metric('amplitude', checked)

    def toggle_depth(self, checked):
        """Show depth."""
        self._toggle_metric('depth', checked)

    def _invalidate(self):
        self._selected_summaries.clear()

    def attach(self, gui):
        self._attaching = True
        try:
            super().attach(gui)
        finally:
            self._attaching = False
        for metric in self._metrics:
            self.actions.add(
                getattr(self, f'toggle_{metric}'),
                checkable=True,
                checked=getattr(self, f'show_{metric}'),
                docstring=f'Show {metric.replace("_", " ")}.',
            )
            self.actions.get(f'toggle_{metric}').setText(f'Show {metric.replace("_", " ")}')

        @self.actions.add(prompt=True)
        def show_source_unit(cluster_id):
            """Inspect a full source unit, including one without sampled evidence."""
            session = getattr(self.controller, 'bounded_session', None)
            if session is not None and int(cluster_id) not in session.unit_ids:
                raise ValueError('Unknown source unit ID.')
            self._cluster_color_index_by_id = {}
            self.on_select([int(cluster_id)])

        @connect(event='bounded_state_changed', sender=self.controller.supervisor)
        def on_state(sender):
            self._invalidate()
            if not self._closed:
                self.plot()

        @connect(event='cluster', sender=self.controller.supervisor)
        def on_cluster(sender, up):
            self._invalidate()
            # Supervisor selection callbacks may precede this listener. Refresh
            # ordinary data after invalidation even if the selected IDs stayed put.
            if (
                getattr(self.controller, 'bounded_session', None) is None
                and self.auto_update
                and not self._closed
                and not (self.defer_hidden_updates and not self._dock_visible)
            ):
                self.plot()

        @connect(event='close_view', sender=self)
        def on_close(sender, *args):
            unconnect(on_state, on_cluster, on_close)
            self._summary_arrays.clear()
            self._selected_summaries.clear()
            self._common_data = None
