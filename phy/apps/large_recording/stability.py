"""Selected-unit full-duration evidence with explicit acquisition exposure."""

from collections import OrderedDict
from sys import getsizeof

import numpy as np
from phylib.utils import Bunch, connect, unconnect

from phy.cluster.views import ManualClusteringView
from phy.gui.qt import screenshot, screenshot_default_path
from phy.plot.plot import PlotCanvasMpl
from phy.utils.color import selected_cluster_color


class UnitStabilityView(ManualClusteringView):
    """Compare exact rates and available source trends for selected units."""

    plot_canvas_class = PlotCanvasMpl
    # Apply the display limit locally, retaining the full presentation selection.
    max_n_clusters = 0
    display_limit = 32
    rate_line_width = 0.6
    display_bin_seconds = 0.0
    summary_cache_bytes = 64 * 1024 * 1024
    _default_position = 'right'
    ex_status = 'Full-duration evidence; sampling cannot establish chronic unit identity.'
    _metrics = ('firing_rate', 'amplitude', 'depth')

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.canvas.subplots(3, 1, sharex=True)
        self.show_firing_rate = self.show_amplitude = self.show_depth = True
        self.state_attrs += tuple(f'show_{metric}' for metric in self._metrics)
        self.state_attrs += ('display_limit', 'rate_line_width')
        self.local_state_attrs += ('display_bin_seconds',)
        self._summary_arrays = {}
        self._selected_summaries = OrderedDict()
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
            data.update(
                counts=None,
                trends={},
                trend_sums={},
                trend_counts={},
                loaded=set(),
                pending=False,
                display=None,
            )
            if session is None:
                data.provenance = 'Loaded recording; continuous acquisition assumed.'
            else:
                units = controller._source_units(cluster_id)
                get_live = getattr(controller, 'get_live_summary', lambda c: None)
                data.live_summary = get_live(cluster_id) if units is None else None
                data.pending = units is None and data.live_summary is None
                data.rows = [] if units is None else [common.lookup[c] for c in units]
                get_status = getattr(
                    controller,
                    'get_summary_status',
                    lambda c: 'Split evidence pending full-data replay.',
                )
                data.provenance = (
                    get_status(cluster_id)
                    if data.pending
                    else 'Complete source sorting; exact counts and finite-value means.'
                )
            self._selected_summaries[cluster_id] = data
        else:
            self._selected_summaries.move_to_end(cluster_id)
        if data.pending:
            self._trim_cache()
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
                    if data.live_summary is not None:
                        counts = data.live_summary['counts']
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
                    if data.live_summary is not None:
                        total = data.live_summary['trend_sums'][metric]
                        valid = data.live_summary['trend_counts'][metric]
                    else:
                        total = np.zeros(len(data.exposure))
                        valid = np.zeros(len(data.exposure), dtype=np.int64)
                        sums = self._array(common.summary[sum_file])
                        counts = self._array(common.summary[count_file])
                        for row in data.rows:
                            total += sums[row]
                            valid += counts[row]
                    mean = np.full(len(data.exposure), np.nan)
                    np.divide(total, valid, out=mean, where=(valid > 0) & (data.exposure > 0))
                    data.trend_sums[metric] = total
                    data.trend_counts[metric] = valid
                    data.trends[metric] = mean
            data.loaded.add(metric)
            data.display = None
        self._trim_cache()
        return data

    @property
    def displayed_ids(self):
        """Presentation subset; the supervisor selection remains authoritative."""
        return self.cluster_ids[: self.display_limit] if self.display_limit else self.cluster_ids

    @staticmethod
    def _data_bytes(data):
        # Common arrays are shared/mapped, and are deliberately outside the row cache.
        overhead = getsizeof(data) + getsizeof(data.loaded)
        rows = data.get('rows', [])
        overhead += getsizeof(rows) + sum(map(getsizeof, rows))
        arrays = [data.get('counts'), data.get('rates')]
        for key in ('trends', 'trend_sums', 'trend_counts'):
            overhead += getsizeof(data.get(key, {}))
            arrays.extend(data.get(key, {}).values())
        display = data.get('display')
        if display is not None:
            overhead += getsizeof(display) + getsizeof(display.trends)
            arrays.extend([display.edges, display.exposure, display.counts, display.rates])
            arrays.extend(display.trends.values())
        return overhead + sum(
            a.nbytes + getsizeof(a) - (a.nbytes if a.flags.owndata else 0)
            for a in arrays
            if isinstance(a, np.ndarray)
        )

    def _trim_cache(self):
        total = sum(self._data_bytes(data) for data in self._selected_summaries.values())
        while self._selected_summaries and total > self.summary_cache_bytes:
            _, data = self._selected_summaries.popitem(last=False)
            total -= self._data_bytes(data)

    def _display_factor(self, seconds=None):
        seconds = self.display_bin_seconds if seconds is None else float(seconds)
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Display bin size must be zero (native) or finite and positive.')
        if seconds == 0:
            return 1
        widths = np.diff(self._common().edges)
        native = widths[0] if len(widths) else np.nan
        if (
            not np.all(np.isfinite(widths))
            or not np.isfinite(native)
            or native <= 0
            or not np.allclose(widths[:-1], native, rtol=1e-10, atol=1e-10)
            or widths[-1] <= 0
            or (widths[-1] > native and not np.isclose(widths[-1], native, rtol=1e-10, atol=1e-10))
        ):
            raise ValueError(
                'Display bin size in seconds requires uniform native bins, '
                'with an optional shorter final bin. Use 0 for native bins.'
            )
        factor = seconds / native
        if (
            not np.isfinite(factor)
            or factor < 1
            or not np.isclose(factor, round(factor), rtol=1e-10, atol=1e-10)
        ):
            raise ValueError(
                'Display bin size must be an integer multiple of the native bin size.'
            )
        return int(round(factor))

    def get_display_data(self, cluster_id, metrics=None):
        """Group adjacent native bins using additive evidence, without accessing spikes."""
        data = self.get_data(cluster_id, metrics)
        factor = self._display_factor()
        if data.pending or factor == 1:
            return data
        if data.display is None or data.display.factor != factor:
            starts = np.arange(0, len(data.exposure), min(factor, len(data.exposure)))
            exposure = np.add.reduceat(data.exposure, starts)
            counts = None if data.counts is None else np.add.reduceat(data.counts, starts)
            rates = np.full(len(starts), np.nan)
            if counts is not None:
                np.divide(counts, exposure, out=rates, where=exposure > 0)
            trends = {}
            for metric, total in data.trend_sums.items():
                sums = np.add.reduceat(total, starts)
                valid = np.add.reduceat(data.trend_counts[metric], starts)
                mean = np.full(len(starts), np.nan)
                np.divide(sums, valid, out=mean, where=(valid > 0) & (exposure > 0))
                trends[metric] = mean
            data.display = Bunch(
                factor=factor,
                edges=np.r_[data.edges[starts], data.edges[-1]],
                exposure=exposure,
                counts=counts,
                rates=rates,
                trends=trends,
                pending=False,
                provenance=data.provenance,
            )
        display = data.display
        self._trim_cache()
        return display

    def set_display_limit(self, value):
        """Set maximum displayed units (0 for all; otherwise a positive integer)."""
        number = float(value)
        if not np.isfinite(number) or number != int(number) or number < 0:
            raise ValueError('Display limit must be an nonnegative integer (0 shows all).')
        self.display_limit = int(number)
        self.plot()

    def set_rate_line_width(self, value):
        """Set rate line width in points (finite and positive)."""
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError('Rate line width must be finite and positive.')
        self.rate_line_width = number
        self.plot()

    def set_display_bin_seconds(self, value):
        """Set display bin size in seconds (0 uses native prepared bins)."""
        self._display_factor(value)
        self.display_bin_seconds = float(value)
        for data in self._selected_summaries.values():
            data.display = None
        self.plot()

    def set_state(self, state):
        state = dict(state)
        for key, default in (
            ('display_limit', 32),
            ('rate_line_width', 0.6),
            ('display_bin_seconds', 0.0),
        ):
            if key not in state:
                continue
            try:
                number = float(state[key])
                valid = np.isfinite(number)
                if key == 'display_limit':
                    valid = valid and number.is_integer() and number >= 0
                elif key == 'rate_line_width':
                    valid = valid and number > 0
                else:
                    self._display_factor(number)
                state[key] = int(number) if key == 'display_limit' and valid else number
                if not valid:
                    state[key] = default
            except (ValueError, TypeError, OverflowError):
                state[key] = default
        super().set_state(state)

    def screenshot(self, dir=None):
        """Save the actual Matplotlib Qt canvas using normal screenshot naming."""
        path = screenshot_default_path(self, dir=dir)
        return screenshot(self.canvas.canvas, path=path)

    @property
    def status(self):
        n = len(self.cluster_ids)
        return (
            f'Showing first {self.display_limit} of {n} selected units; '
            'display truncation does not change the selection or merge operation.'
            if self.display_limit and n > self.display_limit
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
            for index, cluster_id in enumerate(self.displayed_ids):
                data = self.get_display_data(cluster_id, visible)
                if data.pending:
                    notes.append(f'Unit {cluster_id}: {data.provenance}')
                    continue
                color = selected_cluster_color(self.cluster_color_index(cluster_id, index))
                x = (data.edges[:-1] + data.edges[1:]) / 7200
                for metric in visible:
                    ax = axes[self._metrics.index(metric)]
                    values = data.rates if metric == 'firing_rate' else data.trends.get(metric)
                    if values is not None and np.any(np.isfinite(values)):
                        ax.plot(
                            x,
                            values,
                            color=color,
                            label=str(cluster_id),
                            linewidth=self.rate_line_width if metric == 'firing_rate' else 1.5,
                        )
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
        selected = set(self.displayed_ids)
        self._selected_summaries = OrderedDict(
            (c, d) for c, d in self._selected_summaries.items() if c in selected
        )
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

    def _release_cache(self):
        self._summary_arrays.clear()
        self._selected_summaries.clear()
        self._common_data = None

    def close(self):
        """Release mapped summary references and cached rows, including unattached views."""
        result = super().close()
        self._release_cache()
        return result

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

        for action in (
            self.set_display_limit,
            self.set_rate_line_width,
            self.set_display_bin_seconds,
        ):
            self.actions.add(action, prompt=True)

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

        @connect(event='bounded_summary_changed', sender=self.controller.supervisor)
        def on_summary(sender):
            self._invalidate()
            if self.auto_update and not self._closed:
                if self.defer_hidden_updates and not self._dock_visible:
                    if self._pending_selection is None:
                        self._pending_selection = (list(self.cluster_ids), {})
                else:
                    self.plot()

        @connect(event='close_view', sender=self)
        def on_close(sender, *args):
            unconnect(on_state, on_cluster, on_summary, on_close)
            self._release_cache()
