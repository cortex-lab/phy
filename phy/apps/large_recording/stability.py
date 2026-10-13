"""Selected-unit full-duration evidence with explicit acquisition exposure."""

import numpy as np
from phylib.utils import Bunch, connect, unconnect

from phy.cluster.views import ManualClusteringView
from phy.plot.plot import PlotCanvasMpl


class UnitStabilityView(ManualClusteringView):
    """Full counts/rates, available source trends, gaps and retained evidence windows."""

    plot_canvas_class = PlotCanvasMpl
    max_n_clusters = 1
    _default_position = 'right'
    ex_status = 'Full-duration evidence; sampling cannot establish chronic unit identity.'

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.canvas.subplots(3, 1, sharex=True)
        self._summary_arrays = {}

    def _array(self, name):
        if name not in self._summary_arrays:
            self._summary_arrays[name] = np.load(self.controller.dir_path / name, mmap_mode='r')
        return self._summary_arrays[name]

    def get_data(self, cluster_id):
        controller = self.controller
        session = getattr(controller, 'bounded_session', None)
        if session is None:
            # Optional ordinary-data view: only the selected cluster is evaluated.
            times = controller.get_spike_times(cluster_id)
            edges = np.linspace(0, controller.model.duration, 101)
            counts, _ = np.histogram(times, edges)
            exposure = np.diff(edges)
            return Bunch(
                edges=edges,
                counts=counts,
                exposure=exposure,
                windows=[],
                intervals=[[0, controller.model.duration]],
                trends={},
                provenance='Loaded recording; continuous acquisition assumed.',
            )
        manifest = session.manifest
        summary = manifest.get('summaries', manifest)
        edges = self._array(summary['edges_file']) / controller.model.sample_rate
        exposure = self._array(summary['exposure_file'])
        units = controller._source_units(cluster_id)
        if units is None:
            return Bunch(
                edges=edges,
                counts=None,
                exposure=exposure,
                windows=[],
                intervals=[],
                trends={},
                provenance='Split evidence pending full-data replay.',
            )
        lookup = {int(c): i for i, c in enumerate(session.unit_ids)}
        rows = [lookup[c] for c in units]
        # Bound temporaries by one unit's summary row, including merged units.
        count_grid = self._array(summary['counts_file'])
        counts = np.zeros(len(exposure), dtype=np.int64)
        for row in rows:
            counts += count_grid[row]
        trends = {}
        for attribute in ('amplitude', 'depth'):
            sum_file, count_file = f'{attribute}_sum_file', f'{attribute}_count_file'
            if sum_file not in summary:
                continue
            total, valid = np.zeros(len(exposure)), np.zeros(len(exposure), dtype=np.int64)
            for row in rows:
                total += self._array(summary[sum_file])[row]
                valid += self._array(summary[count_file])[row]
            mean = np.full(len(exposure), np.nan)
            np.divide(total, valid, out=mean, where=valid > 0)
            trends[attribute] = mean
        return Bunch(
            edges=edges,
            counts=counts,
            exposure=exposure,
            trends=trends,
            windows=controller.evidence_windows / controller.model.sample_rate,
            intervals=np.asarray(manifest['valid_intervals']) / controller.model.sample_rate,
            provenance='Complete source sorting; exact counts and finite-value means.',
        )

    def plot(self, **kwargs):
        axes = list(self.canvas.iter_ax())
        for ax in axes:
            ax.clear()
        session = getattr(self.controller, 'bounded_session', None)
        if not self.cluster_ids:
            axes[0].text(
                0.02,
                0.6,
                'Select a unit to inspect full-duration evidence.',
                transform=axes[0].transAxes,
            )
        else:
            cluster_id = self.cluster_ids[0]
            data = self.get_data(cluster_id)
            if data.counts is None:
                axes[0].text(0.02, 0.6, data.provenance, transform=axes[0].transAxes)
            else:
                rates = np.full(len(data.exposure), np.nan)
                np.divide(data.counts, data.exposure, out=rates, where=data.exposure > 0)
                x = (data.edges[:-1] + data.edges[1:]) / 7200
                axes[0].plot(x, rates)
                for ax, attribute in zip(axes[1:], ('amplitude', 'depth')):
                    if attribute in data.trends:
                        ax.plot(x, data.trends[attribute])
                    else:
                        ax.text(
                            0.02,
                            0.6,
                            f'{attribute.title()} measurement unavailable.',
                            transform=ax.transAxes,
                        )
                for ax in axes:
                    for start, end in data.windows:
                        ax.axvspan(start / 3600, end / 3600, color='cyan', alpha=0.15)
                    previous = 0
                    for start, end in data.intervals:
                        if start > previous:
                            ax.axvspan(previous / 3600, start / 3600, color='gray', alpha=0.4)
                        previous = end
                    if previous < data.edges[-1]:
                        ax.axvspan(previous / 3600, data.edges[-1] / 3600, color='gray', alpha=0.4)
                axes[0].set_title(f'Unit {cluster_id} — {data.provenance}', fontsize=9)
        axes[0].set_ylabel('Rate (Hz)')
        axes[1].set_ylabel('Mean amplitude\n(source units)')
        axes[2].set_ylabel('Mean depth\n(source units)')
        axes[2].set_xlabel('Elapsed acquisition clock (hours)')
        if session is not None:
            missing = [
                c for c, n in zip(session.unit_ids, session.manifest['bounded_counts']) if not n
            ]
            if missing:
                axes[2].text(
                    0.02,
                    -0.42,
                    f'{len(missing)} source units without evidence: '
                    f'{", ".join(map(str, missing[:12]))}' + (' …' if len(missing) > 12 else ''),
                    transform=axes[2].transAxes,
                    fontsize=8,
                )
        self.canvas.figure.tight_layout()
        self.canvas.update()

    def on_select(self, cluster_ids=None, **kwargs):
        self.cluster_ids = tuple(cluster_ids or ())[:1]
        self.plot(**kwargs)

    def attach(self, gui):
        super().attach(gui)

        @self.actions.add(prompt=True)
        def show_source_unit(cluster_id):
            """Inspect a full source unit, including one without sampled evidence."""
            session = getattr(self.controller, 'bounded_session', None)
            if session is not None and int(cluster_id) not in session.unit_ids:
                raise ValueError('Unknown source unit ID.')
            self.on_select([int(cluster_id)])

        @connect(event='bounded_state_changed', sender=self.controller.supervisor)
        def on_state(sender):
            self.plot()

        @connect(event='close_view', sender=self)
        def on_close(sender, *args):
            unconnect(on_state, on_close)
            self._summary_arrays.clear()
