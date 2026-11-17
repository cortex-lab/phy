"""Exact merge-parent lineage and isolated, read-only comparison providers."""

import hashlib
import json
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np
from phylib.stats import correlograms
from phylib.utils import Bunch, connect, emit, unconnect

from phy.gui.qt import Qt
from phy.utils.color import colormaps

logger = logging.getLogger(__name__)


def _fingerprint(assignments, model):
    digest = hashlib.sha256()
    digest.update(str(len(assignments)).encode())
    for start in range(0, len(assignments), 1_000_000):
        digest.update(np.asarray(assignments[start : start + 1_000_000], dtype='<i8').tobytes())
    # Row identity matters as well as cluster labels: exchanging source spikes
    # within one current cluster must not silently retarget historical parents.
    for name, dtype in (('spike_times', '<f8'), ('spike_templates', '<i8')):
        values = getattr(model, name, None)
        digest.update(name.encode())
        if values is not None:
            for start in range(0, len(values), 1_000_000):
                digest.update(np.asarray(values[start : start + 1_000_000], dtype=dtype).tobytes())
    return digest.hexdigest()


class ParentLineage:
    """Retain exact ordinary merge partitions; bounded journals are authoritative.

    Ordinary sidecars are published only after a successful curation save and
    bound to the saved assignment vector. Legacy merges are deliberately unknown.
    """

    def __init__(self, controller):
        self.controller = controller
        self.records = {}
        self.color_records = {}
        self.origins = {}
        self.restorations = {}
        self._pending_colors = None
        self.path = controller.dir_path / '_phy_merge_parents.npz'
        if not hasattr(controller, 'bounded_session'):
            self._load()
        else:
            self._load_bounded_colors()
        self.bind()

    def bind(self):
        self.close()
        supervisor = self.controller.supervisor
        connect(self._before_merge, event='before_merge', sender=supervisor)
        connect(self._on_cluster, event='cluster', sender=supervisor.clustering)
        connect(self._on_saved, event='curation_saved', sender=supervisor)

    def close(self):
        unconnect(self._before_merge, self._on_cluster, self._on_saved)

    def _before_merge(self, sender, parents):
        # This explicit observer runs before clustering/table callbacks can reset
        # the source selection. Unselected explicit parents get unused slots.
        self._pending_colors = None
        if len(parents) > 16:
            return  # Display limits must never restrict otherwise valid curation.
        selected = getattr(sender, 'selection_color_indices', {})
        colors = {int(p): int(selected[p]) for p in parents if p in selected}
        used = set(colors.values())
        for parent in parents:
            if parent not in colors:
                slot = next(i for i in range(len(colormaps.default)) if i not in used)
                colors[int(parent)] = slot
                used.add(slot)
        self._pending_colors = colors

    def _on_cluster(self, sender, up):
        if up.history == 'undo':
            for target in up.added:
                self.restorations.pop(int(target), None)
        if up.description == 'restore_parents' and up.history != 'undo':
            self.restorations[int(up.parent_restore_target)] = list(up.parent_restore_origins)
            for child, parent in up.parent_restore_origins.items():
                self.origins[int(child)] = int(parent)
                if parent in self.records:
                    self.records[int(child)] = self.records[parent]
                if parent in self.color_records:
                    self.color_records[int(child)] = self.color_records[parent]
        if up.description == 'merge' and up.history != 'undo' and len(up.deleted) >= 2:
            target = int(up.added[0])
            if not hasattr(self.controller, 'bounded_session'):
                # Keep references: Clustering replaces these arrays, never mutates them.
                self.records[target] = dict(up.parent_spike_ids)
            if up.history != 'redo':
                colors = self._pending_colors
                if colors is not None and set(colors) == set(up.deleted):
                    self.color_records[target] = colors
                else:
                    self.color_records.pop(target, None)
            self._pending_colors = None

    @staticmethod
    def _parse_colors(value):
        if not isinstance(value, dict):
            raise ValueError('invalid optional parent colors')
        result = {}
        for target, groups in value.items():
            if not isinstance(groups, dict):
                raise ValueError('invalid optional parent colors')
            colors = {}
            for parent, index in groups.items():
                if type(index) is not int or index < 0:
                    raise ValueError('invalid optional parent color index')
                colors[int(parent)] = index
            if len(set(colors.values())) != len(colors):
                raise ValueError('duplicate optional parent color indices')
            result[int(target)] = colors
        return result

    def _bounded_color_path(self):
        identity = hashlib.sha256(str(self.controller.dir_path.resolve()).encode()).hexdigest()[
            :20
        ]
        return Path(self.controller.config_dir) / f'parent-colors-{identity}.json'

    def _bounded_binding(self):
        session = self.controller.bounded_session
        value = [session.manifest['source'], session.manifest.get('bounded_files'), session.events]
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _load_bounded_colors(self):
        try:
            value = json.loads(self._bounded_color_path().read_text())
            if value['binding'] == self._bounded_binding():
                self.color_records = self._parse_colors(value['colors'])
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Disposable display hints never control journal validity.

    def colors(self, target, groups=None):
        """Immutable parent→palette slots, persisted where known, stable otherwise.

        Historical merges without optional display hints use numeric parent order.
        ``groups`` lets a caller reuse an already validated exact partition.
        """
        groups = self.parents(target) if groups is None else groups
        saved = self.color_records.get(int(target), {})
        colors = {p: saved[p] for p in groups if p in saved}
        used = set(colors.values())
        for parent in sorted(groups):
            if parent not in colors:
                index = next(i for i in range(len(groups) + len(used) + 1) if i not in used)
                colors[parent] = index
                used.add(index)
        return MappingProxyType(colors)

    def _load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open('rb') as stream, np.load(stream, allow_pickle=False) as data:
                if int(data['version']) != 1 or str(data['fingerprint']) != _fingerprint(
                    self.controller.supervisor.clustering.spike_clusters, self.controller.model
                ):
                    raise ValueError('saved assignments differ from recorded lineage')
                catalog = json.loads(str(data['catalog']))
                if not isinstance(catalog, dict):
                    raise ValueError('invalid lineage catalog')
                records = {}
                for target, parents in catalog.items():
                    target = int(target)
                    groups = {}
                    for parent in parents:
                        ids = data[f'g_{target}_{parent}']
                        if (
                            ids.dtype.kind not in 'iu'
                            or ids.ndim != 1
                            or (
                                len(ids)
                                and (
                                    ids[0] < 0
                                    or ids[-1] >= len(self.controller.model.spike_times)
                                    or np.any(ids[1:] <= ids[:-1])
                                )
                            )
                        ):
                            raise ValueError('invalid parent spike membership')
                        ids.flags.writeable = False
                        groups[int(parent)] = ids
                    if len(groups) < 2:
                        raise ValueError('a merge needs at least two parents')
                    records[target] = groups
                self.records = records
                if 'origins' in data:
                    origins = json.loads(str(data['origins']))
                    if not isinstance(origins, dict) or any(
                        type(value) is not int or value < 0 for value in origins.values()
                    ):
                        raise ValueError('invalid restored parent origins')
                    self.origins = {int(child): parent for child, parent in origins.items()}
                if 'restorations' in data:
                    restorations = json.loads(str(data['restorations']))
                    if not isinstance(restorations, dict) or any(
                        not isinstance(children, list)
                        or len(children) < 2
                        or len(set(children)) != len(children)
                        or any(type(c) is not int or c < 0 for c in children)
                        for children in restorations.values()
                    ):
                        raise ValueError('invalid restored merge history')
                    self.restorations = {int(t): children for t, children in restorations.items()}
                try:
                    self.color_records = self._parse_colors(json.loads(str(data['color_indices'])))
                except (OSError, ValueError, KeyError, TypeError, BadZipFile):
                    self.color_records = {}
        except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
            logger.warning('Merge-parent history unavailable: %s.', error)
            self.records = {}
            self.origins = {}
            self.restorations = {}

    def _on_saved(self, sender):
        if hasattr(self.controller, 'bounded_session'):
            # The journal is already committed; optional cache failures must not
            # turn a successful curation save into a reported failure.
            try:
                path = self._bounded_color_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, filename = tempfile.mkstemp(prefix='.parent-colors-', dir=path.parent)
                try:
                    with os.fdopen(descriptor, 'w') as stream:
                        json.dump(
                            {'binding': self._bounded_binding(), 'colors': self.color_records},
                            stream,
                        )
                    os.replace(filename, path)
                finally:
                    if os.path.exists(filename):
                        os.unlink(filename)
            except (OSError, ValueError, TypeError):
                logger.debug('Could not save optional parent colors.', exc_info=True)
            return
        live = self.controller.supervisor.clustering.spikes_per_cluster
        # Keep nested ancestry reachable through live merged or restored units.
        # A restored alias shares immutable partitions with its historical unit.
        reachable, pending = set(), list(live)
        while pending:
            target = pending.pop()
            if target in reachable:
                continue
            reachable.add(target)
            pending.extend(self.records.get(target, {}))
        records = {
            target: groups for target, groups in self.records.items() if target in reachable
        }
        arrays = {
            f'g_{target}_{parent}': ids
            for target, groups in records.items()
            for parent, ids in groups.items()
        }
        arrays.update(
            version=np.asarray(1),
            fingerprint=np.asarray(
                _fingerprint(
                    self.controller.supervisor.clustering.spike_clusters, self.controller.model
                )
            ),
            catalog=np.asarray(json.dumps({str(t): sorted(g) for t, g in records.items()})),
            origins=np.asarray(
                json.dumps({t: p for t, p in self.origins.items() if t in reachable})
            ),
            restorations=np.asarray(json.dumps(self.restorations)),
            color_indices=np.asarray(
                json.dumps({t: self.color_records[t] for t in records if t in self.color_records})
            ),
        )
        descriptor, filename = tempfile.mkstemp(
            prefix='.phy-parents-', suffix='.npz', dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(filename, self.path)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def parents(self, target):
        target = int(target)
        live = self.controller.supervisor.clustering.spikes_per_cluster
        if target not in live:
            raise ValueError('Choose a current merged unit.')
        session = getattr(self.controller, 'bounded_session', None)
        if session is not None:
            _, groups = session.restore_parent_groups(target)
        else:
            groups = self.records.get(target)
            if groups is None:
                raise ValueError(
                    'Merge parents unavailable: this merge predates recorded lineage.'
                )
        # Check exact partition, including splits, undo/redo and external changes.
        union = np.sort(np.concatenate(list(groups.values())))
        if not np.array_equal(union, live[target]) or len(np.unique(union)) != len(union):
            raise ValueError('Recorded parents do not exactly partition this unit.')
        result = {}
        saved = self.color_records.get(target, {})
        for parent in sorted(groups, key=lambda p: (saved.get(p, len(colormaps.default)), p)):
            ids = np.asarray(groups[parent]).view()
            ids.flags.writeable = False
            result[int(parent)] = ids
        return MappingProxyType(result)

    def full_counts(self, target, groups=None):
        """Exact counts from saved summaries/whole-unit merges; splits stay unknown."""
        session = getattr(self.controller, 'bounded_session', None)
        if session is None:
            return {
                p: len(ids)
                for p, ids in (self.parents(target) if groups is None else groups).items()
            }
        counts = {int(p): int(n) for p, n in zip(session.unit_ids, session.manifest['counts'])}
        origin, _ = session.restore_parent_groups(target)
        for index, event in enumerate(session.events):
            if event['operation'] == 'merge':
                parents = event['parents']
                if index == origin:
                    return {p: counts.get(p) for p in parents}
                values = [counts.get(p) for p in parents]
                counts[event['target']] = sum(values) if None not in values else None
            elif event['operation'] == 'threshold_split':
                counts.update(dict.fromkeys(event['children']))
            elif event['operation'] == 'restore_parents':
                parents = session.events[event['merge_event_index']]['parents']
                counts.update(
                    {
                        child: counts.get(parent)
                        for child, parent in zip(event['children'], parents)
                    }
                )
        return {}


class ParentProviders:
    """Providers over explicit spike arrays, never live cluster-ID lookups.

    Parent IDs belong only to this display. No selector, mutable assignment or
    controller provider is replaced. All display sampling uses shared budgets.
    """

    def __init__(self, controller, target, groups):
        self.controller, self.target = controller, int(target)
        self.model, self.groups = controller.model, groups
        self.channels = list(controller.get_best_channels(target))
        self.ids = list(groups)
        self.bounded = hasattr(controller, 'bounded_session')
        if len(groups) > 16:
            raise ValueError('Parent comparison supports at most 16 immediate parents.')

    def sampled(self, parent, per_cluster, total):
        from .base import _allocate_spike_counts, _sample_spikes_evenly

        allocated = _allocate_spike_counts(
            [len(self.groups[p]) for p in self.ids], per_cluster=per_cluster, total=total
        )
        return _sample_spikes_evenly(self.groups[parent], int(allocated[self.ids.index(parent)]))

    def feature_ids(self, parent):
        ids = self.sampled(
            parent,
            getattr(self.controller, 'n_spikes_features', 2500),
            getattr(self.controller, 'n_spikes_features_total', 40000),
        )
        rows = getattr(self.model, 'features_rows', None)
        return np.intersect1d(ids, rows) if rows is not None else ids

    def feature_times(self, parent=None, **kwargs):
        ids = self.feature_ids(parent) if parent is not None else np.array([], dtype=np.int64)
        return Bunch(data=self.model.spike_times[ids], spike_ids=ids, lim=(0, self.model.duration))

    def features(self, parent=None, channel_ids=None, **kwargs):
        if parent is None:
            return Bunch()
        ids = self.feature_ids(parent)
        if not len(ids):
            return Bunch()
        channels = self.channels if channel_ids is None else channel_ids
        return self.controller._get_spike_features(ids, channels)

    def amplitudes(self, parents, **kwargs):
        output = []
        for parent in parents:
            if parent is None:
                ids = np.array([], dtype=np.int64)
            else:
                ids = self.sampled(
                    parent,
                    self.controller.n_spikes_amplitudes,
                    self.controller.n_spikes_amplitudes_total,
                )
            values = self.model.amplitudes
            output.append(
                Bunch(
                    spike_ids=ids,
                    spike_times=self.model.spike_times[ids],
                    amplitudes=values[ids].reshape(-1),
                )
            )
        return output

    def waveform_ids(self, parent):
        from .base import _sample_spikes_evenly

        per_parent = getattr(self.controller, 'n_spikes_waveforms', 100)
        total = getattr(self.controller, 'n_spikes_waveforms_total', None)
        ids = self.sampled(parent, per_parent, total)
        if getattr(self.model, 'spike_waveforms', None) is not None:
            ids = np.intersect1d(self.groups[parent], self.model.spike_waveforms.spike_ids)
            count = min(per_parent, total // len(self.ids)) if total is not None else per_parent
            ids = _sample_spikes_evenly(ids, count)
        return ids

    def waveforms(self, parent):
        ids = self.waveform_ids(parent)
        data = self.model.get_waveforms(ids, self.channels)
        if data is None:
            raise ValueError('Raw/stored waveforms unavailable for these parents.')
        data = data - np.median(data, axis=1)[:, None, :]
        data = self.controller.raw_data_filter.apply(data, axis=1)
        return Bunch(
            data=data,
            spike_ids=ids,
            channel_ids=self.channels,
            channel_labels=self.controller._get_channel_labels(self.channels),
            channel_positions=self.model.channel_positions[self.channels],
            masks=np.ones((len(data), len(self.channels))),
        )

    def templates(self, parent):
        """Template traces weighted by exact membership, never template-based grouping."""
        templates, counts = np.unique(
            self.model.spike_templates[self.groups[parent]], return_counts=True
        )
        keep = np.argsort(-counts, kind='stable')[:100]
        traces, weights = [], []
        for index in keep:
            template = self.model.get_template(int(templates[index]))
            if not template:
                continue
            waveform = np.zeros((self.model.n_samples_waveforms, self.model.n_channels))
            waveform[:, template.channel_ids] = template.template
            traces.append(waveform[:, self.channels])
            weights.append(counts[index])
        if not traces:
            raise ValueError('No template waveforms available for these parent spikes.')
        masks = np.tile(np.asarray(weights)[:, None] / max(weights), (1, len(self.channels)))
        return Bunch(
            data=np.asarray(traces),
            masks=masks,
            channel_ids=self.channels,
            channel_labels=self.controller._get_channel_labels(self.channels),
            channel_positions=self.model.channel_positions[self.channels],
        )

    def correlograms(self, parents, bin_size, window_size):
        if self.bounded:
            from .large_recording.statistics import window_correlograms

            # Only retained evidence rows are visited. No full-source I/O.
            ids = np.sort(np.concatenate([self.groups[p] for p in parents]))
            labels = np.empty(len(ids), dtype=np.int64)
            for index, parent in enumerate(parents):
                labels[np.searchsorted(ids, self.groups[parent])] = index
            return window_correlograms(
                self.model.spike_samples[ids],
                labels,
                self.controller.evidence_windows,
                list(range(len(parents))),
                self.model.sample_rate,
                bin_size,
                window_size,
            )
        from .base import _allocate_spike_counts

        budgets = _allocate_spike_counts(
            [len(self.groups[p]) for p in self.ids],
            per_cluster=self.controller.n_spikes_correlograms,
            total=self.controller.n_spikes_correlograms_total,
        )
        sampled = []
        for parent in parents:
            ids = self.groups[parent]
            n = int(budgets[self.ids.index(parent)])
            # Regular thinning can erase short-lag pairs. Independent reproducible
            # sampling avoids imposing an artificial refractory period.
            if n < len(ids):
                ids = ids[np.random.default_rng(parent).choice(len(ids), size=n, replace=False)]
            sampled.append(ids)
        ids = np.concatenate(sampled)
        labels = np.concatenate([np.full(len(s), index) for index, s in enumerate(sampled)])
        times = self.model.spike_times[ids]
        order = np.argsort(times, kind='stable')
        return correlograms(
            times[order],
            labels[order],
            cluster_ids=np.arange(len(parents)),
            sample_rate=self.model.sample_rate,
            bin_size=bin_size,
            window_size=window_size,
        )


class ParentInspection:
    """Own review widgets and callbacks; never borrow live providers or curation actions."""

    def __init__(self, controller):
        self.controller = controller
        self.lineage = ParentLineage(controller)
        controller.supervisor.parent_lineage = self.lineage
        controller.supervisor.parent_origins.update(self.lineage.origins)
        controller.supervisor._refresh_propositions()
        self.window = None
        self.views = []
        self._gui = None
        connect(self._on_selection, event='select', sender=controller.supervisor)
        connect(self._on_mutation, event='cluster', sender=controller.supervisor)

    def _on_selection(self, sender, *args, **kwargs):
        self.close_review()

    def _on_mutation(self, sender, *args, **kwargs):
        self.close_review()

    def attach(self, gui):
        from phy.gui.qt import QEvent, QMenu, QObject

        self.lineage.bind()
        self.controller.supervisor._refresh_propositions()
        unconnect(self._on_selection, self._on_mutation)
        connect(self._on_selection, event='select', sender=self.controller.supervisor)
        connect(self._on_mutation, event='cluster', sender=self.controller.supervisor)
        self._gui = gui
        table = self.controller.supervisor.cluster_view
        inspection = self

        class ContextMenu(QObject):
            def eventFilter(self, obj, event):
                if event.type() != QEvent.ContextMenu:
                    return False
                row = table._row_for_proxy_index(table.table_view.indexAt(event.pos()))
                if row is None:
                    return False
                target = int(row['id'])
                menu = QMenu(table.table_view)
                menu.setToolTipsVisible(True)
                restore = menu.addAction('Restore parents in Merge Mode')
                action = menu.addAction('Inspect parents (read-only)')
                try:
                    groups = inspection.lineage.parents(target)
                    if any(not len(ids) for ids in groups.values()):
                        restore.setEnabled(False)
                        restore.setToolTip(
                            'Every parent needs retained evidence before restoration.'
                        )
                    if len(groups) > 16:
                        raise ValueError(
                            'Parent comparison supports at most 16 immediate parents.'
                        )
                except ValueError as error:
                    action.setEnabled(False)
                    action.setToolTip(str(error))
                    if 'groups' not in locals():
                        restore.setEnabled(False)
                        restore.setToolTip(str(error))
                restore.triggered.connect(lambda checked=False: inspection.restore_parents(target))
                action.triggered.connect(lambda checked=False: inspection._try_show(target))
                try:
                    menu.exec_(event.globalPos())
                finally:
                    menu.deleteLater()
                return True

        self._context_menu = ContextMenu(gui)
        table.table_view.viewport().installEventFilter(self._context_menu)
        connect(self._on_gui_closed, event='close', sender=gui)
        gui.view_actions.add(
            self.show_selected,
            name='show_parents',
            docstring='Inspect parents (read-only).',
        )
        gui.view_actions.add(
            self.restore_parents,
            name='restore_parents',
            docstring='Restore parents in Merge Mode.',
        )

    def restore_parents(self, target=None):
        """Restore recorded parents through the native Supervisor command."""
        self.close_review()
        try:
            return self.controller.supervisor.restore_parents(target)
        except ValueError as error:
            logger.warning('Cannot restore merge parents: %s', error)
            if self._gui is not None:
                self._gui.statusBar().showMessage(str(error), 10000)

    def _try_show(self, target):
        try:
            return self.show(target)
        except ValueError as error:
            logger.warning('Cannot inspect merge parents: %s', error)
            if self._gui is not None:
                self._gui.statusBar().showMessage(str(error), 10000)

    def show_selected(self):
        selected = self.controller.supervisor.selected
        if len(selected) != 1:
            raise ValueError('Select one merged unit to show its parents.')
        return self.show(selected[0])

    def show(self, target):
        try:
            return self._show(target)
        except Exception:
            self.close_review()
            if hasattr(self, '_temporary'):
                self._temporary.cleanup()
                del self._temporary
            self.providers = None
            raise

    def _show(self, target):
        from phy.cluster.views import AmplitudeView, CorrelogramView, FeatureView, WaveformView

        from ._parent_workspace import ParentWorkspace

        groups = self.lineage.parents(target)
        self.close_review()
        if self._gui is None:
            raise ValueError('Attach the main GUI before inspecting parents.')
        if self.controller.supervisor.selected != [target]:
            self.controller.supervisor.select([target])
            self.controller.supervisor._settle_selection()
        providers = ParentProviders(self.controller, target, groups)
        self.providers = providers
        self._temporary = tempfile.TemporaryDirectory(prefix='phy-parent-review-')
        colors = self.lineage.colors(target, groups)
        workspace = ParentWorkspace(
            self._gui,
            self.controller.supervisor,
            target,
            groups,
            self.lineage.full_counts(target, groups),
            colors,
            providers.bounded,
            Path(self._temporary.name),
            self.close_review,
        )
        self._workspace = workspace
        self.window = window = workspace.gui
        self.host_gui = self._gui
        self.return_button = workspace.return_button
        self.parent_table = workspace.parent_table
        ccg = CorrelogramView(
            correlograms=providers.correlograms, sample_rate=self.controller.model.sample_rate
        )
        ccg.ex_status = (
            'Read-only parents; selected windows only, raw pair counts.'
            if providers.bounded
            else 'Read-only parents; sampled pair counts.'
        )
        specs = [(ccg, providers.ids)]
        waveform_ids = [p for p in groups if len(providers.waveform_ids(p))]
        if waveform_ids and (
            getattr(self.controller.model, 'spike_waveforms', None) is not None
            or getattr(self.controller.model, 'traces', None) is not None
        ):
            specs.append(
                (
                    WaveformView(
                        waveforms={'waveforms': providers.waveforms},
                        sample_rate=self.controller.model.sample_rate,
                    ),
                    waveform_ids,
                )
            )
            if len(waveform_ids) < len(groups):
                workspace.scope.setText(
                    'Parents without retained waveforms are omitted from that plot.'
                )
        elif getattr(self.controller.model, 'spike_templates', None) is not None:
            specs.append(
                (
                    WaveformView(
                        waveforms={'templates': providers.templates},
                        sample_rate=self.controller.model.sample_rate,
                    ),
                    [p for p in groups if len(groups[p])],
                )
            )
            workspace.scope.setText(
                workspace.scope.text()
                + ' Raw/stored waveforms unavailable: showing up to 100 template traces per parent.'
            )
        if getattr(self.controller.model, 'amplitudes', None) is not None:
            specs.append(
                (
                    AmplitudeView(
                        amplitudes=providers.amplitudes,
                        duration=self.controller.model.duration,
                        split_is_eligible=lambda: False,
                    ),
                    providers.ids,
                )
            )
        if (
            getattr(self.controller.model, 'features', None) is not None
            or getattr(self.controller.model, 'spike_waveforms', None) is not None
        ):
            specs.append(
                (
                    FeatureView(
                        features=providers.features, attributes={'time': providers.feature_times}
                    ),
                    [p for p in groups if len(providers.feature_ids(p))],
                )
            )
        try:
            for view, ids in specs:
                # add_view, deliberately not attach: no global selection/split/
                # channel callbacks and no actions capable of reaching curation.
                window.add_view(view, position='right')
                view._cluster_color_index_by_id = colors
                # Canvas click handlers normally publish channel/time/transfer
                # events that ordinary views elsewhere in the application may
                # consume. Keep these display-only views local, including when
                # the user modifier-clicks; PanZoom remains independently active.
                view.on_mouse_click = lambda event: None
                self.views.append(view)
                if ids:
                    view.on_select(cluster_ids=ids)
            # Two columns with two rows leave the correlogram matrix usable.
            # The same arrangement gracefully handles unavailable data views.
            if len(self.views) > 1:
                window.splitDockWidget(self.views[0].dock, self.views[1].dock, Qt.Horizontal)
            if len(self.views) > 2:
                window.splitDockWidget(self.views[0].dock, self.views[2].dock, Qt.Vertical)
            if len(self.views) > 3:
                window.splitDockWidget(self.views[1].dock, self.views[3].dock, Qt.Vertical)
            if len(self.views) > 1:
                window.resizeDocks([v.dock for v in self.views[:2]], [610, 460], Qt.Horizontal)
            if len(self.views) > 2:
                window.resizeDocks(
                    [self.views[0].dock, self.views[2].dock], [430, 300], Qt.Vertical
                )
            if len(self.views) > 3:
                window.resizeDocks(
                    [self.views[1].dock, self.views[3].dock], [430, 300], Qt.Vertical
                )
            workspace.show()
            return window
        except Exception:
            self.close_review()
            raise

    def _dispose_review(self):
        for view in self.views:
            owned = (
                view,
                view.canvas,
                getattr(view, 'dock', None),
                getattr(view.canvas, 'panzoom', None),
                getattr(view.canvas, 'axes', None),
                *getattr(view.canvas, '_attached', ()),
            )
            # None is a meaningful sender in phylib's registry. Passing it
            # would disconnect unrelated application-wide callbacks.
            unconnect(*(obj for obj in owned if obj is not None))
            with suppress(RuntimeError):  # Qt may already have destroyed the owning window.
                view.canvas.close()
            view._closed = True
        self.views = []
        if self.window is not None:
            unconnect(self.window)
        self.window = None
        self.providers = None
        if hasattr(self, '_temporary'):
            self._temporary.cleanup()
            del self._temporary

    def close_review(self):
        if self.window is None or getattr(self, '_closing', False):
            return
        self._closing = True
        try:
            # Mark the workspace inactive before restoration can flush deferred selections.
            self._workspace.restore()
            self._dispose_review()
            self._workspace = None
        finally:
            self._closing = False

    def _on_gui_closed(self, sender):
        # ``close`` is a request: a later save prompt may cancel it. Restore the
        # workspace before geometry saving, retaining the live host attachment.
        self.close_review()

    def close(self):
        self.close_review()
        if self._gui is not None and hasattr(self, '_context_menu'):
            with suppress(RuntimeError):
                self.controller.supervisor.cluster_view.table_view.viewport().removeEventFilter(
                    self._context_menu
                )
        self._gui = None
        self.lineage.close()
        unconnect(self._on_selection, self._on_mutation, self._on_gui_closed)
