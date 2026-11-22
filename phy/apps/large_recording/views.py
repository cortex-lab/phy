"""Explicitly sampled statistics and source-amplitude evidence views."""

import numpy as np
from phylib.utils import connect, unconnect

from phy.cluster._proposition_annotations import project_proposition_labels
from phy.cluster.views import AmplitudeView, CorrelogramView, ISIView
from phy.gui.qt import QLabel, Qt


class SampledISIView(ISIView):
    ex_status = 'Selected windows only; intervals crossing window boundaries excluded.'


class SampledCorrelogramView(CorrelogramView):
    """Window-local pair counts with a conditional-uniform off-center chance line.

    The correlograms provider returns ``(counts, chance)`` computed from the same
    retained windows. The line is the mean off-center null expectation; the
    folded central bin has a different width/count convention.
    """

    ex_status = 'Selected windows only; chance line is the mean off-center window-local null.'

    default_shortcuts = {
        **CorrelogramView.default_shortcuts,
        'toggle_auto_exclusion': 'd',
        'toggle_cross_exclusion': 'shift+d',
    }

    def __init__(self, *args, exclusion_correlograms=None, proposition_context=None, **kwargs):
        # An explicit provider preserves the original three-argument plugin API.
        self.exclusion_correlograms = exclusion_correlograms
        self.exclusion_ms = 0.3
        self.exclude_auto = False
        self.exclude_cross = False
        self.proposition_context = proposition_context
        self._proposition_workspace = None
        self._proposition_color_keys = ()
        self._proposition_labels = project_proposition_labels(None, ())
        self._proposition_visible_keys = None
        self._proposition_labels_pending = False
        self._proposition_connections = False
        self.proposition_legend = None
        super().__init__(*args, **kwargs)
        attrs = ('exclusion_ms', 'exclude_auto', 'exclude_cross')
        self.state_attrs += attrs
        self.local_state_attrs += attrs

    def _refresh_proposition_annotations(self):
        catalog, active_keys, visible_keys = (
            self.proposition_context()
            if self.proposition_context is not None
            else (None, (), None)
        )
        workspace = tuple(active_keys)
        if workspace != self._proposition_workspace:
            self._proposition_color_keys = ()
            self._proposition_workspace = workspace
        self._proposition_labels = project_proposition_labels(
            catalog, self.cluster_ids, workspace, self._proposition_color_keys
        )
        self._proposition_color_keys = self._proposition_labels.color_keys
        self._proposition_visible_keys = None if visible_keys is None else set(visible_keys)
        self._update_proposition_legend()

    def _update_proposition_legend(self):
        legend = self.proposition_legend
        if legend is None:
            return
        groups = self._proposition_labels.groups
        items, details = (
            [],
            ['SLAY source membership; pending now, not a guarantee of future review.'],
        )
        for group in groups:
            rgb = ','.join(str(round(component * 255)) for component in group.color[:3])
            suffix = ' (active)' if group.active else ''
            if len(items) < 4:
                items.append(f'<span style="color:rgb({rgb})">{group.display_id}{suffix}</span>')
            hidden = (
                self._proposition_visible_keys is not None
                and group.key not in self._proposition_visible_keys
            )
            units = [
                str(unit)
                for unit, annotation in self._proposition_labels.annotations.items()
                if group in annotation.groups
            ]
            details.append(
                f'{group.display_id}: pending{suffix}'
                f'{"; hidden by proposition filter" if hidden else ""}; displayed IDs: '
                + ', '.join(units)
            )
        if len(groups) > 4:
            items.append(f'+{len(groups) - 4} groups (details)')
        overlapping = [
            annotation
            for annotation in self._proposition_labels.annotations.values()
            if len(annotation.groups) > 1
        ]
        if overlapping:
            items.append('* multiple groups')
            details.append(
                'Overlapping IDs use their first listed color (active groups first); * retains all groups.'
            )
            details.extend(
                f'{annotation.text}: ' + ', '.join(group.display_id for group in annotation.groups)
                for annotation in overlapping
            )
        legend.setText('SLAY IDs: ' + ' · '.join(items))
        details.append('White IDs have no current pending source-proposition membership.')
        legend.setToolTip('\n'.join(details))
        legend.setVisible(bool(groups))

    def _plot_labels(self):
        self._refresh_proposition_annotations()
        self._update_display_scale()
        n = len(self.cluster_ids)
        for k, cluster_id in enumerate(self.cluster_ids):
            annotation = self._proposition_labels.annotations[cluster_id]
            for pos, anchor, box_index in (
                ([-1, 0], [-1, 0], (k, 0)),
                ([0, -1], [0, -1], (n - 1, k)),
            ):
                self.text_visual.add_batch_data(
                    pos=pos,
                    text=annotation.text,
                    color=annotation.color,
                    anchor=anchor,
                    data_bounds=None,
                    box_index=box_index,
                )

    def _display_scale_for_size(self, width, height):
        sx, sy = super()._display_scale_for_size(width, height)
        annotations = self._proposition_labels.annotations
        max_chars = max((len(str(unit)) for unit in self.cluster_ids), default=1)
        label_chars = max((len(item.text) for item in annotations.values()), default=max_chars)
        glyph_width = self.text_visual._tex.shape[1] // 16 * self.text_visual.font_size / 12
        return max(0.5, sx - 2 * max(0, label_chars - max_chars) * glyph_width / max(width, 1)), sy

    def refresh_proposition_labels(self):
        """Refresh only label buffers after catalog/review changes, never pair counts."""
        if self._closed:
            return
        self._refresh_proposition_annotations()
        if not self.cluster_ids or not self._dock_visible:
            self._proposition_labels_pending = bool(self.cluster_ids)
            return
        self._proposition_labels_pending = False
        self.text_visual.reset_batch()
        self._plot_labels()
        self.canvas.update_visual(self.text_visual)
        self.canvas.update()

    def _on_proposition_visibility_changed(self, visible):
        if visible and self._proposition_labels_pending:
            self.refresh_proposition_labels()

    def _on_proposition_close(self, sender, gui):
        self._cleanup_proposition_labels()

    def _on_proposition_gui_closed(self, sender):
        self._cleanup_proposition_labels()

    def _cleanup_proposition_labels(self):
        if not self._proposition_connections:
            return
        self._proposition_connections = False
        self.dock.visibilityChanged.disconnect(self._on_proposition_visibility_changed)
        unconnect(self._on_proposition_close, self._on_proposition_gui_closed)

    @property
    def exclusion_ms(self):
        return self._exclusion_ms

    @exclusion_ms.setter
    def exclusion_ms(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError('Exclusion half-width must be finite and nonnegative.')
        self._exclusion_ms = value

    @property
    def status(self):
        status = super().status
        if self.exclusion_correlograms is not None:
            enabled = [
                name
                for name, value in (('auto', self.exclude_auto), ('cross', self.exclude_cross))
                if value
            ]
            if enabled:
                # Dock status text is elided in the middle. Keep every active
                # switch and the width in the visible prefix.
                status = f'EXCL {"+".join(enabled)} ±{self.exclusion_ms:g} ms | {status}'
        return status

    def _get_correlograms_data(self):
        if self.exclusion_correlograms is not None:
            return self.exclusion_correlograms(
                self.cluster_ids,
                self.bin_size,
                self.window_size,
                exclusion_ms=self.exclusion_ms,
                exclude_auto=self.exclude_auto,
                exclude_cross=self.exclude_cross,
            )
        return self.correlograms(self.cluster_ids, self.bin_size, self.window_size)

    def _update_exclusion(self):
        if hasattr(self, 'actions'):
            for name, checked in (
                ('toggle_auto_exclusion', self.exclude_auto),
                ('toggle_cross_exclusion', self.exclude_cross),
            ):
                action = self.actions.get(name)
                if action is not None:
                    action.setChecked(checked)
        self.update_status()
        if self.cluster_ids:
            self.plot()

    def set_exclusion_zone(self, value):
        """Set the near-zero pair exclusion half-width in milliseconds.

        Zero excludes exact-time pairs when the corresponding switch is enabled.
        Only displayed pair counts change; spike trains and assignments are retained.
        """
        self.exclusion_ms = value
        self._update_exclusion()

    def toggle_auto_exclusion(self, checked):
        """Exclude near-zero pairs from sampled autocorrelograms."""
        self.exclude_auto = bool(checked)
        self._update_exclusion()

    def toggle_cross_exclusion(self, checked):
        """Exclude near-zero pairs from sampled cross-correlograms."""
        self.exclude_cross = bool(checked)
        self._update_exclusion()

    def attach(self, gui):
        super().attach(gui)
        if self.proposition_context is not None:
            self.proposition_legend = QLabel()
            self.proposition_legend.setWordWrap(True)
            self.proposition_legend.setTextFormat(Qt.RichText)
            self.proposition_legend.setStyleSheet('color: white; background: black; padding: 2px;')
            self.dock.widget().layout().addWidget(self.proposition_legend)
            self.dock.visibilityChanged.connect(self._on_proposition_visibility_changed)
            connect(self._on_proposition_close, event='close_view', sender=self)
            connect(self._on_proposition_gui_closed, event='closed', sender=gui)
            self._proposition_connections = True
            self._refresh_proposition_annotations()
        if self.exclusion_correlograms is None:
            return
        self.actions.separator()
        self.actions.add(
            self.set_exclusion_zone,
            prompt=True,
            prompt_default=lambda: self.exclusion_ms,
        )
        self.actions.add(self.toggle_auto_exclusion, checkable=True, checked=self.exclude_auto)
        self.actions.add(self.toggle_cross_exclusion, checkable=True, checked=self.exclude_cross)
        # Other view shortcuts are window-wide. Scope these two to this dock so
        # typing in a table/filter cannot silently change the displayed counts.
        self.dock._widget.setFocusPolicy(Qt.StrongFocus)
        for name in ('toggle_auto_exclusion', 'toggle_cross_exclusion'):
            action = self.actions.get(name)
            gui.removeAction(action)
            self.dock.addAction(action)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.update_status()


class EvidenceAmplitudeView(AmplitudeView):
    """Sampled source amplitudes with a full-recording clock or local window."""

    def __init__(self, *args, **kwargs):
        self._amplitude_scope = 'overview'
        super().__init__(*args, **kwargs)
        self.state_attrs += ('amplitude_scope',)

    @property
    def amplitude_scope(self):
        return self._amplitude_scope

    @amplitude_scope.setter
    def amplitude_scope(self, scope):
        if scope not in ('overview', 'window'):
            raise ValueError('Amplitude scope must be overview or window.')
        self.clear_amplitude_split_threshold()
        self._amplitude_scope = scope
        if hasattr(self, '_controller'):
            self._controller._update_amplitude_scope(self)

    def toggle_amplitude_scope(self):
        """Toggle the full recording overview and current evidence window."""
        self.amplitude_scope = 'window' if self.amplitude_scope == 'overview' else 'overview'

    def on_request_split(self, sender=None):
        # Bounded commits consume a semantic predicate, never sampled row IDs.
        return np.array([], dtype=np.int64)

    window_caption = 'Recording time unavailable'

    def set_window_context(self, index, start_seconds, end_seconds):
        self.window_caption = (
            f'Evidence window {index} | Recording time: {start_seconds:.3f}–{end_seconds:.3f} s'
        )
        if hasattr(self, 'window_label'):
            self.window_label.setText(self.window_caption)

    def attach(self, gui):
        super().attach(gui)
        self.actions.add(self.toggle_amplitude_scope, show_shortcut=False)
        layout = self.dock.widget().layout()
        self.window_label = QLabel(self.window_caption)
        self.window_label.setWordWrap(True)
        self.time_axis_label = QLabel(
            'Recording time (seconds)'
            if self.amplitude_scope == 'overview'
            else 'Seconds from window start'
        )
        for label in (self.window_label, self.time_axis_label):
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet('color: white; background-color: black; padding: 4px;')
        layout.insertWidget(0, self.window_label)
        layout.insertWidget(layout.count() - 1, self.time_axis_label)

    def _get_data_bounds(self, bunchs):
        if not any(len(bunch.amplitudes) for bunch in bunchs):
            return (0, 0, self.duration, 1)
        xmin, ymin, xmax, ymax = super()._get_data_bounds(bunchs)
        if ymin == ymax:
            ymin, ymax = ymin - 0.5, ymax + 0.5
        return xmin, ymin, xmax, ymax
