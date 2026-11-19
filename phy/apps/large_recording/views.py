"""Explicitly sampled statistics and source-amplitude evidence views."""

import numpy as np

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

    def __init__(self, *args, exclusion_correlograms=None, **kwargs):
        # An explicit provider preserves the original three-argument plugin API.
        self.exclusion_correlograms = exclusion_correlograms
        self.exclusion_ms = 0.3
        self.exclude_auto = False
        self.exclude_cross = False
        super().__init__(*args, **kwargs)
        attrs = ('exclusion_ms', 'exclude_auto', 'exclude_cross')
        self.state_attrs += attrs
        self.local_state_attrs += attrs

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
