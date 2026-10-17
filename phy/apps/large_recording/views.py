"""Explicitly sampled statistics and window-relative amplitude views."""

from phy.cluster.views import AmplitudeView, CorrelogramView, ISIView
from phy.gui.qt import QLabel, Qt


class SampledISIView(ISIView):
    ex_status = 'Selected windows only; intervals crossing window boundaries excluded.'


class SampledCorrelogramView(CorrelogramView):
    ex_status = 'Selected windows only; raw pair counts, no full-recording rate baseline.'


class EvidenceAmplitudeView(AmplitudeView):
    """Raw source amplitude in an explicitly rebased evidence window."""

    window_caption = 'Recording time unavailable'

    def set_window_context(self, index, start_seconds, end_seconds):
        self.window_caption = (
            f'Evidence window {index} | Recording time: {start_seconds:.3f}–{end_seconds:.3f} s'
        )
        if hasattr(self, 'window_label'):
            self.window_label.setText(self.window_caption)

    def attach(self, gui):
        super().attach(gui)
        layout = self.dock.widget().layout()
        self.window_label = QLabel(self.window_caption)
        self.window_label.setWordWrap(True)
        self.time_axis_label = QLabel('Seconds from window start')
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
