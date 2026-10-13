"""Explicitly sampled statistics and window-relative amplitude views."""

from phy.cluster.views import AmplitudeView, CorrelogramView, ISIView


class SampledISIView(ISIView):
    ex_status = 'Selected windows only; intervals crossing window boundaries excluded.'


class SampledCorrelogramView(CorrelogramView):
    ex_status = 'Selected windows only; raw pair counts, no full-recording rate baseline.'


class EvidenceAmplitudeView(AmplitudeView):
    """Raw source amplitude in an explicitly rebased evidence window."""

    def _get_data_bounds(self, bunchs):
        if not any(len(bunch.amplitudes) for bunch in bunchs):
            return (0, 0, self.duration, 1)
        xmin, ymin, xmax, ymax = super()._get_data_bounds(bunchs)
        if ymin == ymax:
            ymin, ymax = ymin - 0.5, ymax + 0.5
        return xmin, ymin, xmax, ymax
