# -*- coding: utf-8 -*-

"""Spike detection."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext.six import string_types


#------------------------------------------------------------------------------
# Thresholder
#------------------------------------------------------------------------------

class Thresholder(object):
    """Threshold traces to detect spikes.

    Parameters
    ----------

    mode : str
        `'positive'`, `'negative'`, or `'both'`.
    thresholds : dict
        A `{str: float}` mapping for multiple thresholds (e.g. `weak`
        and `strong`).

    """
    def __init__(self,
                 mode=None,
                 thresholds=None,
                 ):
        assert mode in ('positive', 'negative', 'both')
        if isinstance(thresholds, (float, int)):
            thresholds = {'default': thresholds}
        assert isinstance(thresholds, dict)
        self._mode = mode
        self._thresholds = thresholds

    def _transform(self, data):
        if self._mode == 'positive':
            return data
        elif self._mode == 'negative':
            return -data
        elif self._mode == 'both':
            return np.abs(data)

    def __call__(self, data, threshold=None):
        # Use the only threshold by default (if there is only one).
        if threshold is None:
            assert len(self._thresholds) == 1
            threshold = list(self._thresholds.keys())[0]
        # Fetch the threshold from its name.
        if isinstance(threshold, string_types):
            assert threshold in self._thresholds
            threshold = self._thresholds[threshold]
        threshold = float(threshold)
        # Transform the data according to the mode.
        data_t = self._transform(data)
        # Threshold the data.
        return data_t > threshold


class FloodFillDetector(object):
    pass
