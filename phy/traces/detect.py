# -*- coding: utf-8 -*-

"""Spike detection."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils.array import _as_array
from ..ext.six import string_types
from ..ext.six.moves import range, zip


#------------------------------------------------------------------------------
# Thresholder
#------------------------------------------------------------------------------

def compute_threshold(arr, single_threshold=True, std_factor=None):
    """Compute the threshold(s) of filtered traces.

    Parameters
    ----------

    arr : ndarray
        Filtered traces, shape `(n_samples, n_channels)`.
    single_threshold : bool
        Whether there should be a unique threshold for all channels, or
        one threshold per channel.
    std_factor : float or 2-tuple
        The threshold in unit of signal std. Two values can be specified
        for multiple thresholds (weak and strong).

    Returns
    -------

    thresholds : ndarray
        A `(2,)` or `(2, n_channels)` array with the thresholds.

    """
    assert arr.ndim == 2
    ns, nc = arr.shape

    assert std_factor is not None
    if isinstance(std_factor, (int, float)):
        std_factor = (std_factor, std_factor)
    assert isinstance(std_factor, (tuple, list))
    assert len(std_factor) == 2
    std_factor = np.array(std_factor)

    if not single_threshold:
        std_factor = std_factor[:, None]

    # Get the median of all samples in all excerpts, on all channels.
    if single_threshold:
        median = np.median(np.abs(arr))
    # Or independently for each channel.
    else:
        median = np.median(np.abs(arr), axis=0)

    # Compute the threshold from the median.
    std = median / .6745
    threshold = std_factor * std
    assert isinstance(threshold, np.ndarray)

    if single_threshold:
        assert threshold.ndim == 1
        assert len(threshold) == 2
    else:
        assert threshold.ndim == 2
        assert threshold.shape == (2, nc)
    return threshold


class Thresholder(object):
    """Threshold traces to detect spikes.

    Parameters
    ----------

    mode : str
        `'positive'`, `'negative'`, or `'both'`.
    thresholds : dict
        A `{str: float}` mapping for multiple thresholds (e.g. `weak`
        and `strong`).

    Example
    -------

    ```python
    thres = Thresholder('positive', thresholds=(.1, .2))
    crossings = thres(traces)
    ```

    """
    def __init__(self,
                 mode=None,
                 thresholds=None,
                 ):
        assert mode in ('positive', 'negative', 'both')
        if isinstance(thresholds, (float, int, np.ndarray)):
            thresholds = {'default': thresholds}
        if thresholds is None:
            thresholds = {}
        assert isinstance(thresholds, dict)
        self._mode = mode
        self._thresholds = thresholds

    def transform(self, data):
        """Return `data`, `-data`, or `abs(data)` depending on the mode."""
        if self._mode == 'positive':
            return data
        elif self._mode == 'negative':
            return -data
        elif self._mode == 'both':
            return np.abs(data)

    def detect(self, data_t, threshold=None):
        """Perform the thresholding operation."""
        # Accept dictionary of thresholds.
        if isinstance(threshold, (list, tuple)):
            return {name: self(data_t, threshold=name)
                    for name in threshold}
        # Use the only threshold by default (if there is only one).
        if threshold is None:
            assert len(self._thresholds) == 1
            threshold = list(self._thresholds.keys())[0]
        # Fetch the threshold from its name.
        if isinstance(threshold, string_types):
            assert threshold in self._thresholds
            threshold = self._thresholds[threshold]
        # threshold = float(threshold)
        # Threshold the data.
        return data_t > threshold

    def __call__(self, data, threshold=None):
        # Transform the data according to the mode.
        data_t = self.transform(data)
        return self.detect(data_t, threshold=threshold)


# -----------------------------------------------------------------------------
# Connected components
# -----------------------------------------------------------------------------

def _to_tuples(x):
    return ((i, j) for (i, j) in x)


def _to_list(x):
    return [(i, j) for (i, j) in x]


def connected_components(weak_crossings=None,
                         strong_crossings=None,
                         probe_adjacency_list=None,
                         join_size=None):
    """Find all connected components in binary arrays of threshold crossings.

    Parameters
    ----------

    weak_crossings : array
        `(n_samples, n_channels)` array with weak threshold crossings
    strong_crossings : array
        `(n_samples, n_channels)` array with strong threshold crossings
    probe_adjacency_list : dict
        A dict `{channel: [neighbors]}`
    join_size : int
        The number of samples defining the tolerance in time for
        finding connected components

    Returns
    -------

    A list of lists of pairs `(samp, chan)` of the connected components in
    the 2D array `weak_crossings`, where a pair is adjacent if the samples are
    within `join_size` of each other, and the channels are adjacent in
    `probe_adjacency_list`, the channel graph.

    Note
    ----

    The channel mapping assumes that column #i in the data array is channel #i
    in the probe adjacency graph.

    """

    if probe_adjacency_list is None:
        probe_adjacency_list = {}

    # Make sure the values are sets.
    probe_adjacency_list = {c: set(cs)
                            for c, cs in probe_adjacency_list.items()}

    if strong_crossings is None:
        strong_crossings = weak_crossings

    assert weak_crossings.shape == strong_crossings.shape

    # Set of connected component labels which contain at least one strong
    # node.
    strong_nodes = set()

    n_s, n_ch = weak_crossings.shape
    join_size = int(join_size or 0)

    # An array with the component label for each node in the array
    label_buffer = np.zeros((n_s, n_ch), dtype=np.int32)

    # Component indices, a dictionary with keys the label of the component
    # and values a list of pairs (sample, channel) belonging to that component
    comp_inds = {}

    # mgraph is the channel graph, but with edge node connected to itself
    # because we want to include ourself in the adjacency. Each key of the
    # channel graph (a dictionary) is a node, and the value is a set of nodes
    # which are connected to it by an edge
    mgraph = {}
    for source, targets in probe_adjacency_list.items():
        # we add self connections
        mgraph[source] = targets.union([source])

    # Label of the next component
    c_label = 1

    # For all pairs sample, channel which are nonzero (note that numpy .nonzero
    # returns (all_i_s, all_i_ch), a pair of lists whose values at the
    # corresponding place are the sample, channel pair which is nonzero. The
    # lists are also returned in sorted order, so that i_s is always increasing
    # and i_ch is always increasing for a given value of i_s. izip is an
    # iterator version of the Python zip function, i.e. does the same as zip
    # but quicker. zip(A,B) is a list of all pairs (a,b) with a in A and b in B
    # in order (i.e. (A[0], B[0]), (A[1], B[1]), .... In conclusion, the next
    # line loops through all the samples i_s, and for each sample it loops
    # through all the channels.
    for i_s, i_ch in zip(*weak_crossings.nonzero()):
        # The next two lines iterate through all the neighbours of i_s, i_ch
        # in the graph defined by graph in the case of edges, and
        # j_s from i_s-join_size to i_s.
        for j_s in range(i_s - join_size, i_s + 1):
            # Allow us to leave out a channel from the graph to exclude bad
            # channels
            if i_ch not in mgraph:
                continue
            for j_ch in mgraph[i_ch]:
                # Label of the adjacent element.
                adjlabel = label_buffer[j_s, j_ch]
                # If the adjacent element is nonzero we need to do something.
                if adjlabel:
                    curlabel = label_buffer[i_s, i_ch]
                    if curlabel == 0:
                        # If current element is still zero, we just assign
                        # the label of the adjacent element to the current one.
                        label_buffer[i_s, i_ch] = adjlabel
                        # And add it to the list for the labelled component.
                        comp_inds[adjlabel].append((i_s, i_ch))

                    elif curlabel != adjlabel:
                        # If the current element is unequal to the adjacent
                        # one, we merge them by reassigning the elements of the
                        # adjacent component to the current one.
                        # samps_chans is an array of pairs sample, channel
                        # currently assigned to component adjlabel.
                        samps_chans = np.array(comp_inds[adjlabel],
                                               dtype=np.int32)

                        # samps_chans[:, 0] is the sample indices, so this
                        # gives only the samp,chan pairs that are within
                        # join_size of the current point.
                        # TODO: is this the right behaviour? If a component can
                        # have a width bigger than join_size I think it isn't!
                        samps_chans = samps_chans[i_s - samps_chans[:, 0] <=
                                                  join_size]

                        # Relabel the adjacent samp,chan points with current
                        # label.
                        samps, chans = samps_chans[:, 0], samps_chans[:, 1]
                        label_buffer[samps, chans] = curlabel

                        # Add them to the current label list, and remove the
                        # adjacent component entirely.
                        comp_inds[curlabel].extend(comp_inds.pop(adjlabel))

                        # Did not deal with merge condition, now fixed it
                        # seems...
                        # WARNING: might this "in" incur a performance hit
                        # here...?
                        if adjlabel in strong_nodes:
                            strong_nodes.add(curlabel)
                            strong_nodes.remove(adjlabel)

                    # NEW: add the current component label to the set of all
                    # strong nodes, if the current node is strong.
                    if curlabel > 0 and strong_crossings[i_s, i_ch]:
                        strong_nodes.add(curlabel)

        if label_buffer[i_s, i_ch] == 0:
            # If nothing is adjacent, we have the beginnings of a new
            # component, # so we label it, create a new list for the new
            # component which is given label c_label,
            # then increase c_label for the next new component afterwards.
            label_buffer[i_s, i_ch] = c_label
            comp_inds[c_label] = [(i_s, i_ch)]
            if strong_crossings[i_s, i_ch]:
                strong_nodes.add(c_label)
            c_label += 1

    # Only return the values, because we don't actually need the labels.
    comps = [comp_inds[key] for key in comp_inds.keys() if key in strong_nodes]
    return comps


class FloodFillDetector(object):
    """Detect spikes in weak and strong threshold crossings.

    Parameters
    ----------

    probe_adjacency_list : dict
        A dict `{channel: [neighbors]}`.
    join_size : int
        The number of samples defining the tolerance in time for
        finding connected components

    Example
    -------

    ```python
    det = FloodFillDetector(probe_adjacency_list=...,
                            join_size=...)
    components = det(weak_crossings, strong_crossings)
    ```

    `components` is a list of `(n, 2)` int arrays with the sample and channel
    for every sample in the component.

    """
    def __init__(self, probe_adjacency_list=None, join_size=None):
        self._adjacency_list = probe_adjacency_list
        self._join_size = join_size

    def __call__(self, weak_crossings=None, strong_crossings=None):
        weak_crossings = _as_array(weak_crossings, np.bool)
        strong_crossings = _as_array(strong_crossings, np.bool)

        cc = connected_components(weak_crossings=weak_crossings,
                                  strong_crossings=strong_crossings,
                                  probe_adjacency_list=self._adjacency_list,
                                  join_size=self._join_size,
                                  )
        # cc is a list of list of pairs (sample, channel)
        return [np.array(c) for c in cc]
