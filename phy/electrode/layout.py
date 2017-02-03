# -*- coding: utf-8 -*-

"""Probe layout."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict

import numpy as np
from phy.io.array import _flatten
from phy.plot.transform import Range, NDC
from phy.plot.utils import _get_boxes
from phy.utils._color import _COLORMAP


#------------------------------------------------------------------------------
# Layout
#------------------------------------------------------------------------------

def _iter_channel(positions):
    size = 100
    margin = 5
    boxes = _get_boxes(positions, keep_aspect_ratio=False)
    xmin, ymin = boxes[:, :2].min(axis=0)
    xmax, ymax = boxes[:, 2:].max(axis=0)
    x = boxes[:, [0, 2]].mean(axis=1)
    y = - boxes[:, [1, 3]].mean(axis=1)
    positions = np.c_[x, y]
    tr = [margin, margin, size - margin, size - margin]
    positions = Range(NDC, tr).apply(positions)
    for x, y in positions:
        yield x, y


def _disk(x, y, r, c, t=0):
    return ('<circle cx="%.5f%%" cy="%.5f%%" '
            'r="%d" fill="%s" '
            'transform="translate(%d, 0)"'
            ' />') % (x, y, r, c, t)


def _rgba(rgb, a=1.):
    r, g, b = rgb
    return 'rgba(%d, %d, %d, %.3f)' % (r, g, b, a)


def _iter_disks(positions, cluster_channels=None):
    """

    positions: Nx2 array
    cluster_channels: {cluster: channels}

    """

    color_masked = _rgba((128,) * 3, 1.)
    color_umasked = lambda clu: _rgba(_COLORMAP[clu % len(_COLORMAP)], 1)

    size_masked = 5
    size_unmasked = 7

    # n_channels = positions.shape[0]
    n_clusters = len(cluster_channels)

    cluster_channels = cluster_channels or {}
    channel_ids = set(_flatten(cluster_channels.values()))
    is_masked = {ch: False for ch in channel_ids}

    # List of clusters per channel.
    clusters_per_channel = defaultdict(lambda: [])
    for clu, channels in cluster_channels.items():
        for channel in channels:
            clusters_per_channel[channel].append(clu)

    # Enumerate the discs for each channel.
    for channel_id, (x, y) in enumerate(_iter_channel(positions)):
        masked = is_masked.get(channel_id, True)
        if masked:
            yield _disk(x, y, size_masked, color_masked)
            continue
        for clu in clusters_per_channel[channel_id]:
            # Translation.
            t = 10 * (clu - .5 * (n_clusters - 1))
            yield _disk(x, y, size_unmasked, color_umasked(clu), t=t)


def probe_layout(positions, cluster_channels):
    contents = '\n'.join(_iter_disks(positions, cluster_channels))
    return """
    <svg style="background: black; width:100%; height:100%;">
      {}
    </svg>
    """.format(contents)
