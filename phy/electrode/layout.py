# -*- coding: utf-8 -*-

"""Probe layout."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

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


def _disk(x, y, r, c):
    return ('<circle cx="%.5f%%" cy="%.5f%%" '
            'r="%d" fill="%s" />') % (x, y, r, c)


def _rgba(rgb, a=1.):
    r, g, b = rgb
    return 'rgba(%d, %d, %d, %.3f)' % (r, g, b, a)


def _iter_disks(positions, cluster_channels=None):
    cluster_channels = cluster_channels or {}
    channel_ids = set(_flatten(cluster_channels.values()))
    channel_color_index = {ch: cl
                           for cl, channels in cluster_channels.items()
                           for ch in channels
                           }
    for i, (x, y) in enumerate(_iter_channel(positions)):
        r = 5 if i not in channel_ids else 7
        cl = channel_color_index.get(i, -1)
        c = _rgba(_COLORMAP[cl % len(_COLORMAP)]) if cl >= 0 else '#777'
        yield _disk(x, y, r, c)


def probe_layout(positions, cluster_channels):
    contents = '\n'.join(_iter_disks(positions, cluster_channels))
    return """
    <svg style="background: black; width:100%; height:100%;">
      {}
    </svg>
    """.format(contents)
