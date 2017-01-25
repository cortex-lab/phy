# -*- coding: utf-8 -*-

"""Probe layout."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from phy.plot.transform import Range, NDC
from phy.plot.utils import _get_boxes


#------------------------------------------------------------------------------
# Layout
#------------------------------------------------------------------------------

def _iter_channel(positions):
    size = 100
    margin = 10
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


def _iter_disks(positions, channel_ids=None):
    channel_ids = channel_ids if channel_ids is not None else []
    for i, (x, y) in enumerate(_iter_channel(positions)):
        r = 5 if i not in channel_ids else 7
        c = '#777' if i not in channel_ids else '#fff'
        yield _disk(x, y, r, c)


def probe_layout(positions, channel_ids):
    contents = '\n'.join(_iter_disks(positions, channel_ids))
    return """
    <svg style="background: black; width:100%; height:100%;">
      {}
    </svg>
    """.format(contents)
