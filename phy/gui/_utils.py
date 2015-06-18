# -*- coding: utf-8 -*-

"""HTML/CSS utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os.path as op


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _read(fn, static_path=None):
    """Read a file in a static directory.

    By default, this is `./static/`."""
    if static_path is None:
        static_path = op.join(op.dirname(op.realpath(__file__)), 'static')
    with open(op.join(static_path, fn), 'r') as f:
        return f.read()
