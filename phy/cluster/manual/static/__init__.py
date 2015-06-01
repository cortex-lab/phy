# -*- coding: utf-8 -*-

"""HTML/CSS utilities."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os.path as op


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _path(fn):
    curdir = op.dirname(op.realpath(__file__))
    return op.join(curdir, fn)


def _read(fn):
    with open(_path(fn), 'r') as f:
        return f.read()


def _get_html(fn=None, html=None, wrap='qt', **params):
    # Read the styles.css file.
    css = _read('styles.css')

    # Read and format the HTML code.
    html = html or _read(fn)
    html = html.format(**params)

    # Read the wrap HTML.
    wrapped = _read('wrap_{}.html'.format(wrap))

    # Insert the CSS and HTML in the wrap.
    return wrapped.replace('%CSS%', css).replace('%HTML%', html)
