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


def _wrap_html(fn=None, html=None, wrap='qt',
               static_path=None,
               css_replacements=None):
    # Read the styles.css file.
    # NOTE: this file is in the default static folder by default, or
    # in the specified static folder.
    # TODO: concatenate default and custom styles.css (right now the default
    # one is discarded, but it is currently empty anyway).
    css = _read('styles.css', static_path=static_path)
    if css_replacements:
        for (x, y) in css_replacements.items():
            css = css.replace(x, y)

    # Read and format the HTML code.
    html = html or _read(fn, static_path=static_path)

    # Read the wrap HTML.
    # NOTE: this file is from `./static/`, the default static folder.
    wrapped = _read('wrap_{}.html'.format(wrap))

    # Insert the CSS and HTML in the wrap.
    return wrapped.replace('%CSS%', css).replace('%HTML%', html)
