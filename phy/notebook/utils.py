# -*- coding: utf-8 -*-

"""Notebook utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------


VISPY_BACKENDS = ['pyqt4', 'wx', 'ipynb_webgl']


def _enable_gui(shell, backend):
    """Enable IPython GUI event loop integration."""
    shell.run_line_magic('gui', backend)


def enable_notebook(backend=None):
    """Enable notebook integration with the given backend for VisPy."""
    # TODO: unit tests
    if backend not in (None,) + VISPY_BACKENDS:
        raise ValueError("'backend' must be one of: "
                         "{0:s}".format(', '.join(VISPY_BACKENDS)))
    # Import VisPy.
    try:
        from vispy import app
    except ImportError:
        raise ImportError("VisPy is required in the notebook.")
    # Import IPython.
    try:
        from IPython import get_ipython
    except ImportError:
        raise ImportError("IPython is required.")
    # Get the IPython shell.
    shell = get_ipython()
    # Default backend.
    if backend is None:
        # TODO: user-level parameter
        backend = 'pyqt4'
    # Enable the VisPy backend.
    app.use_app(backend)
    # Enable IPython event loop integration.
    if backend == 'pyqt4':
        _enable_gui(shell, 'qt')
    elif backend == 'wx':
        _enable_gui(shell, 'wx')
