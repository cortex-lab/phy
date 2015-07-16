# -*- coding: utf-8 -*-

"""Default settings for the session."""


# -----------------------------------------------------------------------------
# Session settings
# -----------------------------------------------------------------------------

def on_open(session):
    """You can update the session when a model is opened.

    For example, you can register custom statistics with
    `session.register_statistic`.

    """
    pass


def on_gui_open(session, gui):
    """You can customize a GUI when it is open."""
    pass


def on_view_open(gui, view):
    """You can customize a view when it is open."""
    pass


# -----------------------------------------------------------------------------
# Misc settings
# -----------------------------------------------------------------------------

# Logging level in the log file.
log_file_level = 'debug'
