"""Failure reporting for Linux graphics context lookup errors."""

import logging
import sys

logger = logging.getLogger(__name__)
_context_warning_shown = False


def _warn_context_failure(error):
    """Explain the known context-lookup failure once, without retrying rendering."""
    global _context_warning_shown
    if (
        _context_warning_shown
        or not sys.platform.startswith('linux')
        or 'Attempt to retrieve context when no valid context' not in str(error)
    ):
        return
    _context_warning_shown = True
    logger.warning(
        'OpenGL could not access the drawing context. Plots may be blank or stale. '
        'Qt and PyOpenGL may be using different graphics backends. '
        'A compatibility launch to test after closing phy is: '
        'QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_glx PYOPENGL_PLATFORM=glx '
        'phy template-gui params.py --debug '
        '(use your dataset params.py). See docs/troubleshooting.md. '
        'This is a candidate workaround, not a confirmed diagnosis.'
    )
