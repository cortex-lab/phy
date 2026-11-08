"""Backend compatibility is opt-in and must not alter working environments."""

import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from phy import _graphics
from phy.plot.base import BaseCanvas, gl, gloo


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux GLX/EGL selection')
@pytest.mark.parametrize(
    'settings,expected',
    [
        ({'XDG_SESSION_TYPE': 'wayland'}, 'EGLPlatform'),
        ({'XDG_SESSION_TYPE': 'x11'}, 'GLXPlatform'),
        (
            {
                'XDG_SESSION_TYPE': 'wayland',
                'QT_QPA_PLATFORM': 'xcb',
                'QT_XCB_GL_INTEGRATION': 'xcb_glx',
            },
            'EGLPlatform',
        ),
        (
            {
                'XDG_SESSION_TYPE': 'wayland',
                'PYOPENGL_PLATFORM': 'glx',
                'QT_QPA_PLATFORM': 'xcb',
                'QT_XCB_GL_INTEGRATION': 'xcb_glx',
            },
            'GLXPlatform',
        ),
        (
            {
                'XDG_SESSION_TYPE': 'wayland',
                'PYOPENGL_PLATFORM': 'egl',
                'QT_QPA_PLATFORM': 'xcb',
                'QT_XCB_GL_INTEGRATION': 'xcb_egl',
            },
            'EGLPlatform',
        ),
        ({'XDG_SESSION_TYPE': 'wayland', 'QT_QPA_PLATFORM': 'offscreen'}, 'EGLPlatform'),
    ],
)
def test_gui_import_preserves_backend_selection(settings, expected):
    env = os.environ.copy()
    keys = (
        'XDG_SESSION_TYPE',
        'WAYLAND_DISPLAY',
        'QT_QPA_PLATFORM',
        'QT_XCB_GL_INTEGRATION',
        'PYOPENGL_PLATFORM',
    )
    for key in keys:
        env.pop(key, None)
    env.update(settings)
    code = f"""
import os
keys = {keys!r}
before = {{k: os.environ.get(k) for k in keys}}
import phy.gui.qt
from OpenGL import platform
assert before == {{k: os.environ.get(k) for k in keys}}
print(type(platform.PLATFORM).__name__)
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert result.stdout.strip() == expected


def test_context_failure_is_explained_once_across_frames(monkeypatch, caplog, qapp):
    monkeypatch.setattr(_graphics, 'sys', SimpleNamespace(platform='linux'))
    monkeypatch.setattr(_graphics, '_context_warning_shown', False)

    def fail(*args, **kwargs):
        raise RuntimeError('Attempt to retrieve context when no valid context')

    monkeypatch.setattr(gl, 'enable_depth_mask', fail)
    monkeypatch.setattr(gloo, 'clear', fail)
    canvas = BaseCanvas()
    try:
        BaseCanvas.initializeGL(canvas)
        BaseCanvas.paintGL(canvas)
        BaseCanvas.paintGL(canvas)
    finally:
        canvas.close()
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'PYOPENGL_PLATFORM=glx' in warnings[0]
    assert 'candidate workaround' in warnings[0]


def test_unrelated_render_error_does_not_suggest_glx(monkeypatch, caplog):
    monkeypatch.setattr(_graphics, '_context_warning_shown', False)
    _graphics._warn_context_failure(RuntimeError('Shader compilation failed'))
    assert not caplog.records
