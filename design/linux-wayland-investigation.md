# Linux graphics compatibility investigation

Date: 2026-11-09. Status: diagnostics implemented; the original Chromebook
Wayland freeze has not been reproduced. Existing backend selection is unchanged.

## Evidence and the supported change

A phy 2.1.0 user on Debian 13 inside Chromebook Linux could initially display
plots, but subsequent cluster selection and zoom left them stale. Switching only
Qt to X11 produced blank plots with repeated PyOpenGL context lookup errors.
Explicitly matching Qt X11/GLX and PyOpenGL GLX restored normal application use.

The mismatched GLX/EGL context error was reproduced locally with a valid Qt
context. The matched case passed. This is regression-tested, and Linux CI now
also runs the selection-redraw and context-mismatch controls on real OpenGL.

The normal renderer is unchanged except for a once-per-process warning in the
existing exception paths for the specific missing-context error. The standalone
diagnostic adds a matched `xcb-glx` experiment. Selection captures are armed
before the action and wait for its frames without calling `canvas.update()`.
Both image and metrics modes detect deliberately suppressed redraw requests.

## Why there is no automatic platform override

Qt's `QT_XCB_GL_INTEGRATION=xcb_glx` is a preference: Qt can fall back to EGL.
Automatically forcing PyOpenGL to GLX from that request could break a currently
working fallback installation. There is also no evidence that all Chromebook
or Wayland installations exhibit the original freeze. Preserving working
environments therefore rules out either broad override.

References: [Qt 5.15 integration selection](https://github.com/qt/qtbase/blob/5.15/src/plugins/platforms/xcb/qxcbconnection.cpp),
[PyOpenGL 3.1.10 platform selection](https://github.com/mcfletch/pyopengl/blob/3.1.10/OpenGL/platform/__init__.py).

## Isolated reproduction

[The maintainer environment](../scripts/tests/wayland/README.md) runs ordinary
TemplateController instances with a public test dataset, default docks, normal
Qt event loop, timed table clicks, and a Correlogram View wheel event. Passive
observation records update requests, paints, frame swaps, exposure and transforms.
It performs no screenshot redraws, framebuffer reads, event pumping, or data
provider replacement. Error signals and swallowed drawing errors invalidate PASS.

The release installation matches all six Phy source hashes and all package versions in
the reported inventory. Python 3.12.13 is used instead of the unavailable
3.12.16 build. The compositor is Weston 14.0.2 on Debian 13 with llvmpipe and DPR
2. The final runner sets `LP_NATIVE_VECTOR_WIDTH=128` to match the original
llvmpipe vector width; the original environment used ChromeOS integration. The optional proxy is
ChromiumOS Sommelier at platform2 commit
`c4c5759d85283c6cd89ad249e3806a4a75824b35`, using its noop driver.

Final observations used harness revision 3, which preserves Phy's GL-before-Qt
import order and requires an eligible measured view in every phase:

| Environment | Dataset copy | phy 2.1.0 | Current checkout |
| --- | --- | --- | --- |
| Weston, scale 2, llvmpipe | Public template data with raw recording | PASS | PASS |
| Sommelier noop proxy over Weston | Same data with raw recording omitted | PASS | PASS |

Each run completed three cluster selections and a Correlogram zoom, with natural
paints in all exposed plots and no collected errors. Ten plot views were observed
with raw data, nine without. These are callback/paint results, not proof of
physical presentation or scientific pixel correctness. A separate deliberately
failed worker produced `FAIL` and exit status 1 despite successful paints; eight
unit cases also prevent incomplete or skipped observations from becoming PASS.

The full application/library suite passed (624 non-app tests and 160 application
tests), as did ten real-OpenGL diagnostic controls. The final standalone-script
test run passed 94 tests, with ten opt-in graphics tests skipped there and run
separately. Lint, formatting, published documentation checks and package build
passed. The unrelated untracked `site/` directory was preserved.

Separate synthetic probes intermittently hit an initial
`xdg_surface has never been configured` protocol error, or reported an unexposed
dock. Other attempts passed. These observations are retained as separate startup
or exposure limitations; they do not reproduce the reported first-frame-only
application freeze. Unexpected `wl_surface.enter` warnings also appeared through
Sommelier while ordinary interactions progressed, so the warning alone is not
a reliable automatic activation condition.

## Remaining boundary

This setup cannot reproduce ChromeOS Exo, the guest's virtgpu/virtwl buffer path,
physical input routing or the affected device's exact compositor version. The
user has a confirmed workaround and needs no further request at this stage.
A future automatic fix requires either the original freeze in a controlled
reproduction or positive evidence of a narrowly detectable failed state. No
claim that the original Wayland cause has been fixed is justified yet.
