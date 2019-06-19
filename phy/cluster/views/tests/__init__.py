import os


def _stop_and_close(qtbot, v):
    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    v.close()
