import sys

from phy.gui.qt import QApplication
from phy.plot.base import BaseCanvas


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    canvas = BaseCanvas()
    print(type(canvas).__name__, flush=True)
    print("before show", flush=True)
    canvas.show()
    print("after show", flush=True)
    app.processEvents()
    print("after events", flush=True)


if __name__ == "__main__":
    main()
