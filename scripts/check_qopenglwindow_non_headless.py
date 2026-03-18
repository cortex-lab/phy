import sys

from PyQt5.QtGui import QOpenGLWindow
from PyQt5.QtWidgets import QApplication


class Window(QOpenGLWindow):
    def initializeGL(self):
        pass

    def paintGL(self):
        pass


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = Window()
    print("before show", flush=True)
    window.show()
    print("after show", flush=True)
    app.processEvents()
    print("after events", flush=True)


if __name__ == "__main__":
    main()
