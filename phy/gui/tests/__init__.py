def show_and_wait(qtbot, widget):
    with qtbot.waitExposed(widget):
        widget.show()
