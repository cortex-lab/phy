# -*- coding: utf-8 -*-

"""ClusterManualGUI CLI tool.

Usage:

    phy cluster manual /path/to/myfile.kwik [-i]

Options:

* `-i`: launch the GUI in interactive mode with an IPython terminal.
  This gives you access to the underlying Python API for programmatic access.

Once within IPython, you have access to the following variables:

* `kwik_path`: the path to the Kwik file
* `session`: the `Session` instance
* `model`: the `KwikModel` instance
* `gui`: the `ClusterManualGUI` instance

Once the GUI is closed, quit IPython with `exit()`.

"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import os.path as op

import phy
from phy.cluster.manual import Session
from phy.utils import start_qt_app, run_qt_app


#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

def _pop(l, el, default=None):
    if el in l:
        l.remove(el)
        return el
    else:
        return default


def main():
    # TODO: use argparse
    args = sys.argv

    if '-h' in args or '--help' in args:
        print(sys.modules[__name__].__doc__)
        return 0

    if '-v' in args or '--version' in args:
        print("phy v{}".format(phy.__version__))
        return 0

    if len(args) <= 3 or args[1] != 'cluster' or args[2] != 'manual':
        print("Only the `phy cluster manual [-i] myfile.kwik` command "
              "is currently supported.")
        return 1

    args = args[3:]

    print("ClusterManualGUI")

    interactive = _pop(args, '-i', False) or _pop(args, '--interactive', False)

    if len(args) == 0:
        print("Please specify a path to a `.kwik` file.")
        return 1

    kwik_path = args[0]
    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return 1

    print("\nLoading {}...".format(kwik_path))
    session = Session(kwik_path)
    print("Data successfully loaded!\n")
    session.model.describe()

    start_qt_app()
    gui = session.show_gui(show=False)

    print("\nPress `ctrl+h` to see the list of keyboard shortcuts.\n")

    # Interactive mode with IPython.
    if interactive:
        print("\nStarting IPython...")
        from IPython import start_ipython

        # Namespace.
        ns = {'phy': phy,
              'session': session,
              'model': session.model,
              'kwik_path': kwik_path,
              'gui': gui,
              }
        start_ipython(["--gui=qt", "-i", "-c='gui.show()'"], user_ns=ns)
    else:
        gui.show()
        run_qt_app()


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    exit(main())
