# -*- coding: utf-8 -*-

"""KlustaViewa CLI tool.

Usage:

    klustaviewa /path/to/myfile.kwik [-i]

Options:

* `-i`: launch the GUI in interactive mode with an IPython terminal.
  This gives you access to the underlying Python API for programmatic access.

Once within IPython, you have access to the following variables:

* `kwik_path`: the path to the Kwik file
* `session`: the `Session` instance
* `model`: the `KwikModel` instance
* `kv`: the `KlustaViewa` instance

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

def main():
    print("KlustaViewa {}".format(phy.__version__))
    # TODO: use argparse

    if '-h' in sys.argv or '--help' in sys.argv:
        print(sys.modules[__name__].__doc__)
        return

    if len(sys.argv) == 1:
        print("Please specify a path to a `.kwik` file.")
        exit(1)

    kwik_path = sys.argv[1]
    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        exit(1)

    print("\nLoading {}...".format(kwik_path))
    session = Session(kwik_path)
    print("Data successfully loaded!\n")
    session.model.describe()

    start_qt_app()
    kv = session.show_gui(show=False)

    print("\nPress `ctrl+h` to see the list of keyboard shortcuts.\n")

    # Interactive mode with IPython.
    if '-i' in sys.argv:
        print("\nStarting IPython...")
        from IPython import start_ipython

        # Namespace.
        ns = {'phy': phy,
              'session': session,
              'model': session.model,
              'kwik_path': kwik_path,
              'kv': kv,
              }
        start_ipython(["--gui=qt", "-i", "-c='kv.show()'"], user_ns=ns)
    else:
        kv.show()
        run_qt_app()


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
