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

import phy
from phy.cluster.manual import Session
from phy.utils import start_qt_app, run_qt_app


#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

def main():
    # TODO: use argparse

    if '-h' in sys.argv or '--help' in sys.argv:
        print(sys.modules[__name__].__doc__)
        return

    kwik_path = sys.argv[1]
    session = Session(kwik_path)

    print("\nLoading the data...")
    session.model.describe()

    start_qt_app()
    kv = session.show_gui(show=False)

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
