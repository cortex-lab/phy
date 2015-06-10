# -*- coding: utf-8 -*-

"""phy main CLI tool.

Usage:

    phy --help

"""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import os.path as op
import argparse


#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

def run(kwik_path, interactive=False):
    import phy
    from phy.cluster import Session
    from phy.gui import start_qt_app, run_qt_app

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


def main():
    desc = sys.modules['phy'].__doc__
    parser = argparse.ArgumentParser(description=desc)

    # Allowed subcommands.
    commands = [
        'cluster-auto',
        'cluster-manual',
        # TODO:
        # 'describe',  # describe a dataset
        # 'notebook',  # start a new analysis notebook
        # 'detect-spikes',
    ]

    parser.add_argument('command', choices=commands,
                        help='command to execute')

    parser.add_argument('file',
                        help='file to execute the command on')

    import phy
    parser.add_argument('--version', '-v', action='version',
                        version=phy.__version__,
                        help='print the version of phy')

    parser.add_argument('--debug', '-d', action='store_true',
                        help='activate debug logging mode')

    parser.add_argument('--profile', '-p', action='store_true',
                        help='activate the profiler')

    parser.add_argument('--profile-line', '-pl',
                        dest='profile_line',
                        action='store_true',
                        help='activate the line-profiler -- you need to '
                        'decorate the functions to profile with `@profile` '
                        'in the code')

    parser.add_argument('--ipython', '-i', action='store_true',
                        help='launch the script in an interactive '
                        'IPython console')

    # Parse the CLI arguments.
    args = parser.parse_args()

    if args.profile or args.profile_line:
        from phy.utils.testing import _enable_profiler, _profile
        prof = _enable_profiler(args.profile_line)
    else:
        prof = None

    if args.debug:
        phy.debug()

    if not prof:
        run(args.file, interactive=args.ipython)
    else:
        _profile(prof, 'run(args.file, interactive=args.ipython)',
                 globals(), locals())


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    exit(main())
