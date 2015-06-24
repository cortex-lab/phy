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
import re
import argparse
from textwrap import dedent

from ..ext.six import exec_


#------------------------------------------------------------------------------
# Main script
#------------------------------------------------------------------------------

class Parser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message + '\n\n')
        self.print_help()
        sys.exit(2)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def _parse_extra(extra):
    kwargs = {}
    reg = re.compile(r'^--([^\=]+)=([^\=]+)$')
    for e in extra:
        r = reg.match(e)
        if r:
            key, value = r.group(1), r.group(2)
            key = key.replace('-', '_')
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            kwargs[key] = value
    return kwargs


def _parse_args(args):
    desc = sys.modules['phy'].__doc__
    epilog = dedent("""

    examples:
      phy -v                display the version of phy
      phy describe my_file.kwik
                            display information about a Kwik dataset
      phy cluster-auto my_file.kwik --num-clusters-start=100
                            run klustakwik on a dataset
      phy cluster-manual my_file.kwik
                            run the manual clustering GUI

    """)
    parser = Parser(description=desc, epilog=epilog,
                    formatter_class=CustomFormatter,
                    )

    # Allowed subcommands.
    commands = [
        'cluster-auto',
        'cluster-manual',
        'describe',  # describe a dataset
        # TODO:
        # 'notebook',  # start a new analysis notebook
        # 'detect-spikes',
    ]

    parser.add_argument('command',
                        choices=commands,
                        help='command to execute')

    parser.add_argument('file',
                        help='file to execute the command on')

    import phy
    parser.add_argument('--version', '-v',
                        action='version',
                        version=phy.__version__,
                        help='print the version of phy')

    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='activate debug logging mode')

    parser.add_argument('--profiler', '-p',
                        action='store_true',
                        help='activate the profiler')

    parser.add_argument('--line-profiler', '-lp',
                        dest='line_profiler',
                        action='store_true',
                        help='activate the line-profiler -- you need to '
                        'decorate the functions to profile with `@profile` '
                        'in the code')

    parser.add_argument('--ipython', '-i', action='store_true',
                        help='launch the script in an interactive '
                        'IPython console')

    parser.add_argument('--clustering', default='main',
                        help='name of the clustering to use')

    parser.add_argument('--cluster_ids', '-c',
                        help='list of clusters to select initially')

    parse, extra = parser.parse_known_args(args)
    kwargs = _parse_extra(extra)
    return parse, kwargs


def run_manual(kwik_path, clustering=None, interactive=False,
               cluster_ids=None):
    import phy
    from phy.cluster import Session
    from phy.gui import start_qt_app, run_qt_app

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return 1

    print("\nLoading {}...".format(kwik_path))
    session = Session(kwik_path,
                      clustering=clustering,
                      )
    print("Data successfully loaded!\n")
    session.model.describe()

    start_qt_app()
    gui = session.show_gui(cluster_ids=cluster_ids, show=False)

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


def run_auto(kwik_path, clustering=None, interactive=False, **kwargs):
    from phy.cluster import Session

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    session = Session(kwik_path, use_store=False)
    session.cluster(clustering=clustering, **kwargs)
    session.save()
    session.close()


def describe(kwik_path, clustering=None):
    from phy.io.kwik import KwikModel

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    model = KwikModel(kwik_path, clustering=clustering)
    model.describe()
    model.close()


def main():

    args, kwargs = _parse_args(sys.argv[1:])

    if args.profiler or args.line_profiler:
        from phy.utils.testing import _enable_profiler, _profile
        prof = _enable_profiler(args.line_profiler)
    else:
        prof = None

    import phy
    if args.debug:
        phy.debug()

    if args.cluster_ids:
        cluster_ids = list(map(int, args.cluster_ids.split(',')))
    else:
        cluster_ids = None

    if args.command == 'cluster-manual':
        cmd = ('run_manual(args.file, clustering=args.clustering, '
               'interactive=args.ipython, cluster_ids=cluster_ids)')
    elif args.command == 'cluster-auto':
        cmd = ('run_auto(args.file, clustering=args.clustering, '
               'interactive=args.ipython, **kwargs)')
    elif args.command == 'describe':
        cmd = 'describe(args.file)'
    else:
        raise NotImplementedError()

    if not prof:
        exec_(cmd, globals(), locals())
    else:
        _profile(prof, cmd, globals(), locals())


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
