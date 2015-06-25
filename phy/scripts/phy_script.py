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
# Parser utilities
#------------------------------------------------------------------------------

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(message + '\n\n')
        self.print_help()
        sys.exit(2)


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


_examples = dedent("""

examples:
  phy -v                display the version of phy
  phy describe my_file.kwik
                        display information about a Kwik dataset
  phy cluster-auto my_file.kwik --num-clusters-start=100
                        run klustakwik on a dataset
  phy cluster-manual my_file.kwik
                        run the manual clustering GUI

""")


#------------------------------------------------------------------------------
# Parser creator
#------------------------------------------------------------------------------

class ParserCreator(object):
    def __init__(self):
        self.create_main()
        self.create_describe()
        self.create_manual()
        self.create_auto()
        self.create_detect()
        self.create_notebook()

    def create_main(self):
        import phy

        desc = sys.modules['phy'].__doc__
        self._parser = Parser(description=desc,
                              epilog=_examples,
                              formatter_class=CustomFormatter,
                              )

        self._parser.add_argument('--version', '-v',
                                  action='version',
                                  version=phy.__version__,
                                  help='print the version of phy')

        self._parser.add_argument('--debug', '-d',
                                  action='store_true',
                                  help='activate debug logging mode')

        self._parser.add_argument('--profiler', '-p',
                                  action='store_true',
                                  help='activate the profiler')

        self._parser.add_argument('--line-profiler', '-lp',
                                  dest='line_profiler',
                                  action='store_true',
                                  help='activate the line-profiler -- you '
                                       'need to decorate the functions '
                                       'to profile with `@profile` '
                                       'in the code')

        self._parser.add_argument('--ipython', '-i', action='store_true',
                                  help='launch the script in an interactive '
                                  'IPython console')

        self._subparsers = self._parser.add_subparsers(help='sub-command help',
                                                       dest='command',
                                                       )

    def create_describe(self):
        p = self._subparsers.add_parser('describe', help='describe a dataset')
        p.add_argument('file', help='path to a `.kwik` file')

    def create_manual(self):
        p = self._subparsers.add_parser('cluster-manual',
                                        help='launch the manual clustering '
                                             'GUI on a `.kwik` file')
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.add_argument('--cluster_ids', '-c',
                       help='list of clusters to select initially')

    def create_auto(self):
        p = self._subparsers.add_parser('cluster-auto',
                                        help='launch the automatic clustering '
                                             'algorithm on a `.kwik` file')
        p.add_argument('file', help='path to a `.kwik` file')

    def create_detect(self):
        p = self._subparsers.add_parser('detect-spikes',
                                        help='launch the spike detection '
                                             'algorithm on a `.prm` file')
        p.add_argument('file', help='path to a `.prm` file')

    def create_notebook(self):
        # TODO
        pass

    def parse(self, args):
        parse, extra = self._parser.parse_known_args(args)
        kwargs = _parse_extra(extra)
        return parse, kwargs


#------------------------------------------------------------------------------
# Main functions
#------------------------------------------------------------------------------

def describe(kwik_path, clustering=None):
    from phy.io.kwik import KwikModel

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    model = KwikModel(kwik_path, clustering=clustering)
    model.describe()
    model.close()


def cluster_manual(kwik_path, clustering=None, interactive=False,
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


def cluster_auto(kwik_path, clustering=None, interactive=False, **kwargs):
    from phy.cluster import Session

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    session = Session(kwik_path, use_store=False)
    session.cluster(clustering=clustering, **kwargs)
    session.save()
    session.close()


def detect(kwik_path, clustering=None, interactive=False, **kwargs):
    from phy.cluster import Session

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    session = Session(kwik_path, use_store=False)
    session.cluster(clustering=clustering, **kwargs)
    session.save()
    session.close()


def main():

    p = ParserCreator()
    args, kwargs = p.parse(sys.argv[1:])

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
        cmd = ('cluster_manual(args.file, clustering=args.clustering, '
               'interactive=args.ipython, cluster_ids=cluster_ids)')
    elif args.command == 'cluster-auto':
        cmd = ('cluster_auto(args.file, clustering=args.clustering, '
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
