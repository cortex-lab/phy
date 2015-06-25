# -*- coding: utf-8 -*-
from __future__ import print_function

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


_examples = dedent("""

examples:
  phy -v                display the version of phy
  phy describe my_file.kwik
                        display information about a Kwik dataset
  phy detect-spikes my_params.prm
                        run spike detection on a parameters file
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
        self.create_detect()
        self.create_auto()
        self.create_manual()
        self.create_notebook()

    @property
    def parser(self):
        return self._parser

    def _add_sub_parser(self, name, desc):
        p = self._subparsers.add_parser(name, help=desc, description=desc)
        return p

    def create_main(self):
        import phy

        desc = sys.modules['phy'].__doc__
        self._parser = Parser(description=desc,
                              epilog=_examples,
                              formatter_class=CustomFormatter,
                              )
        self._parser.set_defaults(func=None)

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

        self._subparsers = self._parser.add_subparsers(dest='command',
                                                       title='subcommand',
                                                       )

    def create_describe(self):
        desc = 'describe a `.kwik` file'
        p = self._add_sub_parser('describe', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.set_defaults(func=describe)

    def create_manual(self):
        desc = 'launch the manual clustering GUI on a `.kwik` file'
        p = self._add_sub_parser('cluster-manual', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.add_argument('--cluster_ids', '-c',
                       help='list of clusters to select initially')
        p.set_defaults(func=cluster_manual)

    def create_auto(self):
        desc = 'launch the automatic clustering algorithm on a `.kwik` file'
        p = self._add_sub_parser('cluster-auto', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.add_argument('--num_starting_clusters', type=int,
                       help='initial number of clusters',
                       )
        p.set_defaults(func=cluster_auto)

    def create_detect(self):
        desc = 'launch the spike detection algorithm on a `.prm` file'
        p = self._add_sub_parser('detect-spikes', desc)
        p.add_argument('file', help='path to a `.prm` file')
        p.set_defaults(func=detect_spikes)

    def create_notebook(self):
        # TODO
        pass

    def parse(self, args):
        return self._parser.parse_args(args)


#------------------------------------------------------------------------------
# Subcommand functions
#------------------------------------------------------------------------------

def _create_session(args, **kwargs):
    kwik_path = args.file
    from phy.cluster import Session

    if not op.exists(kwik_path):
        print("The file `{}` doesn't exist.".format(kwik_path))
        return

    session = Session(kwik_path, **kwargs)
    return session


def describe(args):
    session = _create_session(args, clustering=args.clustering)
    return 'session.model.describe()', dict(session=session)


def cluster_manual(args):
    session = _create_session(args, clustering=args.clustering)
    cluster_ids = (list(map(int, args.cluster_ids.split(',')))
                   if args.cluster_ids else None)

    session = _create_session(args)
    session.model.describe()

    from phy.gui import start_qt_app
    start_qt_app()
    gui = session.show_gui(cluster_ids=cluster_ids, show=False)
    print("\nPress `ctrl+h` to see the list of keyboard shortcuts.\n")
    return 'gui.show()', dict(session=session, gui=gui, requires_qt=True)


def cluster_auto(args):
    session = _create_session(args, use_store=False)
    ns = dict(session=session,
              clustering=args.clustering,
              n_s_clusters=args.num_starting_clusters,
              )
    cmd = ('session.cluster(clustering=clustering, '
           'num_starting_clusters=n_s_clusters)')
    return (cmd, ns)


def detect_spikes(args):
    session = _create_session(args, use_store=False)
    return 'session.detect()', dict(session=session)


#------------------------------------------------------------------------------
# Main functions
#------------------------------------------------------------------------------

def main():

    p = ParserCreator()
    args = p.parse(sys.argv[1:])

    if args.profiler or args.line_profiler:
        from phy.utils.testing import _enable_profiler, _profile
        prof = _enable_profiler(args.line_profiler)
    else:
        prof = None

    import phy
    if args.debug:
        phy.debug()

    func = args.func
    if func is None:
        p.parser.print_help()
        return

    cmd, ns = func(args)
    requires_qt = ns.pop('requires_qt', False)
    ns.update(phy=phy, model=ns['session'].model, path=args.file)

    # Interactive mode with IPython.
    if args.ipython:
        print("\nStarting IPython...")
        from IPython import start_ipython
        args_ipy = ["-i", "-c='{}'".format(cmd)]
        if requires_qt:
            args_ipy += ["--gui=qt"]
        start_ipython(args_ipy, user_ns=ns)
    else:
        if requires_qt:
            from phy.gui import run_qt_app
            if not prof:
                exec_(cmd, {}, ns)
            else:
                _profile(prof, cmd, {}, ns)
            run_qt_app()


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
