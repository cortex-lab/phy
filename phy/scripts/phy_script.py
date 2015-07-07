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

import numpy as np

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
  phy download hybrid_120sec.dat -o data/
                        download a sample raw data file in `data/`
  phy describe my_file.kwik
                        display information about a Kwik dataset
  phy spikesort my_params.prm
                        run the whole suite (spike detection and clustering)
  phy detect my_params.prm
                        run spike detection on a parameters file
  phy cluster-auto my_file.kwik
                        run klustakwik on a dataset (after spike detection)
  phy cluster-manual my_file.kwik
                        run the manual clustering GUI

""")


#------------------------------------------------------------------------------
# Parser creator
#------------------------------------------------------------------------------

class ParserCreator(object):
    def __init__(self):
        self.create_main()
        self.create_download()
        self.create_traces()
        self.create_describe()
        self.create_spikesort()
        self.create_detect()
        self.create_auto()
        self.create_manual()
        self.create_notebook()

    @property
    def parser(self):
        return self._parser

    def _add_sub_parser(self, name, desc):
        p = self._subparsers.add_parser(name, help=desc, description=desc)
        self._add_options(p)
        return p

    def _add_options(self, parser):
        parser.add_argument('--debug', '-d',
                            action='store_true',
                            help='activate debug logging mode')

        parser.add_argument('--hide-traceback',
                            action='store_true',
                            help='hide the traceback for cleaner error '
                                 'messages')

        parser.add_argument('--profiler', '-p',
                            action='store_true',
                            help='activate the profiler')

        parser.add_argument('--line-profiler', '-lp',
                            dest='line_profiler',
                            action='store_true',
                            help='activate the line-profiler -- you '
                                 'need to decorate the functions '
                                 'to profile with `@profile` '
                                 'in the code')

        parser.add_argument('--ipython', '-i', action='store_true',
                            help='launch the script in an interactive '
                            'IPython console')

        parser.add_argument('--pdb', action='store_true',
                            help='activate the Python debugger')

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
                                  version=phy.__version_git__,
                                  help='print the version of phy')
        self._add_options(self._parser)
        self._subparsers = self._parser.add_subparsers(dest='command',
                                                       title='subcommand',
                                                       )

    def create_download(self):
        desc = 'download a sample dataset'
        p = self._add_sub_parser('download', desc)
        p.add_argument('file', help='dataset filename')
        p.add_argument('--output-dir', '-o', help='output directory')
        p.add_argument('--base',
                       default='cortexlab',
                       choices=('cortexlab', 'github'),
                       help='data repository name: `cortexlab` or `github`',
                       )
        p.set_defaults(func=download)

    def create_describe(self):
        desc = 'describe a `.kwik` file'
        p = self._add_sub_parser('describe', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.set_defaults(func=describe)

    def create_traces(self):
        desc = 'show the traces of a raw data file'
        p = self._add_sub_parser('traces', desc)
        p.add_argument('file', help='path to a `.kwd` or `.dat` file')
        p.add_argument('interval', help='interval in number '
                       'of samples',)
        p.add_argument('--n-channels', '-n',
                       help='number of channels in the recording '
                       '(only required when using a flat binary file)')
        p.add_argument('--dtype',
                       help='NumPy data type '
                       '(only required when using a flat binary file)',
                       default='int16',
                       )
        p.set_defaults(func=traces)

    def create_spikesort(self):
        desc = 'launch the whole spike sorting pipeline on a `.prm` file'
        p = self._add_sub_parser('spikesort', desc)
        p.add_argument('file', help='path to a `.prm` file')
        p.add_argument('--kwik-path', help='filename of the `.kwik` file '
                       'to create (by default, `"experiment_name".kwik`)')
        p.add_argument('--overwrite', action='store_true', default=False,
                       help='overwrite the `.kwik` file ')
        p.add_argument('--interval',
                       help='detection interval in seconds (e.g. `0,10`)')
        p.set_defaults(func=spikesort)

    def create_detect(self):
        desc = 'launch the spike detection algorithm on a `.prm` file'
        p = self._add_sub_parser('detect', desc)
        p.add_argument('file', help='path to a `.prm` file')
        p.add_argument('--kwik-path', help='filename of the `.kwik` file '
                       'to create (by default, `"experiment_name".kwik`)')
        p.add_argument('--overwrite', action='store_true', default=False,
                       help='overwrite the `.kwik` file ')
        p.set_defaults(func=detect)

    def create_auto(self):
        desc = 'launch the automatic clustering algorithm on a `.kwik` file'
        p = self._add_sub_parser('cluster-auto', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.set_defaults(func=cluster_auto)

    def create_manual(self):
        desc = 'launch the manual clustering GUI on a `.kwik` file'
        p = self._add_sub_parser('cluster-manual', desc)
        p.add_argument('file', help='path to a `.kwik` file')
        p.add_argument('--clustering', default='main',
                       help='name of the clustering to use')
        p.add_argument('--cluster-ids', '-c',
                       help='list of clusters to select initially')
        p.add_argument('--no-store', action='store_true', default=False,
                       help='do not create the store (faster loading time, '
                            'slower GUI)')
        p.set_defaults(func=cluster_manual)

    def create_notebook(self):
        # TODO
        pass

    def parse(self, args):
        return self._parser.parse_args(args)


#------------------------------------------------------------------------------
# Subcommand functions
#------------------------------------------------------------------------------

def _get_kwik_path(args):
    kwik_path = args.file

    if not op.exists(kwik_path):
        raise IOError("The file `{}` doesn't exist.".format(kwik_path))

    return kwik_path


def _create_session(args, **kwargs):
    from phy.cluster import Session
    kwik_path = _get_kwik_path(args)
    session = Session(kwik_path, **kwargs)
    return session


def describe(args):
    from phy.io.kwik import KwikModel
    path = _get_kwik_path(args)
    model = KwikModel(path, clustering=args.clustering)
    return 'model.describe()', dict(model=model)


def download(args):
    from phy import download_sample_data
    download_sample_data(args.file,
                         output_dir=args.output_dir,
                         base=args.base,
                         )


def traces(args):
    from vispy.app import run
    from phy.plot.traces import TraceView
    from phy.io.h5 import open_h5
    from phy.io.traces import read_kwd, read_dat

    path = args.file
    if path.endswith('.kwd'):
        f = open_h5(args.file)
        traces = read_kwd(f)
    elif path.endswith(('.dat', '.bin')):
        if not args.n_channels:
            raise ValueError("Please specify `--n-channels`.")
        if not args.dtype:
            raise ValueError("Please specify `--dtype`.")
        n_channels = int(args.n_channels)
        dtype = np.dtype(args.dtype)
        traces = read_dat(path, dtype=dtype, n_channels=n_channels)

    start, end = map(int, args.interval.split('-'))

    c = TraceView(keys='interactive')
    c.visual.traces = .01 * traces[start:end, ...]
    c.show()
    run()

    return None, None


def detect(args):
    from phy.io import create_kwik

    assert args.file.endswith('.prm')
    kwik_path = args.kwik_path
    kwik_path = create_kwik(args.file,
                            overwrite=args.overwrite,
                            kwik_path=kwik_path)
    # Create the session with the newly-created .kwik file.
    args.file = kwik_path
    session = _create_session(args, use_store=False)
    return 'session.detect()', dict(session=session)


def cluster_auto(args):
    assert args.file.endswith('.kwik')
    session = _create_session(args, use_store=False)
    ns = dict(session=session,
              clustering=args.clustering,
              )
    cmd = ('session.cluster(clustering=clustering)')
    return (cmd, ns)


def spikesort(args):
    from phy.io import create_kwik

    assert args.file.endswith('.prm')
    kwik_path = args.kwik_path
    kwik_path = create_kwik(args.file,
                            overwrite=args.overwrite,
                            kwik_path=kwik_path,
                            )
    # Create the session with the newly-created .kwik file.
    args.file = kwik_path
    session = _create_session(args, use_store=False)

    interval = args.interval
    if interval is not None:
        interval = list(map(float, interval.split(',')))

    ns = dict(session=session,
              interval=interval,
              n_s_clusters=100,  # TODO: better handling of KK parameters
              )
    cmd = ('session.detect(interval=interval); session.cluster();')
    return (cmd, ns)


def cluster_manual(args):
    session = _create_session(args,
                              clustering=args.clustering,
                              use_store=not(args.no_store),
                              )
    cluster_ids = (list(map(int, args.cluster_ids.split(',')))
                   if args.cluster_ids else None)

    session.model.describe()

    from phy.gui import start_qt_app
    start_qt_app()

    gui = session.show_gui(cluster_ids=cluster_ids, show=False)
    print("\nPress `ctrl+h` to see the list of keyboard shortcuts.\n")
    return 'gui.show()', dict(session=session, gui=gui, requires_qt=True)


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

    # Hide the traceback.
    if args.hide_traceback:
        def exception_handler(exception_type, exception, traceback):
            print("{}: {}".format(exception_type.__name__, exception))

        sys.excepthook = exception_handler

    # Activate IPython debugger.
    if args.pdb:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                             color_scheme='Linux',
                                             call_pdb=1,
                                             )

    func = args.func
    if func is None:
        p.parser.print_help()
        return

    out = func(args)
    if not out:
        return
    cmd, ns = out
    if not cmd:
        return
    requires_qt = ns.pop('requires_qt', False)
    requires_vispy = ns.pop('requires_vispy', False)

    # Default variables in namespace.
    ns.update(phy=phy, path=args.file)
    if 'session' in ns:
        ns['model'] = ns['session'].model

    # Interactive mode with IPython.
    if args.ipython:
        print("\nStarting IPython...")
        from IPython import start_ipython
        args_ipy = ["-i", "-c='{}'".format(cmd)]
        if requires_qt or requires_vispy:
            # Activate Qt event loop integration with Qt.
            args_ipy += ["--gui=qt"]
        start_ipython(args_ipy, user_ns=ns)
    else:
        if not prof:
            exec_(cmd, {}, ns)
        else:
            _profile(prof, cmd, {}, ns)

        if requires_qt:
            # Launch the Qt app.
            from phy.gui import run_qt_app
            run_qt_app()
        elif requires_vispy:
            # Launch the VisPy Qt app.
            from vispy.app import use_app, run
            use_app('pyqt4')
            run()


#------------------------------------------------------------------------------
# Entry point
#------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
