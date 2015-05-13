# -*- coding: utf-8 -*-
# flake8: noqa

"""Utilities.

"""

from .logging import debug, info, warn, register, unregister, set_level
from ._types import _is_array_like, _as_array, _as_tuple, _as_list, Bunch
from .array import _unique, _index_of
from .event import EventEmitter, ProgressReporter
from .dock import DockWindow, start_qt_app, run_qt_app, qt_app, enable_qt
from .datasets import download_file, download_test_data
