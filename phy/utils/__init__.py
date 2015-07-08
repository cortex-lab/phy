# -*- coding: utf-8 -*-
# flake8: noqa

"""Utilities."""

from ._types import _is_array_like, _as_array, _as_tuple, _as_list, Bunch
from .datasets import download_file, download_sample_data
from .event import EventEmitter, ProgressReporter
from .logging import debug, info, warn, register, unregister, set_level
from .settings import Settings, _ensure_dir_exists
