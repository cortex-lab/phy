# -*- coding: utf-8 -*-
# flake8: noqa

"""Utilities: plugin system, event system, configuration system, profiling, debugging, cacheing,
basic read/write functions.
"""

from .plugin import IPlugin, attach_plugins
from .config import ensure_dir_exists, load_master_config, phy_config_dir
from .context import Context
from .color import(
    colormaps, selected_cluster_color, add_alpha, ClusterColorSelector
)

from phylib.utils import (
    Bunch, emit, connect, unconnect, silent, reset, set_silent,
    load_json, save_json, load_pickle, save_pickle, read_python,
    read_text, write_text, read_tsv, write_tsv,
)
