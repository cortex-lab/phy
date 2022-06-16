# -*- coding: utf-8 -*-

"""Finite state machine handling the cluster selection wizard."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from functools import partial
import inspect
import logging
from pprint import pprint

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect, silent
from phy.gui.actions import Actions
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, _uniq

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Objects
# -----------------------------------------------------------------------------

@dataclass(kw_only=True)
class State:
    clusters: list[int] = field(default_factory=list)
    next_cluster: int | None = None
    similar: list[int] = field(default_factory=list)
    next_similar: int | None = None


@dataclass(kw_only=True)
class Transition:
    name: str = 'transition'
    kwargs: dict = field(default_factory=dict)
    before: State
    after: State


# ----------------------------------------------------------------------------
# Automaton
# -----------------------------------------------------------------------------

class Automaton:
    def __init__(
            self, next=None, prev=None, first=None, last=None,
            similar=None, merge=None, split=None):
        pass

    def current_state(self) -> State:
        pass

    def transition(self, transition_name: str, **kwargs) -> State:
        pass

    def _after_move(self, state: State = None, **kwargs) -> State:
        state = state or self.current_state()

    def _append(self, before, transition, after):
        pass

    def undo(self):
        pass

    def redo(self):
        pass

    def on_transition(self, name, callback):
        pass
