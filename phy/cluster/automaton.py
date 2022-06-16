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
from typing import Tuple, Callable

import numpy as np

from ._history import GlobalHistory
from ._utils import create_cluster_meta
from .clustering import Clustering

from phylib.utils import Bunch, emit, connect, unconnect, silent
from phy.gui.actions import Actions
from phy.gui.qt import _block, set_busy, _wait
from phy.gui.widgets import Table, _uniq

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Objects
# -----------------------------------------------------------------------------

@dataclass(kw_only=True)
class State:
    clusters: list[int] = field(default_factory=list)
    # next_cluster: int | None = None
    similar: list[int] = field(default_factory=list)
    # next_similar: int | None = None


@dataclass(kw_only=True)
class Transition:
    name: str
    kwargs: dict = field(default_factory=dict)
    before: State | None = None
    after: State


@dataclass()
class Callback:
    name: str
    function: Callable


@dataclass(kw_only=True)
class ClusterInfo:

    # Default functions

    def _default_next(self, clusters):
        if not clusters:
            return
        nx = clusters[0] + 1
        while nx in clusters:
            nx += 1
        return nx

    def _default_prev(self, clusters):
        if not clusters:
            return
        nx = clusters[0] - 1
        while nx in clusters:
            nx -= 1
        return nx

    def _default_merge_split(self, clusters):
        return self.new_cluster_id() + 1

    # Mandatory:
    first: Callable[[], int]
    last: Callable[[], int]
    similar: Callable[[list[int]], int]  # first similar cluster to the specified set of clusters
    new_cluster_id: Callable[[], int]

    # Optional:
    next: Callable[[], int] | None = _default_next
    prev: Callable[[], int] | None = _default_prev
    merge: Callable[[list[int]], int] | None = _default_merge_split
    split: Callable[[list[int]], int] | None = _default_merge_split


# ----------------------------------------------------------------------------
# Automaton
# ----------------------------------------------------------------------------

class Automaton:
    _history: list[Transition]
    _callbacks: list[Callback]
    _cursor = -1

    def __init__(self, state: State, cluster_info: ClusterInfo):

        assert cluster_info

        self.first = cluster_info.first
        self.last = cluster_info.last
        self.similar = cluster_info.similar
        self.new_cluster_id = cluster_info.new_cluster_id

        self.prev = cluster_info.prev or self._default_prev
        self.next = cluster_info.next or self._default_next
        self.merge = cluster_info.merge or self._default_merge_split
        self.split = cluster_info.split or self._default_merge_split

        self._history = [Transition(name='init', after=state)]
        self._callbacks = []

    # -------------------------------------------------------------------------
    # After private methods
    # -------------------------------------------------------------------------

    def _after_first(self, before: State = None, **kwargs) -> State:
        """Determine the state after a first transition."""
        return State(clusters=self.first(), similar=[])

    def _after_last(self, before: State = None, **kwargs) -> State:
        """Determine the state after a last transition."""
        return State(clusters=self.last(), similar=[])

    def _after_next_best(self, before: State = None, **kwargs) -> State:
        """Determine the state after a next_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = self.next(before.clusters)
        if self.current_similar():
            # Similarity view.
            after.similar = self.similar(after.clusters)

        return after

    def _after_prev_best(self, before: State = None, **kwargs) -> State:
        """Determine the state after a prev_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = self.prev(before.clusters)
        if self.current_similar():
            # Similarity view.
            after.similar = self.similar(after.clusters)

        return after

    def _after_next(self, before: State = None, **kwargs) -> State:
        """Determine the state after a next transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = before.clusters
            after.similar = self.similar(before.clusters)

        # Similarity view.
        else:
            after.clusters = before.clusters
            after.similar = self.next(before.similar)

        return after

    def _after_prev(self, before: State = None, **kwargs) -> State:
        """Determine the state after a prev transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            return before

        # Similarity view.
        else:
            after.clusters = before.clusters
            after.similar = self.prev(before.similar)

        return after

    def _after_label(self, before: State = None, **kwargs) -> State:
        """Determine the state after a label transition."""
        return before

    def _after_move(self, before: State = None, **kwargs) -> State:
        """Determine the state after a move transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self._next_cluster()]

        # Similarity view.
        else:
            which = kwargs.get('which')
            if which == 'similar':
                after.clusters = before.clusters
                after.similar = [self.next(before.similar)]
            else:
                after.clusters = [self._next_cluster()]
                after.similar = self.next(after.clusters)

        return after

    def _after_merge(self, before: State = None, **kwargs) -> State:
        """Determine the state after a merge transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self.merge(before.clusters)]

        # Similarity view.
        else:
            after.clusters = self.merge(before.clusters + before.similar)
            after.similar = self.next(after.clusters)

        return after

    def _after_split(self, before: State = None, **kwargs) -> State:
        """Determine the state after a split transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self.split(before.clusters)]

        # Similarity view.
        else:
            after.clusters = self.split(before.clusters + before.similar)
            after.similar = self.next(after.clusters)

        return after

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _next_cluster(self) -> int | None:
        """Return the next cluster in the cluster view."""
        return self.next(self.current_clusters())

    def _next_similar(self) -> int | None:
        """Return the next cluster in the similarity view."""
        return self.next(self.similar())

    # -------------------------------------------------------------------------
    # Getter methods
    # -------------------------------------------------------------------------

    def current_state(self) -> State | None:
        """Return the current state."""
        return self._history[self._cursor].after if self._history else None

    def can_undo(self) -> bool:
        """Whether we can undo."""
        return abs(self._cursor) < len(self._history)

    def can_redo(self) -> bool:
        """Whether we can redo."""
        return self._cursor != -1

    def history_length(self) -> int:
        """Return the number of transitions in the history."""
        return len(self._history)

    def current_clusters(self) -> list[int]:
        """Currently-selected clusters in the cluster view."""
        return self.current_state().clusters

    def current_similar(self) -> list[int]:
        """Currently-selected similar clusters in the similarity view."""
        return self.current_state().similar

    # -------------------------------------------------------------------------
    # Ection methods
    # -------------------------------------------------------------------------

    def transition(self, transition_name: str, **kwargs) -> State:
        """Make a new transition."""
        before = self.current_state()

        method_name = '_after_%s' % transition_name
        method = getattr(self, method_name, None)
        assert method, f'method {method_name} not implemented'
        after = method(**kwargs)

        # Create the transition object.
        transition = Transition(
            name=transition_name,
            kwargs=kwargs,
            before=before,
            after=after,
        )

        # Delete the transitions after the current cursor (destroy actions after undos).
        del self._history[self._cursor + 1:]
        # Add it to the history list.
        self._history.append(transition)

        # Call the registered callbacks.
        for cb in self._callbacks:
            if cb.name == transition_name:
                cb.function(before, after, **kwargs)

    def undo(self):
        """Undo the last transition."""
        if self.can_undo():
            self._cursor -= 1
        assert self._cursor <= -1

    def redo(self):
        """Redo the last transition."""
        if self.can_redo():
            self._cursor += 1
        assert self._cursor <= -1

    def on_transition(self, name, callback):
        """Register a callback when a given transition occurs."""
        self._callbacks.append(Callback(name, callback))
