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
from typing import Tuple, Callable, Optional

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
    similar: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class Transition:
    name: str
    kwargs: dict = field(default_factory=dict)
    before: Optional[State] = None
    after: State


@dataclass()
class Callback:
    name: str
    function: Callable


@dataclass(kw_only=True)
class ClusterInfo:

    # Mandatory:
    first: Callable[[], int]
    last: Callable[[], int]
    similar: Callable[[list[int]], int]  # first similar cluster to the specified set of clusters
    new_cluster_id: Callable[[], int]

    # Optional:

    # id of the cluster after the specified clusters
    next: Optional[Callable[[], int]] = None

    # id of the cluster before the specified clusters
    prev: Optional[Callable[[], int]] = None

    # id of the new merged cluster that should be selected after a merge
    merge: Optional[Callable[[list[int]], int]] = None

    # id of the new clusters that should be selected after a split
    split: Optional[Callable[[list[int]], list[int]]] = None


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

        self.prev = cluster_info.prev
        self.next = cluster_info.next
        self.merge = cluster_info.merge
        self.split = cluster_info.split

        self._history = [Transition(name='init', after=state)]
        self._callbacks = []

    # -------------------------------------------------------------------------
    # After private methods
    # -------------------------------------------------------------------------

    def _after_manual(self, before: State = None, clusters=None, similar=None) -> State:
        """Determine the state after a manual transition."""
        return State(clusters=clusters, similar=similar)

    def _after_first(self, before: State = None) -> State:
        """Determine the state after a first transition."""
        return State(clusters=self.first(), similar=[])

    def _after_last(self, before: State = None) -> State:
        """Determine the state after a last transition."""
        return State(clusters=self.last(), similar=[])

    def _after_next_best(self, before: State = None) -> State:
        """Determine the state after a next_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = [self.next(before.clusters)]
        if self.current_similar():
            # Similarity view.
            after.similar = self.similar(after.clusters)

        return after

    def _after_prev_best(self, before: State = None) -> State:
        """Determine the state after a prev_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = [self.prev(before.clusters)]
        if self.current_similar():
            # Similarity view.
            after.similar = self.similar(after.clusters)

        return after

    def _after_next(self, before: State = None) -> State:
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
            after.similar = [self.next(before.similar)]

        return after

    def _after_prev(self, before: State = None) -> State:
        """Determine the state after a prev transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            return before

        # Similarity view.
        else:
            after.clusters = before.clusters
            after.similar = [self.prev(before.similar)]

        return after

    def _after_label(self, before: State = None) -> State:
        """Determine the state after a label transition."""
        return before

    def _after_move(self, before: State = None, which=None) -> State:
        """Determine the state after a move transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self._next_cluster()]

        # Similarity view.
        else:
            assert which
            if which == 'similar':
                after.clusters = before.clusters
                after.similar = [self.next(before.similar)]
            elif which in ('all', 'best'):
                after.clusters = [self._next_cluster()]
                after.similar = [self.next(after.clusters)]
            else:
                raise NotImplementedError(which)

        return after

    def _after_merge(self, before: State = None) -> State:
        """Determine the state after a merge transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self.merge(before.clusters)]

        # Similarity view.
        else:
            after.clusters = [self.merge(before.clusters + before.similar)]
            after.similar = [self.similar(after.clusters)]

        return after

    def _after_split(self, before: State = None) -> State:
        """Determine the state after a split transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = self.split(before.clusters)

        # Similarity view.
        else:
            after.clusters = self.split(before.clusters + before.similar)
            after.similar = [self.similar(after.clusters)]

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

    def current_clusters(self) -> list[int]:
        """Currently-selected clusters in the cluster view."""
        return self.current_state().clusters

    def current_similar(self) -> list[int]:
        """Currently-selected similar clusters in the similarity view."""
        return self.current_state().similar

    def can_undo(self) -> bool:
        """Whether we can undo."""
        return abs(self._cursor) < len(self._history)

    def can_redo(self) -> bool:
        """Whether we can redo."""
        return self._cursor != -1

    def history_length(self) -> int:
        """Return the number of transitions in the history."""
        return len(self._history)

    # -------------------------------------------------------------------------
    # Transition methods
    # -------------------------------------------------------------------------

    def set_state(
            self, clusters: Optional[list[int]] = None, similar: Optional[list[int]] = None):
        """Manually set a state by making a manual transition."""
        if not clusters and similar:
            clusters = self.current_clusters()
        clusters = clusters or []
        similar = similar or []

        assert isinstance(clusters, list)
        assert isinstance(similar, list)

        self.transition(transition_name='manual', clusters=clusters, similar=similar)

    def transition(self, transition_name: str, **kwargs) -> State:
        """Make a new transition."""
        before = self.current_state()

        method_name = '_after_%s' % transition_name
        method = getattr(self, method_name, None)
        assert method, f'method {method_name} not implemented'
        after = method(**kwargs)

        assert isinstance(before.clusters, list)
        assert isinstance(before.similar, list)
        assert isinstance(after.clusters, list)
        assert isinstance(after.similar, list)

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
