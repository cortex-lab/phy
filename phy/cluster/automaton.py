# -*- coding: utf-8 -*-

"""Finite state machine handling the cluster selection wizard."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from functools import wraps
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Objects
# -----------------------------------------------------------------------------

@dataclass(kw_only=True)
class State:
    clusters: list[int] = field(default_factory=list)
    similar: list[int] = field(default_factory=list)

    def copy(self):
        return State(clusters=self.clusters, similar=self.similar)


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

    # id of the cluster after the specified clusters in the cluster view
    next_best: Optional[Callable[[], int]] = None

    # id of the cluster before the specified clusters in the cluster view
    prev_best: Optional[Callable[[], int]] = None

    # id of the cluster after the specified clusters in the similarity view
    next_similar: Optional[Callable[[], int]] = None

    # id of the cluster before the specified clusters in the similarity view
    prev_similar: Optional[Callable[[], int]] = None

    # id of the new merged cluster that should be selected after a merge
    merge: Optional[Callable[[list[int], int], int]] = None

    # id of the new clusters that should be selected after a split
    split: Optional[Callable[[list[int]], list[int]]] = None


# ----------------------------------------------------------------------------
# Automaton
# ----------------------------------------------------------------------------

def ensure_int(f):
    """Ensure the output of a function is an integer."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        if not isinstance(out, int):
            raise TypeError(f"Output `{out}` of function {f} is not an integer.")
        return out
    return wrapped


class Automaton:
    KEEP_IN_HISTORY = ('label', 'move', 'merge', 'split')

    _history: list[Transition]
    _callbacks: list[Callback]
    _state: State
    _cursor = -1

    def __init__(self, state: State, cluster_info: ClusterInfo):

        assert cluster_info

        self.fn_first = ensure_int(cluster_info.first)
        self.fn_last = ensure_int(cluster_info.last)
        self.fn_similar = ensure_int(cluster_info.similar)
        self.fn_new_cluster_id = ensure_int(cluster_info.new_cluster_id)

        self.fn_next_best = ensure_int(cluster_info.next_best)
        self.fn_prev_best = ensure_int(cluster_info.prev_best)

        self.fn_next_similar = ensure_int(cluster_info.next_similar)
        self.fn_prev_similar = ensure_int(cluster_info.prev_similar)

        self.fn_merge = cluster_info.merge
        self.fn_split = cluster_info.split

        self._state = state
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
        return State(clusters=[self.fn_first()], similar=[])

    def _after_last(self, before: State = None) -> State:
        """Determine the state after a last transition."""
        return State(clusters=[self.fn_last()], similar=[])

    def _after_next_best(self, before: State = None) -> State:
        """Determine the state after a next_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = [self.fn_next_best(before.clusters)]
        # if self.current_similar():
        #     # Similarity view.
        #     after.similar = [self.fn_similar(after.clusters)]

        return after

    def _after_prev_best(self, before: State = None) -> State:
        """Determine the state after a prev_best transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        after.clusters = [self.fn_prev_best(before.clusters)]
        if self.current_similar():
            # Similarity view.
            after.similar = [self.fn_similar(after.clusters)]

        return after

    def _after_next(self, before: State = None) -> State:
        """Determine the state after a next transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = before.clusters or [self.fn_first()]
            after.similar = [self.fn_similar(after.clusters)]

        # Similarity view.
        else:
            after.clusters = before.clusters
            after.similar = [self.fn_next_similar(before.similar)]
            # If we're at the end of the similarity view, we switch to the next best cluster.
            # if after.similar == before.similar:
            #     after.clusters = [self.fn_next_best(before.clusters)]
            #     after.similar = [self.fn_similar(after.clusters)]

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
            after.similar = [self.fn_prev_similar(before.similar)]

        return after

    def _after_label(self, before: State = None) -> State:
        """Determine the state after a label transition."""
        before = before or self.current_state()
        return before

    def _after_move(self, before: State = None, which=None, group: str = None) -> State:
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
                after.similar = [self.fn_next_best(before.similar)]
            elif which in ('all', 'best'):
                after.clusters = [self._next_cluster()]
                after.similar = [self.fn_next_similar(after.clusters)]
            else:
                raise NotImplementedError(which)

        return after

    def _after_merge(self, before: State = None, to: int = None) -> State:
        """Determine the state after a merge transition."""

        assert to is not None
        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = [self.fn_merge(before.clusters, to)]

        # Similarity view.
        else:
            after.clusters = [self.fn_merge(before.clusters + before.similar, to)]
            after.similar = [self.fn_similar(after.clusters)]

        return after

    def _after_split(self, before: State = None, new_clusters: list[int] = None) -> State:
        """Determine the state after a split transition."""

        before = before or self.current_state()
        after = State()

        # Only cluster view
        if not self.current_similar():
            after.clusters = self.fn_split(before.clusters, new_clusters=new_clusters)

        # Similarity view.
        else:
            after.clusters = self.fn_split(
                before.clusters + before.similar, new_clusters=new_clusters)
            # after.similar = [self.fn_similar(after.clusters)]

        return after

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _next_cluster(self) -> int | None:
        """Return the next cluster in the cluster view."""
        cl = self.current_clusters()
        ncl = self.fn_next_best(cl)
        assert ncl is not None
        return ncl

    def _next_similar(self) -> int | None:
        """Return the next cluster in the similarity view."""
        return self.fn_next_similar(self.fn_similar())

    # -------------------------------------------------------------------------
    # Getter methods
    # -------------------------------------------------------------------------

    def current_state(self) -> State | None:
        """Return the current state."""
        # return self._history[self._cursor].after if self._history else None
        return self._state

    def current_clusters(self) -> list[int]:
        """Currently-selected clusters in the cluster view."""
        return self.current_state().clusters

    def current_similar(self) -> list[int]:
        """Currently-selected similar clusters in the similarity view."""
        return self.current_state().similar

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

        logger.log(
            5,
            f'Transition `{transition_name}` '
            f'(before: {before.clusters} {before.similar}, '
            f'after: {after.clusters} {after.similar}).')

        # Create the transition object.
        transition = Transition(
            name=transition_name,
            kwargs=kwargs,
            before=before,
            after=after,
        )
        self._state = after

        # Delete the transitions after the current cursor (destroy actions after undos).
        if transition_name in self.KEEP_IN_HISTORY:
            if self._cursor <= -2:
                del self._history[self._cursor + 1:]
            # Add it to the history list.
            self._history.append(transition)

        # Call the registered callbacks.
        for cb in self._callbacks:
            if cb.name == transition_name:
                cb.function(before, after, **kwargs)

    def on_transition(self, name, callback):
        """Register a callback when a given transition occurs."""
        self._callbacks.append(Callback(name, callback))

    def connect(self, f):
        """Register a callback for a transition, defined by the function's name e.g. on_merge."""
        name = f.__name__
        if not name.startswith('on_'):
            raise ValueError(f"function name `{f}` should start with on_")
        self.on_transition(name[3:], f)

    def first(self):
        return self.transition('first')

    def last(self):
        return self.transition('last')

    def next_best(self):
        return self.transition('next_best')

    def prev_best(self):
        return self.transition('prev_best')

    def next(self):
        return self.transition('next')

    def prev(self):
        return self.transition('prev')

    def label(self):
        return self.transition('label')

    def move(self, which=None, group=None):
        assert which in ('all', 'best', 'similar')
        return self.transition('move', which=which, group=group)

    def merge(self, to=None):
        assert to is not None
        return self.transition('merge', to=to)

    def split(self, new_clusters=None):
        return self.transition('split', new_clusters=new_clusters)

    # -------------------------------------------------------------------------
    # History methods
    # -------------------------------------------------------------------------

    def can_undo(self) -> bool:
        """Whether we can undo."""
        return abs(self._cursor) < len(self._history)

    def can_redo(self) -> bool:
        """Whether we can redo."""
        return self._cursor != -1

    def history_length(self) -> int:
        """Return the number of transitions in the history."""
        return len(self._history)

    def undo(self):
        """Undo the last transition."""
        if not self.can_undo():
            return
        # Transition leading to the state to revert to.
        transition = self._history[self._cursor]
        self._cursor -= 1
        assert self._cursor <= -1
        self._state = transition.before

        for cb in self._callbacks:
            if cb.name == 'undo':
                cb.function(transition.name, transition.before,
                            transition.after, **transition.kwargs)

    def redo(self):
        """Redo the last transition."""
        if not self.can_redo():
            return
        # Transition leading to the state to revert.
        self._cursor += 1
        assert self._cursor <= -1
        transition = self._history[self._cursor]
        self._state = transition.after

        for cb in self._callbacks:
            if cb.name == 'redo':
                cb.function(transition.name, transition.before,
                            transition.after, **transition.kwargs)
