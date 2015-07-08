# -*- coding: utf-8 -*-
from __future__ import print_function

"""GUI creator."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

import phy
from ...gui.base import BaseGUI
from ...gui.qt import _prompt
from ..view_models import (WaveformViewModel,
                           FeatureGridViewModel,
                           FeatureViewModel,
                           CorrelogramViewModel,
                           TraceViewModel,
                           StatsViewModel,
                           )
from ...utils.logging import debug, info, warn
from ...io.kwik.model import cluster_group_id
from ._history import GlobalHistory
from ._utils import ClusterMetadataUpdater
from .clustering import Clustering
from .wizard import Wizard, WizardViewModel


#------------------------------------------------------------------------------
# Manual clustering window
#------------------------------------------------------------------------------

def _check_list_argument(arg, name='clusters'):
    if not isinstance(arg, (list, tuple, np.ndarray)):
        raise ValueError("The argument should be a list or an array.")
    if len(name) == 0:
        raise ValueError("No {0} were selected.".format(name))


def _to_wizard_group(group_id):
    """Return the group name required by the wizard, as a function
    of the Kwik cluster group."""
    if hasattr(group_id, '__len__'):
        group_id = group_id[0]
    return {
        0: 'ignored',
        1: 'ignored',
        2: 'good',
        3: None,
        None: None,
    }.get(group_id, 'good')


def _process_ups(ups):
    """This function processes the UpdateInfo instances of the two
    undo stacks (clustering and cluster metadata) and concatenates them
    into a single UpdateInfo instance."""
    if len(ups) == 0:
        return
    elif len(ups) == 1:
        return ups[0]
    elif len(ups) == 2:
        up = ups[0]
        up.update(ups[1])
        return up
    else:
        raise NotImplementedError()


class ClusterManualGUI(BaseGUI):
    """Manual clustering GUI.

    This object represents a main window with:

    * multiple views
    * high-level clustering methods
    * global keyboard shortcuts

    Events
    ------

    cluster
    select
    request_save

    """

    _vm_classes = {
        'waveforms': WaveformViewModel,
        'features': FeatureViewModel,
        'features_grid': FeatureGridViewModel,
        'correlograms': CorrelogramViewModel,
        'traces': TraceViewModel,
        'wizard': WizardViewModel,
        'stats': StatsViewModel,
    }

    def __init__(self, model=None, store=None, cluster_ids=None, **kwargs):
        self.store = store
        self.wizard = Wizard()
        self._is_dirty = False
        self._cluster_ids = cluster_ids
        super(ClusterManualGUI, self).__init__(model=model,
                                               vm_classes=self._vm_classes,
                                               **kwargs)

    def _initialize_views(self):
        # The wizard needs to be started *before* the views are created,
        # so that the first cluster selection is already set for the views
        # when they're created.
        self.connect(self._connect_view, event='add_view')

        @self.main_window.connect_
        def on_close_gui():
            # Return False if the close event should be discarded, so that
            # the close Qt event is discarded.
            return self._prompt_save()

        self.on_open()
        self.wizard.start()
        if self._cluster_ids is None:
            self._cluster_ids = self.wizard.selection
        super(ClusterManualGUI, self)._initialize_views()

    # View methods
    # ---------------------------------------------------------------------

    @property
    def title(self):
        """Title of the main window."""
        name = self.__class__.__name__
        filename = getattr(self.model, 'kwik_path', 'mock')
        clustering = self.model.clustering
        channel_group = self.model.channel_group
        template = ("{filename} (shank {channel_group}, "
                    "{clustering} clustering) "
                    "- {name} - phy {version}")
        return template.format(name=name,
                               version=phy.__version_git__,
                               filename=filename,
                               channel_group=channel_group,
                               clustering=clustering,
                               )

    def _connect_view(self, view):
        """Connect a view to the GUI's events (select and cluster)."""
        @self.connect
        def on_select(cluster_ids, auto_update=True):
            view.select(cluster_ids, auto_update=auto_update)

        @self.connect
        def on_cluster(up):
            view.on_cluster(up)

    def _connect_store(self):
        @self.connect
        def on_cluster(up=None):
            self.store.update_spikes_per_cluster(self.model.spikes_per_cluster)
            # No need to delete the old clusters from the store, we can keep
            # them for possible undo, and regularly clean up the store.
            for item in self.store.items.values():
                item.on_cluster(up)

    def _set_default_view_connections(self):
        """Set view connections."""

        # Select feature dimension from waveform view.
        @self.connect_views('waveforms', 'features')
        def channel_click(waveforms, features):

            @waveforms.view.connect
            def on_channel_click(e):
                # The box id is set when the features grid view is to be
                # updated.
                if e.box_idx is not None:
                    return
                dimension = (e.channel_idx, 0)
                features.set_dimension(e.ax, dimension)
                features.update()

        # Select feature grid dimension from waveform view.
        @self.connect_views('waveforms', 'features_grid')
        def channel_click_grid(waveforms, features_grid):

            @waveforms.view.connect
            def on_channel_click(e):
                # The box id is set when the features grid view is to be
                # updated.
                if e.box_idx is None:
                    return
                if not (1 <= e.box_idx <= features_grid.n_rows - 1):
                    return
                dimension = (e.channel_idx, 0)
                box = (e.box_idx, e.box_idx)
                features_grid.set_dimension(e.ax, box, dimension)
                features_grid.update()

        # Enlarge feature subplot.
        @self.connect_views('features_grid', 'features')
        def enlarge(grid, features):

            @grid.view.connect
            def on_enlarge(e):
                features.set_dimension('x', e.x_dim, smart=False)
                features.set_dimension('y', e.y_dim, smart=False)
                features.update()

    def _view_model_kwargs(self, name):
        kwargs = {'model': self.model,
                  'store': self.store,
                  'wizard': self.wizard,
                  'cluster_ids': self._cluster_ids,
                  }
        return kwargs

    # Creation methods
    # ---------------------------------------------------------------------

    def _get_clusters(self, which):
        # Move best/match/both to noise/mua/good.
        return {
            'best': [self.wizard.best],
            'match': [self.wizard.match],
            'both': [self.wizard.best, self.wizard.match],
        }[which]

    def _create_actions(self):
        for action in ['reset_gui',
                       'save',
                       'exit',
                       'show_shortcuts',
                       'select',
                       # Wizard.
                       'reset_wizard',
                       'first',
                       'last',
                       'next',
                       'previous',
                       'pin',
                       'unpin',
                       # Actions.
                       'merge',
                       'split',
                       'undo',
                       'redo',
                       # Views.
                       'toggle_correlogram_normalization',
                       'toggle_waveforms_mean',
                       'toggle_waveforms_overlap',
                       'show_features_time',
                       ]:
            self._add_gui_shortcut(action)

        def _make_func(which, group):
            """Return a function that moves best/match/both clusters to
            a group."""

            def func():
                clusters = self._get_clusters(which)
                if None in clusters:
                    return
                self.move(clusters, group)

            name = 'move_{}_to_{}'.format(which, group)
            func.__name__ = name
            setattr(self, name, func)
            return name

        for which in ('best', 'match', 'both'):
            for group in ('noise', 'mua', 'good'):
                self._add_gui_shortcut(_make_func(which, group))

    def _create_cluster_metadata(self):
        self._cluster_metadata_updater = ClusterMetadataUpdater(
            self.model.cluster_metadata)

        @self.connect
        def on_cluster(up):
            for cluster in up.metadata_changed:
                group_0 = self._cluster_metadata_updater.group(cluster)
                group_1 = self.model.cluster_metadata.group(cluster)
                assert group_0 == group_1

    def _create_clustering(self):
        self.clustering = Clustering(self.model.spike_clusters)

        @self.connect
        def on_cluster(up):
            spc = self.clustering.spikes_per_cluster
            self.model.update_spikes_per_cluster(spc)

    def _create_global_history(self):
        self._global_history = GlobalHistory(process_ups=_process_ups)

    def _create_wizard(self):

        # Initialize the groups for the wizard.
        def _group(cluster):
            group_id = self._cluster_metadata_updater.group(cluster)
            return _to_wizard_group(group_id)

        groups = {cluster: _group(cluster)
                  for cluster in self.clustering.cluster_ids}
        self.wizard.cluster_groups = groups

        self.wizard.reset()

        # Set the similarity and quality functions for the wizard.
        @self.wizard.set_similarity_function
        def similarity(target, candidate):
            """Compute the distance between the mean masked features."""

            mu_0 = self.store.mean_features(target).ravel()
            mu_1 = self.store.mean_features(candidate).ravel()

            omeg_0 = self.store.mean_masks(target)
            omeg_1 = self.store.mean_masks(candidate)

            omeg_0 = np.repeat(omeg_0, self.model.n_features_per_channel)
            omeg_1 = np.repeat(omeg_1, self.model.n_features_per_channel)

            d_0 = mu_0 * omeg_0
            d_1 = mu_1 * omeg_1

            # WARNING: "-" because this is a distance, not a score.
            return -np.linalg.norm(d_0 - d_1)

        @self.wizard.set_quality_function
        def quality(cluster):
            """Return the maximum mean_masks across all channels
            for a given cluster."""
            return self.store.mean_masks(cluster).max()

        @self.connect
        def on_cluster(up):
            # HACK: get the current group as it is not available in `up`
            # currently.
            if up.description.startswith('metadata'):
                up = up.copy()
                cluster = up.metadata_changed[0]
                group = self.model.cluster_metadata.group(cluster)
                up.metadata_value = _to_wizard_group(group)

            # This called for both regular and history actions.
            # Save the wizard selection and update the wizard.
            self.wizard.on_cluster(up)

            # Update the wizard selection after a clustering action.
            self._wizard_select_after_clustering(up)

    def _wizard_select_after_clustering(self, up):
        # Make as few updates as possible in the views after clustering
        # actions. This allows for better before/after comparisons.
        if up.added:
            self.select(up.added, auto_update=False)
        elif up.description == 'metadata_group':
            # Select the last selected clusters after undo/redo.
            if up.history and up.selection:
                self.select(up.selection, auto_update=False)
                return
            cluster = up.metadata_changed[0]
            if cluster == self.wizard.best:
                self.wizard.next_best()
            elif cluster == self.wizard.match:
                self.wizard.next_match()
            self._wizard_select()
        elif up.selection:
            self.select(up.selection, auto_update=False)

    # Open data
    # -------------------------------------------------------------------------

    def on_open(self):
        """Reinitialize the GUI after new data has been loaded."""
        self._create_global_history()
        # This connects the callback that updates the model spikes_per_cluster.
        self._create_clustering()
        self._create_cluster_metadata()
        # This connects the callback that updates the store.
        self._connect_store()
        self._create_wizard()
        self._is_dirty = False

        @self.connect
        def on_cluster(up):
            self._is_dirty = True

    def save(self):
        """Save the changes."""
        # The session saves the model when this event is emitted.
        self.emit('request_save')

    def _prompt_save(self):
        """Display a prompt for saving and return whether the GUI should be
        closed."""
        if (self.settings.get('prompt_save_on_exit', False) and
                self._is_dirty):
            res = _prompt(self.main_window,
                          "Do you want to save your changes?",
                          ('save', 'cancel', 'close'))
            if res == 'save':
                self.save()
                return True
            elif res == 'cancel':
                return False
            elif res == 'close':
                return True
        return True

    # General actions
    # ---------------------------------------------------------------------

    def start(self):
        """Start the wizard."""
        self.wizard.start()
        self._cluster_ids = self.wizard.selection

    @property
    def cluster_ids(self):
        """Array of all cluster ids used in the current clustering."""
        return self.clustering.cluster_ids

    @property
    def n_clusters(self):
        """Number of clusters in the current clustering."""
        return self.clustering.n_clusters

    # View-related actions
    # ---------------------------------------------------------------------

    def toggle_correlogram_normalization(self):
        """Toggle CCG normalization in the correlograms views."""
        for vm in self.get_views('correlograms'):
            vm.toggle_normalization()

    def toggle_waveforms_mean(self):
        """Toggle mean mode in the waveform views."""
        for vm in self.get_views('waveforms'):
            vm.show_mean = not(vm.show_mean)

    def toggle_waveforms_overlap(self):
        """Toggle cluster overlap in the waveform views."""
        for vm in self.get_views('waveforms'):
            vm.overlap = not(vm.overlap)

    def show_features_time(self):
        """Set the x dimension to time in all feature views."""
        for vm in self.get_views('features'):
            vm.set_dimension('x', 'time')
            vm.update()

    # Selection
    # ---------------------------------------------------------------------

    def select(self, cluster_ids, **kwargs):
        """Select clusters."""
        cluster_ids = list(cluster_ids)
        assert len(cluster_ids) == len(set(cluster_ids))
        # Do not re-select an already-selected list of clusters.
        if cluster_ids == self._cluster_ids:
            return
        if not set(cluster_ids) <= set(self.clustering.cluster_ids):
            n_selected = len(cluster_ids)
            cluster_ids = [cl for cl in cluster_ids
                           if cl in self.clustering.cluster_ids]
            n_kept = len(cluster_ids)
            warn("{} of the {} selected clusters do not exist.".format(
                 n_selected - n_kept, n_selected))
        if len(cluster_ids) >= 14:
            warn("You cannot select more than 14 clusters in the GUI.")
            return
        debug("Select clusters {0:s}.".format(str(cluster_ids)))
        self._cluster_ids = cluster_ids
        self.emit('select', cluster_ids, **kwargs)

    @property
    def selected_clusters(self):
        """The list of selected clusters."""
        return self._cluster_ids

    # Wizard list
    # ---------------------------------------------------------------------

    def _wizard_select(self, **kwargs):
        self.select(self.wizard.selection, **kwargs)

    def reset_wizard(self):
        """Restart the wizard."""
        self.wizard.start()
        self._wizard_select()

    def first(self):
        """Go to the first cluster proposed by the wizard."""
        self.wizard.first()
        self._wizard_select()

    def last(self):
        """Go to the last cluster proposed by the wizard."""
        self.wizard.last()
        self._wizard_select()

    def next(self):
        """Go to the next cluster proposed by the wizard."""
        self.wizard.next()
        self._wizard_select()

    def previous(self):
        """Go to the previous cluster proposed by the wizard."""
        self.wizard.previous()
        self._wizard_select()

    def pin(self):
        """Pin the current best cluster."""
        cluster = (self.selected_clusters[0]
                   if len(self.selected_clusters) else None)
        self.wizard.pin(cluster)
        self._wizard_select()

    def unpin(self):
        """Unpin the current best cluster."""
        self.wizard.unpin()
        self._wizard_select()

    # Cluster actions
    # ---------------------------------------------------------------------

    def merge(self, clusters=None):
        """Merge some clusters."""
        if clusters is None:
            clusters = self.selected_clusters
        clusters = list(clusters)
        if len(clusters) <= 1:
            return
        up = self.clustering.merge(clusters)
        up.selection = self.selected_clusters
        info("Merge clusters {} to {}.".format(str(clusters),
                                               str(up.added[0])))
        self._global_history.action(self.clustering)
        self.emit('cluster', up=up)
        return up

    def _spikes_to_split(self):
        """Find the spikes lasso selected in a feature view for split."""
        for features in self.get_views('features', 'features_grid'):
            spikes = features.spikes_in_lasso()
            if spikes is not None:
                features.lasso.clear()
                return spikes

    def split(self, spikes=None):
        """Make a new cluster out of some spikes.

        Notes
        -----

        Spikes belonging to affected clusters, but not part of the `spikes`
        array, will move to brand new cluster ids. This is because a new
        cluster id must be used as soon as a cluster changes.

        """
        if spikes is None:
            spikes = self._spikes_to_split()
            if spikes is None:
                return
        if not len(spikes):
            info("No spikes to split.")
            return
        _check_list_argument(spikes, 'spikes')
        info("Split {0:d} spikes.".format(len(spikes)))
        up = self.clustering.split(spikes)
        up.selection = self.selected_clusters
        self._global_history.action(self.clustering)
        self.emit('cluster', up=up)
        return up

    def move(self, clusters, group):
        """Move some clusters to a cluster group.

        Here is the list of cluster groups:

        * 0=Noise
        * 1=MUA
        * 2=Good
        * 3=Unsorted

        """
        _check_list_argument(clusters)
        info("Move clusters {0} to {1}.".format(str(clusters), group))
        group_id = cluster_group_id(group)
        up = self._cluster_metadata_updater.set_group(clusters, group_id)
        up.selection = self.selected_clusters
        self._global_history.action(self._cluster_metadata_updater)
        self.emit('cluster', up=up)
        return up

    def _undo_redo(self, up):
        if up:
            info("{} {}.".format(up.history.title(),
                                 up.description,
                                 ))
            self.emit('cluster', up=up)

    def undo(self):
        """Undo the last clustering action."""
        up = self._global_history.undo()
        self._undo_redo(up)
        return up

    def redo(self):
        """Redo the last undone action."""
        # debug("The saved selection before the undo is {}.".format(clusters))
        up = self._global_history.redo()
        if up:
            up.selection = self.selected_clusters
        self._undo_redo(up)
        return up
