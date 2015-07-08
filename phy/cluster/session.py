# -*- coding: utf-8 -*-
from __future__ import print_function

"""Session structure."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import shutil

import numpy as np

from ..utils.logging import debug, info, FileLogger, unregister, register
from ..utils.settings import _ensure_dir_exists
from ..io.base import BaseSession
from ..io.kwik.model import KwikModel
from ..io.kwik.store_items import create_store
from .manual.gui import ClusterManualGUI
from .algorithms import KlustaKwik, SpikeDetekt


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

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


class Session(BaseSession):
    """A manual clustering session.

    This is the main object used for manual clustering. It implements
    all common actions:

    * Loading a dataset (`.kwik` file)
    * Listing the clusters
    * Changing the current channel group or current clustering
    * Showing views (waveforms, features, correlograms, etc.)
    * Clustering actions: merge, split, undo, redo
    * Wizard: cluster quality, best clusters, most similar clusters
    * Save back to .kwik

    """

    _vm_classes = ClusterManualGUI._vm_classes
    _gui_classes = {'cluster_manual': ClusterManualGUI}

    def __init__(self,
                 kwik_path=None,
                 clustering=None,
                 model=None,
                 use_store=True,
                 phy_user_dir=None,
                 waveform_filter=True,
                 ):
        self._clustering = clustering
        self._use_store = use_store
        self._file_logger = None
        self._waveform_filter = waveform_filter
        curdir = op.dirname(op.realpath(__file__))
        settings_path = op.join(curdir, 'default_settings.py')
        if kwik_path:
            kwik_path = op.realpath(kwik_path)
        super(Session, self).__init__(model=model,
                                      path=kwik_path,
                                      phy_user_dir=phy_user_dir,
                                      default_settings_path=settings_path,
                                      vm_classes=self._vm_classes,
                                      gui_classes=self._gui_classes,
                                      )

    def _backup_kwik(self, kwik_path):
        """Save a copy of the Kwik file before opening it."""
        if kwik_path is None:
            return
        backup_kwik_path = kwik_path + '.bak'
        if not op.exists(backup_kwik_path):
            info("Saving a backup of the Kwik file "
                 "in {0}.".format(backup_kwik_path))
            shutil.copyfile(kwik_path, backup_kwik_path)

    def _create_model(self, path):
        model = KwikModel(path,
                          clustering=self._clustering,
                          waveform_filter=self._waveform_filter,
                          )
        self._create_logger(path)
        return model

    def _create_logger(self, path):
        path = op.splitext(path)[0] + '.log'
        level = self.settings['log_file_level']
        if not self._file_logger:
            self._file_logger = FileLogger(filename=path, level=level)
            register(self._file_logger)

    def _save_model(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self.model.cluster_metadata.group(cluster)
                  for cluster in self.cluster_ids}
        self.model.save(self.model.spike_clusters,
                        groups,
                        clustering_metadata=self.model.clustering_metadata,
                        )
        info("Saved {0:s}.".format(self.model.kwik_path))

    # File-related actions
    # -------------------------------------------------------------------------

    def open(self, kwik_path=None, model=None):
        """Open a `.kwik` file."""
        self._backup_kwik(kwik_path)
        return super(Session, self).open(model=model, path=kwik_path)

    @property
    def kwik_path(self):
        """Path to the `.kwik` file."""
        return self.model.path

    @property
    def has_unsaved_changes(self):
        """Whether there are unsaved changes in the model.

        If true, a prompt message for saving will be displayed when closing
        the GUI.

        """
        # TODO
        pass

    # Properties
    # -------------------------------------------------------------------------

    @property
    def n_spikes(self):
        """Number of spikes in the current channel group."""
        return self.model.n_spikes

    @property
    def cluster_ids(self):
        """Array of all cluster ids used in the current clustering."""
        return self.model.cluster_ids

    @property
    def n_clusters(self):
        """Number of clusters in the current clustering."""
        return self.model.n_clusters

    # Event callbacks
    # -------------------------------------------------------------------------

    def _create_cluster_store(self):
        # Do not create the store if there is only one cluster.
        if self.model.n_clusters <= 1 or not self._use_store:
            # Just use a mock store.
            self.store = create_store(self.model,
                                      self.model.spikes_per_cluster,
                                      )
            return

        # Kwik store in experiment_dir/name.phy/1/main/cluster_store.
        store_path = op.join(self.settings.exp_settings_dir,
                             'cluster_store',
                             str(self.model.channel_group),
                             self.model.clustering
                             )
        _ensure_dir_exists(store_path)

        # Instantiate the store.
        spc = self.model.spikes_per_cluster
        cs = self.settings['features_masks_chunk_size']
        wns = self.settings['waveforms_n_spikes_max']
        wes = self.settings['waveforms_excerpt_size']
        self.store = create_store(self.model,
                                  path=store_path,
                                  spikes_per_cluster=spc,
                                  features_masks_chunk_size=cs,
                                  waveforms_n_spikes_max=wns,
                                  waveforms_excerpt_size=wes,
                                  )

        # Generate the cluster store if it doesn't exist or is invalid.
        # If the cluster store already exists and is consistent
        # with the data, it is not recreated.
        self.store.generate()

    def change_channel_group(self, channel_group):
        """Change the current channel group."""
        self.model.channel_group = channel_group
        info("Switched to channel group {}.".format(channel_group))
        self.emit('open')

    def change_clustering(self, clustering):
        """Change the current clustering."""
        self._clustering = clustering
        self.model.clustering = clustering
        info("Switched to `{}` clustering.".format(clustering))
        self.emit('open')

    def on_open(self):
        self._create_cluster_store()

    def on_close(self):
        if self._file_logger:
            unregister(self._file_logger)
            self._file_logger = None

    def register_statistic(self, func=None, shape=(-1,)):
        """Decorator registering a custom cluster statistic.

        Parameters
        ----------

        func : function
            A function that takes a cluster index as argument, and returns
            some statistics (generally a NumPy array).

        Notes
        -----

        This function will be called on every cluster when a dataset is opened.
        It is also automatically called on new clusters when clusters change.
        You can access the data from the model and from the cluster store.

        """
        if func is not None:
            return self.register_statistic()(func)

        def decorator(func):

            name = func.__name__

            def _wrapper(cluster):
                out = func(cluster)
                self.store.memory_store.store(cluster, **{name: out})

            # Add the statistics.
            stats = self.store.items['statistics']
            stats.add(name, _wrapper, shape)
            # Register it in the global cluster store.
            self.store.register_field(name, 'statistics')
            # Compute it on all existing clusters.
            stats.store_all(name=name, mode='force')
            info("Registered statistic `{}`.".format(name))

        return decorator

    # Spike sorting
    # -------------------------------------------------------------------------

    def detect(self, traces=None,
               interval=None,
               algorithm='spikedetekt',
               **kwargs):
        """Detect spikes in traces.

        Parameters
        ----------

        traces : array
            An `(n_samples, n_channels)` array. If unspecified, the Kwik
            file's raw data is used.
        interval : tuple (optional)
            A tuple `(start, end)` (in seconds) where to detect spikes.
        algorithm : str
            The algorithm name. Only `spikedetekt` currently.
        **kwargs : dictionary
            Algorithm parameters.

        Returns
        -------

        result : dict
            A `{channel_group: tuple}` mapping, where the tuple is:

            * `spike_times` : the spike times (in seconds).
            * `masks`: the masks of the spikes `(n_spikes, n_channels)`.

        """
        assert algorithm == 'spikedetekt'
        # Create `.phy/spikedetekt/` directory for temporary files.
        sd_dir = op.join(self.settings.exp_settings_dir, 'spikedetekt')
        _ensure_dir_exists(sd_dir)
        # Default interval.
        if interval is not None:
            (start_sec, end_sec) = interval
            sr = self.model.sample_rate
            interval_samples = (int(start_sec * sr),
                                int(end_sec * sr))
        else:
            interval_samples = None
        # Find the raw traces.
        traces = traces if traces is not None else self.model.traces
        # Take the parameters in the Kwik file, coming from the PRM file.
        params = self.model.metadata
        params.update(kwargs)
        # Probe parameters required by SpikeDetekt.
        params['probe_channels'] = self.model.probe.channels_per_group
        params['probe_adjacency_list'] = self.model.probe.adjacency
        # Start the spike detection.
        debug("Running SpikeDetekt with the following parameters: "
              "{}.".format(params))
        sd = SpikeDetekt(tempdir=sd_dir, **params)
        out = sd.run_serial(traces, interval_samples=interval_samples)

        # Add the spikes in the `.kwik` and `.kwx` files.
        for group in out.groups:
            spike_samples = out.spike_samples[group]
            self.model.creator.add_spikes(group=group,
                                          spike_samples=spike_samples,
                                          spike_recordings=None,  # TODO
                                          masks=out.masks[group],
                                          features=out.features[group],
                                          )
        self.emit('open')

        if out.groups:
            self.change_channel_group(out.groups[0])

    def cluster(self,
                clustering=None,
                algorithm='klustakwik',
                spike_ids=None,
                **kwargs):
        """Run an automatic clustering algorithm on all or some of the spikes.

        Parameters
        ----------

        clustering : str
            The name of the clustering in which to save the results.
        algorithm : str
            The algorithm name. Only `klustakwik` currently.
        spike_ids : array-like
            Array of spikes to cluster.

        Returns
        -------

        spike_clusters : array
            The spike_clusters assignements returned by the algorithm.

        """
        if clustering is None:
            clustering = 'main'
        # Make sure the clustering name does not exist already.
        if clustering in self.model.clusterings:
            raise ValueError("The clustering `{}` ".format(clustering) +
                             "already exists.")
        # Take KK2's default parameters.
        from klustakwik2.default_parameters import default_parameters
        params = default_parameters.copy()
        # Update the PRM ones, by filtering them.
        params.update({k: v for k, v in self.model.metadata.items()
                       if k in default_parameters})
        # Update the ones passed to the function.
        params.update(kwargs)

        # Original spike_clusters array.
        if self.model.spike_clusters is None:
            n_spikes = (len(spike_ids) if spike_ids is not None
                        else self.model.n_spikes)
            spike_clusters_orig = np.zeros(n_spikes, dtype=np.int32)
        else:
            spike_clusters_orig = self.model.spike_clusters.copy()

        # HACK: there needs to be one clustering.
        if not self.model.clusterings:
            self.model.add_clustering('empty', spike_clusters_orig)

        # Instantiate the KlustaKwik instance.
        kk = KlustaKwik(**kwargs)

        # Save the current clustering in the Kwik file.
        @kk.connect
        def on_iter(sc):
            # Update the original spike clusters.
            spike_clusters = spike_clusters_orig.copy()
            spike_clusters[spike_ids] = sc

            # Replace the kk2_current clustering.
            if 'kk2_current' in self.model.clusterings:
                self.model.delete_clustering('kk2_current')
            self.model.add_clustering('kk2_current', spike_clusters)
            info("Updated `kk2_current` clustering in the `.kwik` file.")

        info("Running {}...".format(algorithm))
        # Run KK.
        sc = kk.cluster(model=self.model, spike_ids=spike_ids)
        info("The automatic clustering process has finished.")

        # Save the results in the Kwik file.
        spike_clusters = spike_clusters_orig.copy()
        spike_clusters[spike_ids] = sc

        # Add a new clustering and switch to it.
        self.model.add_clustering(clustering, spike_clusters)

        # Copy the main clustering to original (only if this is the very
        # first run of the clustering algorithm).
        if clustering == 'main':
            self.model.copy_clustering('main', 'original')
        self.change_clustering(clustering)

        # Set the new clustering metadata.
        params = kk.params
        params['version'] = kk.version
        metadata = {'{}_{}'.format(algorithm, name): value
                    for name, value in params.items()}
        self.model.clustering_metadata.update(metadata)
        self.save()
        info("The clustering has been saved in the "
             "`{}` clustering in the `.kwik` file.".format(clustering))
        return sc

    # GUI
    # -------------------------------------------------------------------------

    def show_gui(self, **kwargs):
        """Show a GUI."""
        gui = super(Session, self).show_gui(store=self.store,
                                            **kwargs)

        @gui.connect
        def on_request_save():
            self.save()

        return gui
