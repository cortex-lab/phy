# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from phylib.stats.clusters import get_waveform_amplitude
from phylib.utils import Bunch, connect
from phylib.utils.geometry import linear_positions

from phy.utils.context import Context
from phy.gui import create_app, run_app
from ..base import WaveformMixin, FeatureMixin, TraceMixin, BaseController
from phy.cluster.supervisor import Supervisor

logger = logging.getLogger(__name__)

try:
    from klusta.kwik import KwikModel
    from klusta.launch import cluster
except ImportError:  # pragma: no cover
    logger.debug("Package klusta not installed: the KwikGUI will not work.")


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    assert path.exists()
    path_backup = str(path) + '.bak'
    if not Path(path_backup).exists():
        logger.info("Backup `%s`.", path_backup)
        shutil.copy(str(path), str(path_backup))


class KwikModelGUI(KwikModel):
    @property
    def features(self):
        return self.all_features

    def get_features(self, spike_ids, channel_ids):
        return self.all_features[spike_ids][:, channel_ids, :]

    def get_waveforms(self, spike_ids, channel_ids):
        return self.all_waveforms[spike_ids][:, channel_ids, :]


class KwikController(WaveformMixin, FeatureMixin, TraceMixin, BaseController):
    """Controller for the Kwik GUI.

    Constructor
    -----------
    kwik_path : str or Path
        Path to the kwik file
    channel_group : int
        The default channel group to load
    clustering : str
        The default clustering to load
    config_dir : str or Path
        Path to the configuration directory
    model : Model
        Model object, optional (it is automatically created otherwise)
    plugins : list
        List of plugins to manually activate, optional (the plugins are automatically loaded from
        the user configuration directory).
    clear_cache : boolean
        Whether to clear the cache on startup.
    enable_threading : boolean
        Whether to enable threading in the views when selecting clusters.

    """

    gui_name = 'KwikGUI'

    # Classes to load by default, in that order. The view refresh follows the same order
    # when the cluster selection changes.
    default_views = (
        'CorrelogramView',
        'ISIView',
        'WaveformView',
        'FeatureView',
        'AmplitudeView',
        'FiringRateView',
        'TraceView',
    )

    def __init__(self, kwik_path=None, **kwargs):
        assert kwik_path
        kwik_path = Path(kwik_path)
        dir_path = kwik_path.parent
        self.channel_group = kwargs.get('channel_group', None)
        self.clustering = kwargs.get('clustering', None)
        super(KwikController, self).__init__(kwik_path=kwik_path, dir_path=dir_path, **kwargs)

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_cache(self, clear_cache=None):
        """Set up the cache, clear it if required, and create the Context instance."""
        self.cache_dir = self.dir_path / '.phy'
        if self.channel_group is not None:
            self.cache_dir = self.cache_dir / str(self.channel_group)
        if clear_cache:
            logger.warn("Deleting the cache directory %s.", self.cache_dir)
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.context = Context(self.cache_dir)

    def _create_model(self, **kwargs):
        kwik_path = kwargs.get('kwik_path')
        _backup(kwik_path)
        kwargs = {k: v for k, v in kwargs.items() if k in ('clustering', 'channel_group')}
        model = KwikModelGUI(str(kwik_path), **kwargs)
        # HACK: handle badly formed channel positions
        if model.channel_positions.ndim == 1:  # pragma: no cover
            logger.warning("Unable to read the channel positions, generating mock ones.")
            model.probe.positions = linear_positions(len(model.channel_positions))
        return model

    def _set_supervisor(self):
        """Create the Supervisor instance."""
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id').get('new_cluster_id', None)

        # Cluster groups.
        cluster_groups = self.model.cluster_groups

        # Create the Supervisor instance.
        supervisor = Supervisor(
            spike_clusters=self.model.spike_clusters,
            cluster_groups=cluster_groups,
            cluster_metrics=self.cluster_metrics,
            similarity=self.similarity_functions[self.similarity],
            new_cluster_id=new_cluster_id,
            context=self.context,
        )

        # Connect the `save_clustering` event raised by the supervisor when saving
        # to the model's saving functions.
        connect(self.on_save_clustering, sender=supervisor)

        @connect(sender=supervisor)
        def on_attach_gui(sender):
            @supervisor.actions.add(shortcut='shift+ctrl+k', set_busy=True)
            def recluster(cluster_ids=None):
                """Relaunch KlustaKwik on the selected clusters."""
                # Selected clusters.
                cluster_ids = supervisor.selected
                spike_ids = self.selector(None, cluster_ids)
                logger.info("Running KlustaKwik on %d spikes.", len(spike_ids))

                # Run KK2 in a temporary directory to avoid side effects.
                n = 10
                with TemporaryDirectory() as tempdir:
                    spike_clusters, metadata = cluster(
                        self.model,
                        spike_ids,
                        num_starting_clusters=n,
                        tempdir=tempdir,
                    )
                self.supervisor.split(spike_ids, spike_clusters)

        self.supervisor = supervisor

    def _get_masks(self, cluster_id):
        spike_ids = self.selector(self.n_spikes_waveforms, [cluster_id])
        if self.model.all_masks is None:
            return np.ones((self.n_spikes_waveforms, self.model.n_channels))
        return self.model.all_masks[spike_ids]

    def _get_mean_masks(self, cluster_id):
        return np.mean(self._get_masks(cluster_id), axis=0)

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        pos = self.model.channel_positions
        spike_ids = self.selector(self.n_spikes_waveforms, [cluster_id])
        data = self.model.all_waveforms[spike_ids]
        mm = self._get_mean_masks(cluster_id)
        mw = np.mean(data, axis=0)
        amp = get_waveform_amplitude(mm, mw)
        masks = self._get_masks(cluster_id)
        # Find the best channels.
        channel_ids = np.argsort(amp)[::-1]
        return Bunch(
            data=data[..., channel_ids],
            channel_ids=channel_ids,
            channel_positions=pos[channel_ids],
            masks=masks[:, channel_ids],
        )

    def _get_mean_waveforms(self, cluster_id):
        b = self._get_waveforms(cluster_id).copy()
        b.data = np.mean(b.data, axis=0)[np.newaxis, ...]
        b.masks = np.mean(b.masks, axis=0)[np.newaxis, ...] ** .1
        b['alpha'] = 1.
        return b

    # Public methods
    # -------------------------------------------------------------------------

    def get_best_channels(self, cluster_id):
        """Get the best channels of a given cluster."""
        mm = self._get_mean_masks(cluster_id)
        channel_ids = np.argsort(mm)[::-1]
        ind = mm[channel_ids] > .1
        if np.sum(ind) > 0:
            channel_ids = channel_ids[ind]
        else:  # pragma: no cover
            channel_ids = channel_ids[:4]
        return channel_ids

    def get_channel_amplitudes(self, cluster_id):
        """Return the channel amplitudes of the best channels of a given cluster."""
        channel_ids = self.get_best_channels(cluster_id)
        return channel_ids, np.ones(len(channel_ids))

    def on_save_clustering(self, sender, spike_clusters, groups, *labels):
        """Save the modified data."""
        groups = {c: g.title() for c, g in groups.items()}
        self.model.save(spike_clusters, groups)
        self._save_cluster_info()


#------------------------------------------------------------------------------
# Kwik commands
#------------------------------------------------------------------------------

def kwik_gui(path, channel_group=None, clustering=None, **kwargs):  # pragma: no cover
    """Launch the Kwik GUI."""
    assert path
    create_app()
    controller = KwikController(
        path, channel_group=channel_group, clustering=clustering, **kwargs)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()


def kwik_describe(path, channel_group=None, clustering=None):
    """Describe a template dataset."""
    assert path
    KwikModel(path, channel_group=channel_group, clustering=clustering).describe()
