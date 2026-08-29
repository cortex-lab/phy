"""Show how to recluster selected spikes from their PC features."""

import logging

import numpy as np

from phy import IPlugin, connect

logger = logging.getLogger('phy')

MAX_CLUSTERS = 8
MAX_DIMS = 10
MAX_SPIKES_FIT = 20_000
MAX_SPIKES_RECLUSTER = 20_000
DIP_THRESHOLD = 2.0


def _reduce(x):
    """Reduce features to the leading principal components."""
    from sklearn.decomposition import PCA

    n_components = min(MAX_DIMS, x.shape[0], x.shape[1])
    return PCA(n_components=n_components, whiten=False, random_state=0).fit_transform(x)


def _isosplit(x, dip_threshold=None):
    """Cluster with ISO-SPLIT, or return None when it is unavailable."""
    try:
        from isosplit import isosplit
    except ImportError:
        return None
    dip = DIP_THRESHOLD if dip_threshold is None else dip_threshold
    return isosplit(x, dip_threshold=dip) - 1


def _fit_predict(x, n_clusters=None):
    """Cluster with a Gaussian mixture, selecting the component count by BIC."""
    from sklearn.mixture import GaussianMixture

    if len(x) > MAX_SPIKES_FIT:
        rng = np.random.default_rng(0)
        x_fit = x[rng.choice(len(x), MAX_SPIKES_FIT, replace=False)]
    else:
        x_fit = x

    def _gmm(n):
        return GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(x_fit)

    if n_clusters is not None:
        best = _gmm(n_clusters)
    else:
        max_clusters = max(1, min(MAX_CLUSTERS, len(x_fit) // (x.shape[1] + 1)))
        candidates = [_gmm(n) for n in range(1, max_clusters + 1)]
        best = min(candidates, key=lambda gmm: gmm.bic(x_fit))
        logger.info('Selected %d subclusters by BIC.', best.n_components)
    return best.predict(x)


def _recluster(controller, n_clusters=None, dip_threshold=None):
    """Recluster the current selection and split it when multiple groups are found."""
    cluster_ids = list(controller.supervisor.selected)
    if not cluster_ids:
        logger.warning('No cluster selected, cannot recluster.')
        return
    if getattr(controller.model, 'features', None) is None:
        logger.warning('No PC features are available, cannot recluster.')
        return

    spike_ids = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
    features_rows = getattr(controller.model, 'features_rows', None)
    if features_rows is not None:
        spike_ids = np.intersect1d(spike_ids, features_rows)
    if len(spike_ids) < 2:
        logger.warning('Not enough spikes with features, cannot recluster.')
        return
    if len(spike_ids) > MAX_SPIKES_RECLUSTER:
        logger.warning(
            'The selection has %d spikes; reclustering is limited to %d.',
            len(spike_ids),
            MAX_SPIKES_RECLUSTER,
        )
        return

    if n_clusters is not None:
        try:
            n_clusters = int(n_clusters)
        except (TypeError, ValueError):
            logger.warning('The number of subclusters must be an integer.')
            return
        if not 2 <= n_clusters <= len(spike_ids):
            logger.warning('The number of subclusters must be between 2 and %d.', len(spike_ids))
            return
    if dip_threshold is not None:
        try:
            dip_threshold = float(dip_threshold)
        except (TypeError, ValueError):
            logger.warning('The dip threshold must be a positive number.')
            return
        if not np.isfinite(dip_threshold) or dip_threshold <= 0:
            logger.warning('The dip threshold must be a positive number.')
            return

    channels = [np.asarray(controller.get_best_channels(c)) for c in cluster_ids]
    channels = [channel_ids for channel_ids in channels if len(channel_ids)]
    if not channels:
        logger.warning('No channels are available, cannot recluster.')
        return
    channel_ids = np.unique(np.concatenate(channels))
    bunch = controller._get_spike_features(spike_ids, channel_ids)
    data = getattr(bunch, 'data', None)
    if data is None or not np.size(data):
        logger.warning('No PC features are available, cannot recluster.')
        return

    x = np.asarray(data).reshape((len(spike_ids), -1))
    if not np.all(np.isfinite(x)):
        logger.warning('PC features contain invalid values, cannot recluster.')
        return
    x = _reduce(x)
    logger.info('Reclustering %d spikes on %d channels.', len(spike_ids), len(channel_ids))

    labels = None if n_clusters is not None else _isosplit(x, dip_threshold=dip_threshold)
    if labels is None:
        labels = _fit_predict(x, n_clusters=n_clusters)
    labels = np.asarray(labels)
    if labels.shape != spike_ids.shape:
        logger.warning('The clustering algorithm returned an invalid assignment.')
        return
    if len(np.unique(labels)) < 2:
        logger.info('Reclustering found a single cluster, nothing to split.')
        return
    controller.supervisor.actions.split(spike_ids, labels)


class ExampleReclusterPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='alt+k', set_busy=True)
            def recluster():
                """Recluster with ISO-SPLIT, falling back to a Gaussian mixture."""
                _recluster(controller)

            @controller.supervisor.actions.add(
                shortcut='shift+alt+k',
                set_busy=True,
                prompt=True,
                n_args=1,
                prompt_default=lambda: 2,
            )
            def recluster_n(n_clusters):
                """Recluster into a requested number of subclusters."""
                _recluster(controller, n_clusters=n_clusters)

            @controller.supervisor.actions.add(
                shortcut='ctrl+alt+k',
                set_busy=True,
                prompt=True,
                n_args=1,
                prompt_default=lambda: DIP_THRESHOLD,
            )
            def recluster_dip(dip_threshold):
                """Recluster with a requested ISO-SPLIT dip threshold."""
                _recluster(controller, dip_threshold=dip_threshold)
