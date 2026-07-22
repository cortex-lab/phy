"""Recluster action based on the PC features.

This is the Template GUI counterpart of the Kwik GUI's `recluster` action. No
spike detection is re-run: the spikes and their PC features already exist in the
sorting output, and only the cluster assignment is recomputed.

The default algorithm is ISO-SPLIT, the one MountainSort uses, if the `isosplit`
package is installed. It is non-parametric: it decides how many subclusters there
are by testing each candidate split for unimodality, so it does not assume the
clusters are Gaussian. Bursting and drifting units are not, which is why a
Gaussian mixture tends to cut them in half. The mixture is kept as a fallback and
for the case where you want to impose the number of subclusters yourself.

This plugin is bundled with phy and attached by default (see
`TemplateController.default_plugins`). Its dependencies are optional and install
with `pip install "phy[recluster]"`; without them the actions are not added and a
message says how to get them.
"""

import importlib.util
import logging

import numpy as np

from phy import IPlugin, connect

logger = logging.getLogger('phy')

# Number of subclusters tried when the count is chosen automatically by the mixture.
MAX_CLUSTERS = 8
# The GMM is fit on at most this many spikes, then applied to all of them.
MAX_SPIKES_FIT = 20000
# Features are reduced to at most this many dimensions before clustering.
MAX_DIMS = 10
# Passed to isosplit(). LOWERING it yields more subclusters -- note that this is the
# opposite of what the isosplit 0.1.4 docstring claims. Lowering it is a blunt
# instrument: on synthetic data, 1.5 already splits a genuinely single unit in two,
# and the count is not monotonic in the threshold. To split a cluster ISO-SPLIT
# considers unimodal, imposing the number of subclusters is the controlled option.
DIP_THRESHOLD = 2.0


def _has(name):
    """Whether a module is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def _reduce(x):
    """Cut the features down to a few dimensions.

    Deliberately not whitened. Only the leading components carry the cluster
    separation; the rest are noise, and rescaling them all to unit variance
    amplifies the noise directions until the separation is buried.
    """
    from sklearn.decomposition import PCA

    n_components = min(MAX_DIMS, x.shape[1], x.shape[0])
    return PCA(n_components=n_components, whiten=False, random_state=0).fit_transform(x)


def _isosplit(x, dip_threshold=None):
    """Cluster an array with ISO-SPLIT. Returns None if the package is missing."""
    try:
        from isosplit import isosplit
    except ImportError:
        return None
    dip = DIP_THRESHOLD if dip_threshold is None else dip_threshold
    # isosplit() labels from 1; phy wants them relative to the selection.
    return isosplit(x, dip_threshold=dip) - 1


def _fit_predict(x, n_clusters=None):
    """Cluster an array with a Gaussian mixture, choosing the number of components by BIC."""
    from sklearn.mixture import GaussianMixture

    # Fit on a subsample of large clusters, so that the action stays responsive.
    if len(x) > MAX_SPIKES_FIT:
        rng = np.random.default_rng(0)
        x_fit = x[rng.choice(len(x), MAX_SPIKES_FIT, replace=False)]
    else:
        x_fit = x

    def _gmm(n):
        return GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(x_fit)

    if n_clusters:
        best = _gmm(n_clusters)
    else:
        # The number of components is capped by the sample size, as a full covariance
        # matrix cannot be estimated from fewer spikes than dimensions.
        max_clusters = max(1, min(MAX_CLUSTERS, len(x_fit) // (x.shape[1] + 1)))
        candidates = [_gmm(n) for n in range(1, max_clusters + 1)]
        best = min(candidates, key=lambda gmm: gmm.bic(x_fit))
        logger.info('Selected %d subclusters by BIC.', best.n_components)

    return best.predict(x)


class ExampleReclusterPlugin(IPlugin):
    def attach_to_controller(self, controller):
        # scikit-learn drives both the PCA reduction and the mixture fallback, so
        # without it there is no working code path: skip rather than register
        # actions that would raise when pressed.
        if not _has('sklearn'):
            logger.debug(
                'scikit-learn is not installed: the recluster actions are disabled. '
                'Install them with `pip install "phy[recluster]"`.'
            )
            return

        def _get_spike_ids(cluster_ids):
            """Return all spikes of the selected clusters that have features."""
            spike_ids = controller.supervisor.clustering.spikes_in_clusters(cluster_ids)
            # Some models only store features for a subset of the spikes.
            features_rows = getattr(controller.model, 'features_rows', None)
            if features_rows is not None:  # pragma: no cover
                spike_ids = np.intersect1d(spike_ids, features_rows)
            return spike_ids

        def _get_channel_ids(cluster_ids):
            """Return the union of the best channels of the selected clusters."""
            channel_ids = [controller.get_best_channels(c) for c in cluster_ids]
            return np.unique(np.concatenate(channel_ids))

        def _recluster(n_clusters=None, dip_threshold=None):
            cluster_ids = controller.supervisor.selected
            if not cluster_ids:
                logger.warning('No cluster selected, cannot recluster.')
                return

            spike_ids = _get_spike_ids(cluster_ids)
            if len(spike_ids) < 2:
                logger.warning('Not enough spikes with features, cannot recluster.')
                return

            # Same features as the ones the feature view is showing. We call
            # _get_spike_features() rather than _get_features(), because the latter is
            # disk-cached and we would be caching every cluster we ever recluster.
            channel_ids = _get_channel_ids(cluster_ids)
            bunch = controller._get_spike_features(spike_ids, channel_ids)

            # (n_spikes, n_channels, n_pcs) -> (n_spikes, n_channels * n_pcs)
            x = _reduce(bunch.data.reshape((len(spike_ids), -1)))
            logger.info('Reclustering %d spikes on %d channels.', len(spike_ids), len(channel_ids))

            # ISO-SPLIT picks the number of subclusters itself, so it cannot honour an
            # explicit n_clusters: fall back to the mixture in that case.
            labels = None if n_clusters else _isosplit(x, dip_threshold=dip_threshold)
            if labels is None:
                labels = _fit_predict(x, n_clusters=n_clusters)
            assert labels.shape == spike_ids.shape

            if len(np.unique(labels)) == 1:
                logger.info(
                    'Reclustering found a single cluster, nothing to split. Lower the dip '
                    'threshold, or impose the number of subclusters, to split it anyway.'
                )
                return
            controller.supervisor.actions.split(spike_ids, labels)

        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='alt+k', set_busy=True)
            def recluster():
                """Recluster the selected clusters with ISO-SPLIT, which picks the number
                of subclusters itself."""
                _recluster()

            @controller.supervisor.actions.add(
                shortcut='shift+alt+k',
                set_busy=True,
                prompt=True,
                n_args=1,
                prompt_default=lambda: 2,
            )
            def recluster_n(n_clusters):
                """Recluster the selected clusters into a given number of subclusters,
                using a Gaussian mixture."""
                _recluster(n_clusters=int(n_clusters))

            @controller.supervisor.actions.add(
                shortcut='ctrl+alt+k',
                set_busy=True,
                prompt=True,
                n_args=1,
                prompt_default=lambda: DIP_THRESHOLD,
            )
            def recluster_dip(dip_threshold):
                """Recluster with ISO-SPLIT at a given dip threshold. Lower it below the
                default of 2 to make it split more readily."""
                _recluster(dip_threshold=float(dip_threshold))
