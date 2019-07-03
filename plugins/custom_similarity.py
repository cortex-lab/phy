"""Show how to add a custom similarity measure."""

from operator import itemgetter
import numpy as np

from phy import IPlugin
from phy.apps.template import from_sparse


def _dot_product(mw1, c1, mw2, c2):
    """Compute the L2 dot product between the mean waveforms of two clusters, given in sparse
    format."""

    mw1 = mw1[0, ...]  # first dimension has only 1 element.
    mw2 = mw2[0, ...]
    assert mw1.ndim == 2  # (n_samples, n_channels_loc_1)
    assert mw2.ndim == 2  # (n_samples, n_channels_loc_2)

    # We normalize the waveforms.
    mw1 /= np.sqrt(np.sum(mw1 ** 2))
    mw2 /= np.sqrt(np.sum(mw2 ** 2))

    # We find the union of the channel ids for both clusters so that we can convert from sparse
    # to dense format.
    channel_ids = np.union1d(c1, c2)

    # We directly return 0 if the channels of the two clusters are disjoint.
    if not len(np.intersect1d(c1, c2)):
        return 0

    # We tile the channels so as to use `from_sparse()`.
    c1 = np.tile(c1, (mw1.shape[0], 1))
    c2 = np.tile(c2, (mw2.shape[0], 1))

    # We convert from sparse to dense format in order to compute the distance.
    mw1 = from_sparse(mw1, c1, channel_ids)  # (n_samples, n_channel_locs_common)
    mw2 = from_sparse(mw2, c2, channel_ids)  # (n_samples, n_channel_locs_common)

    # We compute the dot product.
    return np.sum(mw1 * mw2)


class ExampleSimilarityPlugin(IPlugin):
    def attach_to_controller(self, controller):

        # We cache this function in memory and on disk.
        @controller.context.memcache
        def mean_waveform_similarity(cluster_id):
            """This function returns a list of pairs `(other_cluster_id, similarity)` sorted
            by decreasing similarity."""

            # We get the cluster's mean waveforms.
            mw = controller._get_mean_waveforms(cluster_id)
            mean_waveforms, channel_ids = mw.data, mw.channel_ids

            assert mean_waveforms is not None

            out = []
            # We go through all clusters except the currently selected one.
            for cl in controller.supervisor.clustering.cluster_ids:
                if cl == cluster_id:
                    continue
                mw = controller._get_mean_waveforms(cl)
                assert mw is not None
                # We compute the dot product between the current cluster and the other cluster.
                d = _dot_product(mean_waveforms, channel_ids, mw.data, mw.channel_ids)
                out.append((cl, d))  # convert from distance to similarity with a minus sign

            # We return the similar clusters by decreasing similarity.
            return sorted(out, key=itemgetter(1), reverse=True)

        # We add the similarity function.
        controller.similarity_functions['mean_waveform'] = mean_waveform_similarity

        # We set the similarity function to the newly-defined one.
        controller.similarity = 'mean_waveform'
