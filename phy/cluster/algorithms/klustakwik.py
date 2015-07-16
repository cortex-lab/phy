# -*- coding: utf-8 -*-

"""Wrapper to KlustaKwik2 implementation."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ...utils.array import PartialArray
from ...utils.event import EventEmitter
from ...io.kwik.sparse_kk2 import sparsify_features_masks


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

class KlustaKwik(EventEmitter):
    """KlustaKwik automatic clustering algorithm."""
    def __init__(self, **kwargs):
        super(KlustaKwik, self).__init__()
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        # Set the version.
        from klustakwik2 import __version__
        self.version = __version__

    def cluster(self,
                model=None,
                spike_ids=None,
                features=None,
                masks=None,
                ):
        """Run the clustering algorithm on the model, or on any features
        and masks.

        Return the `spike_clusters` assignements.

        Emit the `iter` event at every KlustaKwik iteration.

        """
        # Get the features and masks.
        if model is not None:
            if features is None:
                features = PartialArray(model.features_masks, 0)
            if masks is None:
                masks = PartialArray(model.features_masks, 1)
        # Select some spikes if needed.
        if spike_ids is not None:
            features = features[spike_ids]
            masks = masks[spike_ids]
        # Convert the features and masks to the sparse structure used
        # by KK.
        data = sparsify_features_masks(features, masks)
        data = data.to_sparse_data()
        # Run KK.
        from klustakwik2 import KK
        kk = KK(data, **self._kwargs)

        @kk.register_callback
        def f(_):
            # Skip split iterations.
            if _.name != '':
                return
            self.emit('iter', kk.clusters)

        self.params = kk.all_params
        kk.cluster_mask_starts()
        spike_clusters = kk.clusters
        return spike_clusters


def cluster(model, algorithm='klustakwik', spike_ids=None, **kwargs):
    """Launch an automatic clustering algorithm on the model.

    Parameters
    ----------

    model : BaseModel
        A model.
    algorithm : str
        Only 'klustakwik' is supported currently.
    **kwargs
        Parameters for KK.

    """
    assert algorithm == 'klustakwik'
    kk = KlustaKwik(**kwargs)
    return kk.cluster(model=model, spike_ids=spike_ids)
