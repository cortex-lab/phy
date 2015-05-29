# -*- coding: utf-8 -*-

"""Automatic clustering launcher."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..utils.array import PartialArray
from ..io.kwik.sparse_kk2 import sparsify_features_masks


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

def run_klustakwik2(model, ipp_view=None, **kwargs):
    from klustakwik2 import KK
    num_starting_clusters = kwargs.pop('num_starting_clusters')
    f = PartialArray(model.features_masks, 0)
    m = PartialArray(model.features_masks, 1)
    data = sparsify_features_masks(f, m)
    data = data.to_sparse_data()
    # TODO: pass ipp_view when KK2 supports it
    kk = KK(data, **kwargs)
    kk.cluster_mask_starts(num_starting_clusters)
    spike_clusters = kk.clusters
    return spike_clusters


def run(model, algorithm='klustakwik2', ipp_view=None, **kwargs):
    """Launch an automatic clustering algorithm on the model.

    Parameters
    ----------

    model : BaseModel
        A model.
    algorithm : str
        Only 'klustakwik2' is supported currently.
    ipp_view : `IPython.parallel.Client` instance
        Use this to run the algorithm with IPython.

    """
    return globals()['run_{}'.format(algorithm)](model, ipp_view, **kwargs)
