"""Bounded template loading without dense caches indexed by cluster namespace."""

from types import MappingProxyType

import numpy as np
from phylib.io.model import TemplateModel


class BoundedTemplateModel(TemplateModel):
    """Load committed assignments and original templates as separate concepts.

    The controller derives live unit evidence from Supervisor assignments.
    phylib's eager merged-cluster waveform cache allocates by the maximum
    absolute ID and does not track subsequent GUI edits; bounded mode does not
    expose that cache as current scientific evidence.
    """

    def __init__(self, *args, session, **kwargs):
        self._bounded_session = session
        super().__init__(*args, **kwargs)
        self.sparse_clusters = None

    def freeze_source(self):
        """Prevent public model edits from bypassing semantic curation."""
        names = set()
        for name, value in vars(self).items():
            if not name.startswith('_') and isinstance(value, np.ndarray):
                value.flags.writeable = False
                names.add(name)
        self.metadata = MappingProxyType(
            {name: MappingProxyType(values) for name, values in self.metadata.items()}
        )
        names.update(
            ('metadata', 'save_spike_clusters', 'save_metadata', 'save_spikes_subset_waveforms')
        )
        self._protected_source_names = frozenset(names)

    def __setattr__(self, name, value):
        if name in getattr(self, '_protected_source_names', ()):
            raise ValueError('Bounded model source is read-only; use semantic curation commands.')
        super().__setattr__(name, value)

    def _load_spike_clusters(self):
        return self._bounded_session.assignments.copy()

    def get_merge_map(self):
        return {}, np.array([], dtype=np.int64)

    def cluster_waveforms(self):
        return None

    def get_cluster_mean_waveforms(self, *args, **kwargs):
        raise ValueError('Use the controller waveform/template providers for live bounded units.')
