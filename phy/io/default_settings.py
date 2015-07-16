# -*- coding: utf-8 -*-

"""Default settings for I/O."""


# -----------------------------------------------------------------------------
# Traces
# -----------------------------------------------------------------------------

traces = {
    'raw_data_files': [],
    'n_channels': None,
    'dtype': None,
    'sample_rate': None,
}


# -----------------------------------------------------------------------------
# Store settings
# -----------------------------------------------------------------------------

# Number of spikes to load at once from the features_masks array
# during the cluster store generation.
features_masks_chunk_size = 100000
