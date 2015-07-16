# -*- coding: utf-8 -*-

"""Default settings for spike detection."""


# -----------------------------------------------------------------------------
# Spike detection
# -----------------------------------------------------------------------------

spikedetekt = {
    'filter_low': 500.,
    'filter_high_factor': 0.95 * .5,  # will be multiplied by the sample rate
    'filter_butter_order': 3,

    # Data chunks.
    'chunk_size_seconds': 1.,
    'chunk_overlap_seconds': .015,

    # Threshold.
    'n_excerpts': 50,
    'excerpt_size_seconds': 1.,
    'use_single_threshold': True,
    'threshold_strong_std_factor': 4.5,
    'threshold_weak_std_factor': 2.,
    'detect_spikes': 'negative',

    # Connected components.
    'connected_component_join_size': 1,

    # Spike extractions.
    'extract_s_before': 10,
    'extract_s_after': 10,
    'weight_power': 2,

    # Features.
    'n_features_per_channel': 3,
    'pca_n_waveforms_max': 10000,

    # Waveform filtering in GUI.
    'waveform_filter': True,
    'waveform_dc_offset': None,
    'waveform_scale_factor': None,

}
