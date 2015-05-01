
# -----------------------------------------------------------------------------
# Correlograms
# -----------------------------------------------------------------------------

# Number of time samples in a bin.
manual_clustering.correlograms_binsize = 20

# Number of bins (odd number).
manual_clustering.correlograms_winsize_bins = 2 * 25 + 1

# Maximum number of spikes for the correlograms.
# Use 'None' to specify an infinite value.
manual_clustering.correlograms_n_spikes_max = 1000000

# Contiguous chunks of spikes for computing the CCGs.
# Use 'None' to have a regular (strided) subselection instead of a chunked
# subselection.
manual_clustering.correlograms_excerpt_size = 100000


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

# Maximum number of spikes to display in the waveform view.
manual_clustering.waveforms_n_spikes_max = 100

# Load contiguous chunks of waveforms (contiguous I/O is faster).
# Higher value = faster loading of waveforms.
manual_clustering.waveforms_excerpt_size = 20

# Maximum number of spikes to display in the feature view.
manual_clustering.features_n_spikes_max = 10000


# -----------------------------------------------------------------------------
# Internal settings
# -----------------------------------------------------------------------------

# Number of spikes to load at once from the features_masks array
# during the cluster store generation.
manual_clustering.store_chunk_size = 100000
