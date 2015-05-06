
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
# Clustering GUI
# -----------------------------------------------------------------------------

manual_clustering.keyboard_shortcuts = {
    'reset_gui': 'alt+r',
    'save': 'ctrl+s',
    'undo': 'ctrl+z',
    'redo': ('ctrl+shift+z', 'ctrl+y'),
    'exit': 'ctrl+q',
    'show_shortcuts': 'h',
    'reset_wizard': 'ctrl+w',
    'next': 'space',
    'previous': 'shift+space',
    'first': 'home',
    'last': 'end',
    'pin': 'return',
    'unpin': 'backspace',
    'merge': 'g',
    'move_best_to_noise': 'alt+n',
    'move_best_to_mua': 'alt+m',
    'move_best_to_good': 'alt+g',
    'move_match_to_noise': 'ctrl+n',
    'move_match_to_mua': 'ctrl+m',
    'move_match_to_good': 'ctrl+g',
    'move_both_to_noise': 'ctrl+alt+n',
    'move_both_to_mua': 'ctrl+alt+m',
    'move_both_to_good': 'ctrl+alt+g',
}


# -----------------------------------------------------------------------------
# Internal settings
# -----------------------------------------------------------------------------

# Number of spikes to load at once from the features_masks array
# during the cluster store generation.
manual_clustering.store_chunk_size = 100000
