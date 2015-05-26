
# -----------------------------------------------------------------------------
# Correlograms
# -----------------------------------------------------------------------------

# Number of time samples in a bin.
correlograms_binsize = 20

# Number of bins (odd number).
correlograms_winsize_bins = 2 * 25 + 1

# Maximum number of spikes for the correlograms.
# Use `None` to specify an infinite value.
correlograms_n_spikes_max = 1000000

# Contiguous chunks of spikes for computing the CCGs.
# Use `None` to have a regular (strided) subselection instead of a chunked
# subselection.
correlograms_excerpt_size = 100000


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

# Maximum number of spikes to display in the waveform view.
waveforms_n_spikes_max = 100

# Load contiguous chunks of waveforms (contiguous I/O is faster).
# Higher value = faster loading of waveforms.
waveforms_excerpt_size = 20

# Maximum number of spikes to display in the feature view.
features_n_spikes_max = 2500

# Load a regular subselection of spikes from the cluster store.
features_excerpt_size = None

# Maximum number of background spikes to display in the feature view.
features_n_spikes_max_bg = features_n_spikes_max


# -----------------------------------------------------------------------------
# Clustering GUI
# -----------------------------------------------------------------------------

keyboard_shortcuts = {
    'reset_gui': 'alt+r',
    'save': 'ctrl+s',
    'undo': 'ctrl+z',
    'redo': ('ctrl+shift+z', 'ctrl+y'),
    'exit': 'ctrl+q',
    'show_shortcuts': 'ctrl+h',
    'reset_wizard': 'ctrl+w',
    'next': 'space',
    'previous': 'shift+space',
    'reset_wizard': 'ctrl+alt+space',
    'first': 'home',
    'last': 'end',
    'pin': 'return',
    'unpin': 'backspace',
    'merge': 'g',
    'split': 'k',
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


gui_config = [
    ('wizard', {'position': 'right'}),
    ('features', {'position': 'left'}),
    ('correlograms', {'position': 'left'}),
    ('waveforms', {'position': 'right'}),
    ('traces', {'position': 'right'}),
]

# Whether to ask the user if they want to save when the GUI is closed.
prompt_save_on_exit = True


# -----------------------------------------------------------------------------
# Store settings
# -----------------------------------------------------------------------------

# Number of spikes to load at once from the features_masks array
# during the cluster store generation.
features_masks_chunk_size = 100000


# -----------------------------------------------------------------------------
# Internal settings
# -----------------------------------------------------------------------------

waveforms_scale_factor = .01
features_scale_factor = .01
traces_scale_factor = .01
