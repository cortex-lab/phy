# -*- coding: utf-8 -*-

"""Default settings for manual sorting."""


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

# Load regularly-spaced waveforms.
waveforms_excerpt_size = None

# Maximum number of spikes to display in the feature view.
features_n_spikes_max = 2500

# Load a regular subselection of spikes from the cluster store.
features_excerpt_size = None

# Maximum number of background spikes to display in the feature view.
features_n_spikes_max_bg = features_n_spikes_max

features_grid_n_spikes_max = features_n_spikes_max
features_grid_excerpt_size = features_excerpt_size
features_grid_n_spikes_max_bg = features_n_spikes_max_bg


# -----------------------------------------------------------------------------
# Clustering GUI
# -----------------------------------------------------------------------------

cluster_manual_shortcuts = {
    'reset_gui': 'alt+r',
    'show_shortcuts': 'ctrl+h',
    'save': 'ctrl+s',
    'exit': 'ctrl+q',
    # Wizard actions.
    'reset_wizard': 'ctrl+w',
    'next': 'space',
    'previous': 'shift+space',
    'reset_wizard': 'ctrl+alt+space',
    'first': 'home',
    'last': 'end',
    'pin': 'return',
    'unpin': 'backspace',
    # Clustering actions.
    'merge': 'g',
    'split': 'k',
    'undo': 'ctrl+z',
    'redo': ('ctrl+shift+z', 'ctrl+y'),
    'move_best_to_noise': 'alt+n',
    'move_best_to_mua': 'alt+m',
    'move_best_to_good': 'alt+g',
    'move_match_to_noise': 'ctrl+n',
    'move_match_to_mua': 'ctrl+m',
    'move_match_to_good': 'ctrl+g',
    'move_both_to_noise': 'ctrl+alt+n',
    'move_both_to_mua': 'ctrl+alt+m',
    'move_both_to_good': 'ctrl+alt+g',
    # Views.
    'show_view_shortcuts': 'h',
    'toggle_correlogram_normalization': 'n',
    'toggle_waveforms_overlap': 'o',
    'toggle_waveforms_mean': 'm',
    'show_features_time': 't',
}


cluster_manual_config = [
    # The wizard panel is less useful now that there's the stats panel.
    # ('wizard', {'position': 'right'}),
    ('stats', {'position': 'right'}),
    ('features_grid', {'position': 'left'}),
    ('features', {'position': 'left'}),
    ('correlograms', {'position': 'left'}),
    ('waveforms', {'position': 'right'}),
    ('traces', {'position': 'right'}),
]


def _select_clusters(gui, args):
    # Range: '5-12'
    if '-' in args:
        m, M = map(int, args.split('-'))
        # The second one should be included.
        M += 1
        clusters = list(range(m, M))
    # List of ids: '5 6 9 12'
    else:
        clusters = list(map(int, args.split(' ')))
    gui.select(clusters)


cluster_manual_snippets = {
    'c': _select_clusters,
}


# Whether to ask the user if they want to save when the GUI is closed.
prompt_save_on_exit = True


# -----------------------------------------------------------------------------
# Internal settings
# -----------------------------------------------------------------------------

waveforms_scale_factor = .01
features_scale_factor = .01
features_grid_scale_factor = features_scale_factor
traces_scale_factor = .01
