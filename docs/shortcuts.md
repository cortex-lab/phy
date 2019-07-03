# Keyboard shortcuts and snippets

This page presents the list of shortcuts and snippets in the template GUI. You can also display them in the console by pressing `F1`.


## List of keyboard shortcuts

```text
All keyboard shortcuts

Keyboard shortcuts for Clustering
- label                                    l
- merge                                    g
- move_all_to_good                         ctrl+alt+g
- move_all_to_mua                          ctrl+alt+m
- move_all_to_noise                        ctrl+alt+n
- move_all_to_unsorted                     ctrl+alt+u
- move_best_to_good                        alt+g
- move_best_to_mua                         alt+m
- move_best_to_noise                       alt+n
- move_best_to_unsorted                    alt+u
- move_similar_to_good                     ctrl+g
- move_similar_to_mua                      ctrl+m
- move_similar_to_noise                    ctrl+n
- move_similar_to_unsorted                 ctrl+u
- next                                     space
- next_best                                down
- previous                                 shift+space
- previous_best                            up
- redo                                     ctrl+shift+z, ctrl+y
- reset                                    ctrl+alt+space
- split                                    k
- undo                                     ctrl+z

Keyboard shortcuts for GUI
- about                                    ?
- enable_snippet_mode                      :
- exit                                     ctrl+q
- save                                     ctrl+s
- show_all_shortcuts                       h

Keyboard shortcuts for AmplitudeView
- change_marker_size                       ctrl+wheel
- next_amplitude_type                      a
- select_x_dim                             alt+left click
- select_y_dim                             alt+right click

Keyboard shortcuts for CorrelogramView
- change_window_size                       ctrl+wheel

Keyboard shortcuts for FeatureView
- add_lasso_point                          ctrl+click
- change_marker_size                       ctrl+wheel
- stop_lasso                               ctrl+right click
- toggle_automatic_channel_selection       c

Keyboard shortcuts for HistogramView
- change_window_size                       ctrl+wheel

Keyboard shortcuts for ProbeView

Keyboard shortcuts for RasterView
- change_marker_size                       ctrl+wheel
- decrease                                 ctrl+shift+-
- increase                                 ctrl+shift++
- select_cluster                           ctrl+click

Keyboard shortcuts for ScatterView
- change_marker_size                       ctrl+wheel

Keyboard shortcuts for TemplateView
- change_template_size                     ctrl+wheel
- decrease                                 ctrl+alt+-
- increase                                 ctrl+alt++
- select_cluster                           ctrl+click

Keyboard shortcuts for TraceView
- change_trace_size                        ctrl+wheel
- decrease                                 alt+down
- go_left                                  alt+left
- go_right                                 alt+right
- go_to                                    alt+t
- go_to_next_spike                         alt+pgdown
- go_to_previous_spike                     alt+pgup
- increase                                 alt+up
- narrow                                   alt++
- select_spike                             ctrl+click
- switch_origin                            alt+o
- toggle_highlighted_spikes                alt+s
- toggle_show_labels                       alt+l
- widen                                    alt+-

Keyboard shortcuts for WaveformView
- change_box_size                          ctrl+wheel
- decrease                                 ctrl+down
- extend_horizontally                      shift+right
- extend_vertically                        shift+up
- increase                                 ctrl+up
- narrow                                   ctrl+left
- next_waveforms_type                      w
- shrink_horizontally                      shift+left
- shrink_vertically                        shift+down
- toggle_mean_waveforms                    m
- toggle_show_labels                       ctrl+l
- toggle_waveform_overlap                  o
- widen                                    ctrl+right

```

## List of snippets

Complex actions cannot be easily bound to a keyboard shortcut as they may require parameters. For these, you can use **keyboard snippets**. Snippets are activated by first pressing the `:` key, typing the snippet name followed by some parameters, and pressing `Enter`.

For example, to change the window size in the Correlogram View (`:cw` snippet as shown in the list below):

1. Activate the snippet mode by typing `:` on the keyboard. You can see the snippet bar at the bottom of the GUI.
2. Type `cw 200` to change the window to 200 ms.
3. Press Enter.

To cancel, press Escape to leave the snippet mode.

![image](https://user-images.githubusercontent.com/1942359/58952151-3cb5cb00-8793-11e9-9ace-f941891448dc.png)


```text
All snippets

Snippets for Selection
- select                                   :c

Snippets for Clustering
- filter                                   :f
- label                                    :l
- merge                                    :g
- select                                   :c
- split                                    :k
- sort                                     :s
- sort by xyz                              :sxy
- color field xyz                          :cfxy
- colormap xyz                             :cmxy

Snippets for CorrelogramView
- set_bin (ms)                             :cb
- set_refractory_period (ms)               :cr
- set_window (ms)                          :cw

Snippets for TraceView
- go_to                                    :tg
- shift                                    :ts

Snippets for FiringRateView
- set n bins                               :frn
- set x min                                :frmin
- set x max                                :frmax

Snippets for ISIView
- set n bins                               :isin
- set x min                                :isimin
- set x max                                :isimax

Snippets for WaveformView
- change number of spikes                  :wn

```

Notes:

* For colormaps and sorting in the cluster view, `xy` in the snippets refer to the first two alphabetical characters of the column/field/name. For example, to quickly sort by number of spikes, use the snippet `:sns`. Sorting is in decreasing order when using this snippet.
* For classes deriving from HistogramView, use the `alias_char = 'X'` property to automatically define snippets `Xn`, `Xmin`, `Xmax` to change the number of bins and the histogram range.
