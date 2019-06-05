# Keyboard shortcuts and snippets

This page presents the list of shortcuts and snippets in the template GUI. You can also display them in the console by pressing `F1`.

## Snippets

Complex actions cannot be easily bound to a keyboard shortcut as they may require parameters. For these, you can use **keyboard snippets**. Snippets are activated by first pressing the `:` key, typing the snippet name followed by some parameters, and pressing `Enter`.

For example, to change the window size in the Correlogram View (`:cw` snippet as shown in the list below):

1. Activate the snippet mode by typing `:` on the keyboard. You can see the snippet bar at the bottom of the GUI.
2. Type `cw 200` to change the window to 200 ms.
3. Press Enter.

To cancel, press Escape to leave the snippet mode.

![image](https://user-images.githubusercontent.com/1942359/58952151-3cb5cb00-8793-11e9-9ace-f941891448dc.png)


## List of keyboard shortcuts and snippets

```
Keyboard shortcuts for TemplateGUI - File
- exit                                     ctrl+q
- save                                     ctrl+s

Keyboard shortcuts for TemplateGUI - Help
- about                                    ?
- show_all_shortcuts                       helpcontents, h

Keyboard shortcuts for TemplateGUI - Snippets
- enable_snippet_mode                      :

Keyboard shortcuts for TemplateGUI - Clustering
- Color field: amplitude                   - (:cfam)
- Color field: channel                     - (:cfch)
- Color field: cluster                     - (:cfcl)
- Color field: depth                       - (:cfde)
- Color field: group                       - (:cfgr)
- Color field: n_spikes                    - (:cfn_)
- Colormap: categorical                    - (:cmca)
- Colormap: diverging                      - (:cmdi)
- Colormap: linear                         - (:cmli)
- Colormap: rainbow                        - (:cmra)
- filter                                   - (:f)
- label                                    l
- merge                                    g
- move                                     - (:move)
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
- reset_wizard                             - (:reset_wizard)
- select                                   - (:c)
- sort                                     - (:s)
- split                                    k
- split_init                               shift+ctrl+k
- toggle_categorical_colormap              - (:toggle_categorical_colormap)
- undo                                     ctrl+z

Keyboard shortcuts for TemplateGUI - AmplitudeHistogramView
- set_n_bins                               - (:an)
- set_x_max                                - (:am)

Keyboard shortcuts for TemplateGUI - AmplitudeView

Keyboard shortcuts for TemplateGUI - CorrelogramView
- set_bin                                  - (:cb)
- set_refractory_period                    - (:cr)
- set_window                               - (:cw)
- toggle_normalization                     n

Keyboard shortcuts for TemplateGUI - FeatureView
- clear_channels                           - (:clear_channels)
- decrease                                 ctrl+-
- increase                                 ctrl++
- toggle_automatic_channel_selection       c

Keyboard shortcuts for TemplateGUI - FiringRateView
- set_n_bins                               - (:fn)
- set_x_max                                - (:fm)

Keyboard shortcuts for TemplateGUI - ISIView
- set_n_bins                               - (:in)
- set_x_max                                - (:im)

Keyboard shortcuts for TemplateGUI - RasterView
- decrease                                 ctrl+shift+-
- increase                                 ctrl+shift++

Keyboard shortcuts for TemplateGUI - TemplateFeatureView

Keyboard shortcuts for TemplateGUI - TemplateView
- decrease                                 ctrl+alt+-
- increase                                 ctrl+alt++

Keyboard shortcuts for TemplateGUI - TraceView
- decrease                                 alt+down
- go_left                                  alt+left
- go_right                                 alt+right
- go_to                                    alt+t
- go_to_next_spike                         alt+pgdown
- go_to_previous_spike                     alt+pgup
- increase                                 alt+up
- narrow                                   alt++
- shift                                    - (:ts)
- toggle_highlighted_spikes                alt+s
- toggle_show_labels                       alt+l
- widen                                    alt+-

Keyboard shortcuts for TemplateGUI - WaveformView
- decrease                                 ctrl+down
- extend_horizontally                      shift+right
- extend_vertically                        shift+up
- increase                                 ctrl+up
- narrow                                   ctrl+left
- shrink_horizontally                      shift+left
- shrink_vertically                        shift+down
- toggle_mean_waveforms                    m
- toggle_show_labels                       ctrl+l
- toggle_templates                         w
- toggle_waveform_overlap                  o
- widen                                    ctrl+right

```
