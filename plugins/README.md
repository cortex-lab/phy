# phy plugin examples

To enable any of these, phy needs both the plugin *directory* (`c.Plugins.dirs`)
and the plugin *name* (`c.<GUIName>.plugins`) in `~/.phy/phy_config.py`. A ready-made
config that enables `ExampleReclusterPlugin` ships in the repo root as
[phy_config.example.py](../phy_config.example.py) -- copy it to `~/.phy/phy_config.py`
(the plugin path is derived from `phy.__file__`, so no editing is needed). The
recluster plugin's extra dependencies install with `pip install -e .[recluster]`.

* [ExampleActionPlugin](action_status_bar.py): Show how to create new actions in the GUI.
* [ExampleClusterMetadataPlugin](cluster_metadata.py): Show how to save the best channel of every cluster in a cluster_channel.tsv file when saving.
* [ExampleClusterMetricsPlugin](cluster_metrics.py): Show how to add a custom cluster metrics.
* [ExampleClusterStatsPlugin](cluster_stats.py): Show how to add a custom cluster histogram view showing cluster statistics.
* [ExampleClusterViewStylingPlugin](cluster_view_styling.py): Show how to customize the styling of the cluster view with Qt stylesheet fragments.
* [ExampleColorSchemePlugin](color_scheme.py): Show how to add a custom color scheme to a view.
* [ExampleCustomButtonPlugin](custom_button.py): Show how to add custom buttons in a view's title bar.
* [ExampleCustomColumnsPlugin](custom_columns.py): Show how to customize the columns in the cluster and similarity views.
* [ExampleCustomFeatureViewPlugin](feature_view_custom_grid.py): Show how to customize the subplot grid specification in the feature view.
* [ExampleCustomSplitPlugin](custom_split.py): Show how to write a custom split action.
* [ExampleFilterFiringRatePlugin](filter_action.py): Show how to create a filter snippet for the cluster view.
* [ExampleFontSizePlugin](font_size.py): Show how to change the default text font size.
* [ExampleHelloPlugin](hello.py): Hello world plugin.
* [ExampleIPythonViewPlugin](ipython_view.py): Show how to injet specific Python variables in the IPython view.
* [ExampleMatplotlibViewPlugin](matplotlib_view.py): Show how to create a custom matplotlib view in the GUI.
* [ExampleNspikesViewsPlugin](n_spikes_views.py): Show how to change the number of displayed spikes in each view.
* [ExampleOpenGLViewPlugin](opengl_view.py): Show how to write a custom OpenGL view. This is for advanced users only.
* [ExampleRawDataFilterPlugin](raw_data_filter.py): Show how to add a custom raw data filter for the TraceView and Waveform View
* [ExampleReclusterPlugin](recluster.py): Show how to add a recluster action based on the PC features, using ISO-SPLIT (the algorithm MountainSort uses) or a Gaussian mixture. This is the Template GUI counterpart of the Kwik GUI's recluster.
* [ExampleSimilarityPlugin](custom_similarity.py): Show how to add a custom similarity measure.
* [ExampleWaveformUMAPPlugin](umap_view.py): Show how to write a custom dimension reduction view.
