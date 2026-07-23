# Power-user guide

phy exposes the same controller, model, selection, views, and clustering objects used by its GUI.
This makes it possible to extend a curation workflow without maintaining a separate application.

## Choose the smallest extension point

* Use a saved GUI setting for layout, scaling, bins, and other existing controls.
* Use a small plugin to change limits, shortcuts, columns, colors, filters, or actions.
* Use the IPython View for exploratory access to the current session.
* Use the Python API for reusable analysis outside the GUI.
* Add a custom view only when the information cannot be expressed in an existing view.

The [plugin cookbook](plugins.md) includes examples for cluster metrics, similarity metrics,
statistics, raw-data filters, matplotlib, OpenGL, UMAP, labels, buttons, and custom actions.

## Objects available in the IPython View

Open **View > Add IPythonView**. The console provides:

* `m`: the dataset model;
* `c`: the controller coordinating data and views;
* `s`: the clustering supervisor;
* `connect` and `emit`: the event helpers.

Code in this console runs in the GUI process. Long calculations can freeze the interface, and
mutating internal arrays can corrupt the active session. Prefer read-only exploration and move
repeatable work into a tested plugin or script.

## Supported surface versus internals

Plugins using controller hooks, documented events, `IPlugin`, and public view/action methods are
the most likely to remain compatible. Code that manipulates private attributes, Qt widget
internals, generated HTML, or OpenGL implementation details may require changes between releases.

Consult the [changelog](changelog.md) before upgrading a plugin and test it against a disposable
copy of a dataset.

## Further reference

* [Configuration and state](configuration.md)
* [Command-line reference](cli.md)
* [Data-analysis examples](analysis.md)
* [Customization API overview](customization.md)
* [Python API](api.md)
