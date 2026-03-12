# phy 2.1.0rc1

With substantial help from AI-assisted development, it has been possible to put time and effort
into this maintenance release for the current 2.x line.

`phy 2.1.0rc1` is focused on making the existing software work better for users. It is not meant
to bring in major new features or feature requests at this stage. More work will likely still be
needed based on user feedback during the release candidate period.

## Main points

* No changes to the dataset or file formats
* Dependency updates and packaging cleanup
* Replacement of a fragile legacy web-based GUI component with a Qt-native implementation
* Expected improvement on systems where the old embedded web component caused blank panes, white windows, or related display failures

## What to test

* Installation on current Linux and Windows environments
* GUI startup and rendering behavior
* Cluster selection and view updates
* Feature, waveform, amplitude, and trace views
* Remote desktop, Wayland, and GPU-specific setups
* Plugin-based workflows

## Compatibility notes

* Dataset formats are unchanged
* Some plugins relying on internal HTML or other web-based GUI components may need updates

## Notes for plugin maintainers

The main compatibility risk for plugins is on the GUI side. The legacy web-based component has
been replaced with a Qt-native implementation, so plugins depending on internal HTML or other
web-based GUI pieces may need to be updated.

Plugins using supported Python-side controller, event, or view APIs are more likely to keep
working unchanged, but they should still be tested.

## Testing window

Testing for `2.1.0rc1` is expected to stay open for at least the next couple of months before a final `2.1.0` release.

## Feedback

When reporting issues, please include:

* operating system
* Python version
* installation method
* local or remote session details
* whether plugins are in use
* a minimal error message or reproduction if possible
