# phy 2.1.0rc1 announcement

With substantial help from AI-assisted development, it has been possible to put time and effort
into a new maintenance release for `phy`, and `2.1.0rc1` is now available for testing.

This release candidate does not change the dataset file formats. The main work so far is dependency and packaging modernization, together with replacing a fragile legacy web-based GUI component with a Qt-native implementation.

This should improve reliability on systems where the old embedded web component caused display problems such as blank panes or white windows. Some plugins relying on internal HTML or web-based GUI components may need updates.

For this release, the goal is not to bring in major new features or feature requests. The goal is
mainly to make `phy` work better for users again, and then continue improving it based on feedback
from testing.

We are looking for beta testers over at least the next couple of months, especially on modern Linux and Windows setups, remote sessions, and plugin-based workflows.

Feedback on installation, rendering, view behavior, and plugin compatibility is particularly useful.
