"""Show how to rebind or disable existing GUI keyboard shortcuts."""

from phy import IPlugin, connect
from phy.gui.actions import _get_qkeysequence


# Keys are ``Actions group.action name``. Press H in phy to see action names;
# the group names used by the standard GUI include File, View, Help, Edit, and Select.
SHORTCUTS = {
    'Edit.merge': 'ctrl+shift+m',
    'Select.select_first_similar': None,  # Disable this keyboard shortcut.
}


def _sequences(shortcut):
    """Return a flat list of QKeySequence objects for an Actions shortcut value."""
    if shortcut is None:
        return []
    sequences = _get_qkeysequence(shortcut)
    return sequences if isinstance(sequences, list) else [sequences]


def _sequence_names(shortcut):
    """Return normalized sequence names for conflict detection."""
    return {
        str(sequence.toString()).lower()
        for sequence in _sequences(shortcut)
        if not sequence.isEmpty()
    }


class ExampleShortcutsPlugin(IPlugin):
    """Rebind existing actions after the GUI and all default views are ready."""

    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            requested = {}

            # Resolve every target before changing anything, so a typo cannot leave
            # the GUI half configured.
            for qualified_name, shortcut in SHORTCUTS.items():
                try:
                    group_name, action_name = qualified_name.split('.', 1)
                except ValueError as e:
                    raise ValueError(
                        f"Shortcut target {qualified_name!r} must be 'Group.action'."
                    ) from e
                matches = [
                    group
                    for group in gui.actions
                    if group.name == group_name and group.get(action_name) is not None
                ]
                if not matches:
                    available = sorted(
                        f'{candidate.name}.{name}'
                        for candidate in gui.actions
                        for name in candidate._actions_dict
                    )
                    raise ValueError(
                        f"Unknown shortcut target {qualified_name!r}. "
                        f"Available targets: {', '.join(available)}"
                    )
                # Multiple instances of the same view have identically named action
                # groups. Only the first instance receives that view's shortcuts.
                group = matches[0]
                requested[qualified_name] = (group, action_name, shortcut)

            # Detect duplicate requested shortcuts.
            claimed = {}
            for qualified_name, (_, _, shortcut) in requested.items():
                for sequence in _sequence_names(shortcut):
                    if sequence in claimed:
                        raise ValueError(
                            f"Shortcut {sequence!r} is requested for both "
                            f"{claimed[sequence]!r} and {qualified_name!r}."
                        )
                    claimed[sequence] = qualified_name

            # Reject collisions with actions that are not being changed.
            targets = {
                (id(group), action_name)
                for group, action_name, _ in requested.values()
            }
            for group in gui.actions:
                for action_name, action in group._actions_dict.items():
                    qualified_name = f'{group.name}.{action_name}'
                    if (id(group), action_name) in targets:
                        continue
                    existing = {
                        str(sequence.toString()).lower()
                        for sequence in action.qaction.shortcuts()
                        if not sequence.isEmpty()
                    }
                    conflicts = existing & set(claimed)
                    if conflicts:
                        sequence = sorted(conflicts)[0]
                        raise ValueError(
                            f"Shortcut {sequence!r} for {claimed[sequence]!r} "
                            f"conflicts with {qualified_name!r}."
                        )

            # Clear all targets only after validation, so swapping two shortcuts is
            # possible and an invalid configuration leaves every binding unchanged.
            for group, action_name, _ in requested.values():
                group.get(action_name).setShortcuts([])

            # Change both the QAction and phy's action metadata. Updating the
            # metadata keeps the shortcut list displayed with H accurate.
            for group, action_name, shortcut in requested.values():
                action = group._actions_dict[action_name]
                action.qaction.setShortcuts(_sequences(shortcut))
                action.shortcut = shortcut
