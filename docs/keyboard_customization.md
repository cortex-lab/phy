# Customize keyboard shortcuts

Press `H` or choose **Help > Show all shortcuts** to print the effective
shortcuts and snippets in the console. The generated
[keyboard shortcut reference](shortcuts.md) lists the defaults, but `H` is the
better source after plugins have changed them.

Keyboard shortcuts invoke actions immediately. Snippets start with `:`, may
take arguments, and run when you press Enter. For example, `:c 10 12` selects
clusters 10 and 12. Rebinding an action does not change its snippet alias.

## Install the example

Copy
[`plugins/shortcuts.py`](https://github.com/cortex-lab/phy/blob/master/plugins/shortcuts.py)
to `~/.phy/plugins/shortcuts.py`, then enable it in
`~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['ExampleShortcutsPlugin']
```

Edit the `SHORTCUTS` mapping near the top of the plugin:

```python
SHORTCUTS = {
    'Edit.merge': 'ctrl+shift+m',
    'Select.select_first_similar': None,
}
```

Keys use `Actions group.action name`. Standard group names include `File`,
`View`, `Help`, `Edit`, and `Select`; views also have their own action groups.
Action names are the identifiers printed by `H`. Setting a value to `None`
disables only that action's keyboard shortcut; the menu item, callable action,
and snippet remain available.

Restart phy after editing the file. The example validates every target before
making changes, rejects shortcut collisions, supports swapping two configured
shortcuts, and updates the list printed by `H`. An invalid group, action name,
or shortcut is reported in the console and phy keeps its default bindings.

## Multiple key sequences

Use a tuple or list when either sequence should invoke the same action:

```python
SHORTCUTS = {
    'Edit.redo': ('ctrl+shift+z', 'ctrl+y'),
}
```

Avoid assigning a plain character that is used while typing in a filter,
prompt, or snippet. Prefer a modified sequence for custom global actions.

## Conflicts

Qt may allow two actions to hold the same shortcut, but the result is ambiguous
and can depend on focus. The example therefore stops when a requested sequence
is already used by an action outside the mapping. Rebind or disable the other
action in the same mapping to resolve the conflict.

Operating-system and desktop shortcuts take precedence before phy receives a
key event. If a valid binding does nothing, check system keyboard settings as
well as the conflicts reported by the plugin.

## macOS modifier names

Qt translates logical and physical modifiers on macOS:

- `ctrl+...` normally represents the standard macOS Command shortcut;
- use `meta+...` when you specifically need the physical Control key.

For example, phy defines the physical **Control+Space** similarity-selection
shortcut as `meta+space` on macOS. macOS may reserve physical Control+Space for
switching input sources; remap that system shortcut if you want phy to receive
it. Always verify the result with `H` on the target platform.

## Compatibility note

Adding a new action is part of phy's public plugin interface. Rebinding an
already-created action currently requires the example to update phy's internal
action metadata as well as its Qt action so that the Help output stays correct.
Keep the example from the same phy version as your installation, and retest it
when upgrading. A future declarative shortcut API may replace this helper.
