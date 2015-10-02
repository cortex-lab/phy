# -*- coding: utf-8 -*-

"""Actions and snippets."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from six import string_types, PY3

from .qt import QtGui
from phy.utils.event import EventEmitter

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Snippet parsing utilities
# -----------------------------------------------------------------------------

def _parse_arg(s):
    """Parse a number or string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_list(s):
    """Parse a comma-separated list of values (strings or numbers)."""
    # Range: 'x-y'
    if '-' in s:
        m, M = map(_parse_arg, s.split('-'))
        return tuple(range(m, M + 1))
    # List of ids: 'x,y,z'
    elif ',' in s:
        return tuple(map(_parse_arg, s.split(',')))
    else:
        return _parse_arg(s)


def _parse_snippet(s):
    """Parse an entire snippet command."""
    return list(map(_parse_list, s.split(' ')))


# -----------------------------------------------------------------------------
# Show shortcut utility functions
# -----------------------------------------------------------------------------

def _show_shortcut(shortcut):
    if isinstance(shortcut, string_types):
        return shortcut
    elif isinstance(shortcut, (tuple, list)):
        return ', '.join(shortcut)


def _show_shortcuts(shortcuts, name=None):
    name = name or ''
    print()
    if name:
        name = ' for ' + name
    print('Keyboard shortcuts' + name)
    for name in sorted(shortcuts):
        print('{0:<40}: {1:s}'.format(name, _show_shortcut(shortcuts[name])))
    print()


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

class Actions(EventEmitter):
    """Handle GUI actions.

    This class attaches to a GUI and implements the following features:

    * Add and remove actions
    * Keyboard shortcuts for the actions
    * Display all shortcuts

    """
    def __init__(self):
        super(Actions, self).__init__()
        self._gui = None
        self._actions = {}

    def reset(self):
        """Reset the actions.

        All actions should be registered here, as follows:

        ```python
        @actions.connect
        def on_reset():
            actions.add(...)
            actions.add(...)
            ...
        ```

        """
        self.remove_all()
        self.emit('reset')

    def attach(self, gui):
        """Attach a GUI."""
        self._gui = gui

        # Register default actions.
        @self.connect
        def on_reset():
            # Default exit action.
            @self.shortcut('ctrl+q')
            def exit():
                gui.close()

    def add(self, name, callback=None, shortcut=None, alias=None,
            checkable=False, checked=False):
        """Add an action with a keyboard shortcut."""
        # TODO: add menu_name option and create menu bar
        # Get the alias from the character after & if it exists.
        if alias is None:
            alias = name[name.index('&') + 1] if '&' in name else name
        name = name.replace('&', '')
        if name in self._actions:
            return
        action = QtGui.QAction(name, self._gui)
        action.triggered.connect(callback)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if shortcut:
            if not isinstance(shortcut, (tuple, list)):
                shortcut = [shortcut]
            for key in shortcut:
                action.setShortcut(key)

        # Add some attributes to the QAction instance.
        # The alias is used in snippets.
        action._alias = alias
        action._callback = callback
        action._name = name
        # HACK: add the shortcut string to the QAction object so that
        # it can be shown in show_shortcuts(). I don't manage to recover
        # the key sequence string from a QAction using Qt.
        action._shortcut_string = shortcut or ''

        # Register the action.
        if self._gui:
            self._gui.addAction(action)
        self._actions[name] = action

        # Log the creation of the action.
        if not name.startswith('_'):
            logger.debug("Add action `%s`, alias `%s`, shortcut %s.",
                         name, alias, shortcut or '')

        if callback:
            setattr(self, name, callback)
        return action

    def get_name(self, alias_or_name):
        """Return an action name from its alias or name."""
        for name, action in self._actions.items():
            if alias_or_name in (action._alias, name):
                return name
        raise ValueError("Action `{}` doesn't exist.".format(alias_or_name))

    def run(self, action, *args):
        """Run an action, specified by its name or object."""
        if isinstance(action, string_types):
            name = self.get_name(action)
            assert name in self._actions
            action = self._actions[name]
        else:
            name = action._name
        if not name.startswith('_'):
            logger.debug("Execute action `%s`.", name)
        return action._callback(*args)

    def remove(self, name):
        """Remove an action."""
        if self._gui:
            self._gui.removeAction(self._actions[name])
        del self._actions[name]
        delattr(self, name)

    def remove_all(self):
        """Remove all actions."""
        names = sorted(self._actions.keys())
        for name in names:
            self.remove(name)

    @property
    def shortcuts(self):
        """A dictionary of action shortcuts."""
        return {name: action._shortcut_string
                for name, action in self._actions.items()}

    def show_shortcuts(self):
        """Print all shortcuts."""
        _show_shortcuts(self.shortcuts,
                        self._gui.title() if self._gui else None)

    def shortcut(self, key=None, name=None, **kwargs):
        """Decorator to add a global keyboard shortcut."""
        def wrap(func):
            self.add(name or func.__name__, shortcut=key,
                     callback=func, **kwargs)
        return wrap


# -----------------------------------------------------------------------------
# Snippets
# -----------------------------------------------------------------------------

class Snippets(object):
    """Provide keyboard snippets to quickly execute actions from a GUI.

    This class attaches to a GUI and an `Actions` instance. To every command
    is associated a snippet with the same name, or with an alias as indicated
    in the action. The arguments of the action's callback functions can be
    provided in the snippet's command with a simple syntax. For example, the
    following command:

    ```
    :my_action string 3-6
    ```

    corresponds to:

    ```python
    my_action('string', (3, 4, 5, 6))
    ```

    The snippet mode is activated with the `:` keyboard shortcut. A snippet
    command is activated with `Enter`, and one can leave the snippet mode
    with `Escape`.

    """

    # HACK: Unicode characters do not appear to work on Python 2
    cursor = '\u200A\u258C' if PY3 else ''

    # Allowed characters in snippet mode.
    # A Qt shortcut will be created for every character.
    _snippet_chars = ("abcdefghijklmnopqrstuvwxyz0123456789"
                      " ,.;?!_-+~=*/\(){}[]")

    def __init__(self):
        self._gui = None
        self._cmd = ''  # only used when there is no gui attached

    def attach(self, gui, actions):
        self._gui = gui
        self._actions = actions

        # Register snippet mode shortcut.
        @actions.connect
        def on_reset():
            @actions.shortcut(':')
            def enable_snippet_mode():
                self.mode_on()

    @property
    def command(self):
        """This is used to write a snippet message in the status bar.

        A cursor is appended at the end.

        """
        msg = self._gui.status_message if self._gui else self._cmd
        n = len(msg)
        n_cur = len(self.cursor)
        return msg[:n - n_cur]

    @command.setter
    def command(self, value):
        value += self.cursor
        if not self._gui:
            self._cmd = value
        else:
            self._gui.status_message = value

    def _backspace(self):
        """Erase the last character in the snippet command."""
        if self.command == ':':
            return
        logger.debug("Snippet keystroke `Backspace`.")
        self.command = self.command[:-1]

    def _enter(self):
        """Disable the snippet mode and execute the command."""
        command = self.command
        logger.debug("Snippet keystroke `Enter`.")
        self.mode_off()
        self.run(command)

    def _create_snippet_actions(self):
        """Delete all existing actions, and add mock ones for snippet
        keystrokes.

        Used to enable snippet mode.

        """
        self._actions.remove_all()

        # One action per allowed character.
        for i, char in enumerate(self._snippet_chars):

            def _make_func(char):
                def callback():
                    logger.debug("Snippet keystroke `%s`.", char)
                    self.command += char
                return callback

            self._actions.add('_snippet_{}'.format(i), shortcut=char,
                              callback=_make_func(char))

        self._actions.add('_snippet_backspace', shortcut='backspace',
                          callback=self._backspace)
        self._actions.add('_snippet_activate', shortcut=('enter', 'return'),
                          callback=self._enter)
        self._actions.add('_snippet_disable', shortcut='escape',
                          callback=self.mode_off)

    def run(self, snippet):
        """Executes a snippet command.

        May be overridden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        snippet_args = _parse_snippet(snippet)
        alias = snippet_args[0]
        name = self._actions.get_name(alias)
        assert name
        func = getattr(self._actions, name)
        try:
            logger.info("Processing snippet `%s`.", snippet)
            func(*snippet_args[1:])
        except Exception as e:
            logger.warn("Error when executing snippet: %s.", str(e))

    def is_mode_on(self):
        return self.command.startswith(':')

    def mode_on(self):
        logger.info("Snippet mode enabled, press `escape` to leave this mode.")
        # Remove all existing actions, and replace them by snippet keystroke
        # actions.
        self._create_snippet_actions()
        self.command = ':'

    def mode_off(self):
        if self._gui:
            self._gui.status_message = ''
        logger.info("Snippet mode disabled.")
        # Reestablishes the shortcuts.
        self._actions.reset()
