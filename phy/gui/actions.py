# -*- coding: utf-8 -*-

"""Actions and snippets."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from functools import partial
import logging

from six import string_types, PY3

from .qt import QtGui
from phy.utils import Bunch

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
        return list(range(m, M + 1))
    # List of ids: 'x,y,z'
    elif ',' in s:
        return list(map(_parse_arg, s.split(',')))
    else:
        return _parse_arg(s)


def _parse_snippet(s):
    """Parse an entire snippet command."""
    return list(map(_parse_list, s.split(' ')))


# -----------------------------------------------------------------------------
# Show shortcut utility functions
# -----------------------------------------------------------------------------

def _get_shortcut_string(shortcut):
    """Return a string representation of a shortcut."""
    if shortcut is None:
        return ''
    if isinstance(shortcut, (tuple, list)):
        return ', '.join([_get_shortcut_string(s) for s in shortcut])
    if isinstance(shortcut, string_types):
        return shortcut.lower()
    assert isinstance(shortcut, QtGui.QKeySequence)
    s = shortcut.toString() or ''
    return str(s).lower()


def _get_qkeysequence(shortcut):
    """Return a QKeySequence or list of QKeySequence from a shortcut string."""
    if shortcut is None:
        return []
    if isinstance(shortcut, (tuple, list)):
        return [_get_qkeysequence(s) for s in shortcut]
    assert isinstance(shortcut, string_types)
    if hasattr(QtGui.QKeySequence, shortcut):
        return QtGui.QKeySequence(getattr(QtGui.QKeySequence, shortcut))
    sequence = QtGui.QKeySequence.fromString(shortcut)
    assert not sequence.isEmpty()
    return sequence


def _show_shortcuts(shortcuts, name=None):
    """Display shortcuts."""
    name = name or ''
    print()
    if name:
        name = ' for ' + name
    print('Keyboard shortcuts' + name)
    for name in sorted(shortcuts):
        shortcut = _get_shortcut_string(shortcuts[name])
        print('{0:<40}: {1:s}'.format(name, shortcut))
    print()


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

def _alias(name):
    # Get the alias from the character after & if it exists.
    alias = name[name.index('&') + 1] if '&' in name else name
    return alias


def _create_qaction(gui, name, callback, shortcut):
    # Create the QAction instance.
    action = QtGui.QAction(name, gui)

    def wrapped(checked, *args, **kwargs):
        return callback(*args, **kwargs)

    action.triggered.connect(wrapped)
    sequence = _get_qkeysequence(shortcut)
    if not isinstance(sequence, (tuple, list)):
        sequence = [sequence]
    for s in sequence:
        action.setShortcut(s)
    return action


class Actions(object):
    """Handle GUI actions.

    This class attaches to a GUI and implements the following features:

    * Add and remove actions
    * Keyboard shortcuts for the actions
    * Display all shortcuts

    """
    def __init__(self):
        self._gui = None
        self._actions = {}
        self._aliases = {}

    def get_action_dict(self):
        return self._actions.copy()

    def attach(self, gui, enable_snippets=True):
        """Attach a GUI."""
        self._gui = gui

        # Default exit action.
        @self.add(shortcut='Quit')
        def exit():
            gui.close()

        # Create and attach snippets.
        if enable_snippets:
            self.snippets = Snippets()
            self.snippets.attach(gui, self)

    def add(self, callback=None, name=None, shortcut=None, alias=None):
        """Add an action with a keyboard shortcut."""
        # TODO: add menu_name option and create menu bar
        if callback is None:
            # Allow to use either add(func) or @add or @add(...).
            return partial(self.add, name=name, shortcut=shortcut, alias=alias)
        assert callback

        # Get the name from the callback function if needed.
        name = name or callback.__name__
        alias = alias or _alias(name)
        name = name.replace('&', '')

        # Skip existing action.
        if name in self._actions:
            return

        # Create and register the action.
        action = _create_qaction(self._gui, name, callback, shortcut)
        action_obj = Bunch(qaction=action, name=name, alias=alias,
                           shortcut=shortcut, callback=callback)
        if self._gui:
            self._gui.addAction(action)
        self._actions[name] = action_obj
        # Register the alias -> name mapping.
        self._aliases[alias] = name

        # Set the callback method.
        if callback:
            setattr(self, name, callback)

    def run(self, name, *args):
        """Run an action as specified by its name."""
        assert isinstance(name, string_types)
        # Resolve the alias if it is an alias.
        name = self._aliases.get(name, name)
        # Get the action.
        action = self._actions.get(name, None)
        if not action:
            raise ValueError("Action `{}` doesn't exist.".format(name))
        if not name.startswith('_'):
            logger.debug("Execute action `%s`.", name)
        return action.callback(*args)

    def remove(self, name):
        """Remove an action."""
        if self._gui:
            self._gui.removeAction(self._actions[name].qaction)
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
        return {name: action.shortcut
                for name, action in self._actions.items()}

    def show_shortcuts(self):
        """Print all shortcuts."""
        _show_shortcuts(self.shortcuts,
                        self._gui.windowTitle() if self._gui else None)


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

    # HACK: Unicode characters do not seem to work on Python 2
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
        # We will keep a backup of all actions so that we can switch
        # safely to the set of shortcut actions when snippet mode is on.
        self._actions_backup = {}

        # Register snippet mode shortcut.
        @actions.add(shortcut=':')
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
        logger.log(5, "Snippet keystroke `Enter`.")
        # NOTE: we need to set back the actions (mode_off) before running
        # the command.
        self.mode_off()
        self.run(command)

    def _create_snippet_actions(self):
        """Add mock Qt actions for snippet keystrokes.

        Used to enable snippet mode.

        """
        # One action per allowed character.
        for i, char in enumerate(self._snippet_chars):

            def _make_func(char):
                def callback():
                    logger.log(5, "Snippet keystroke `%s`.", char)
                    self.command += char
                return callback

            self._actions.add(name='_snippet_{}'.format(i),
                              shortcut=char,
                              callback=_make_func(char))

        self._actions.add(name='_snippet_backspace',
                          shortcut='backspace',
                          callback=self._backspace)
        self._actions.add(name='_snippet_activate',
                          shortcut=('enter', 'return'),
                          callback=self._enter)
        self._actions.add(name='_snippet_disable',
                          shortcut='escape',
                          callback=self.mode_off)

    def run(self, snippet):
        """Executes a snippet command.

        May be overridden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        snippet_args = _parse_snippet(snippet)
        name = snippet_args[0]

        logger.info("Processing snippet `%s`.", snippet)
        try:
            self._actions.run(name, *snippet_args[1:])
        except Exception as e:
            logger.warn("Error when executing snippet: \"%s\".", str(e))

    def is_mode_on(self):
        return self.command.startswith(':')

    def mode_on(self):
        logger.info("Snippet mode enabled, press `escape` to leave this mode.")
        self._actions_backup = self._actions.get_action_dict()
        # Remove all existing actions.
        self._actions.remove_all()
        # Add snippet keystroke actions.
        self._create_snippet_actions()
        self.command = ':'

    def mode_off(self):
        if self._gui:
            self._gui.status_message = ''
        # Remove all existing actions.
        self._actions.remove_all()
        logger.info("Snippet mode disabled.")
        # Reestablishes the shortcuts.
        for action_obj in self._actions_backup.values():
            self._actions.add(callback=action_obj.callback,
                              name=action_obj.name,
                              shortcut=action_obj.shortcut,
                              alias=action_obj.alias,
                              )
