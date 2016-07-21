# -*- coding: utf-8 -*-

"""Actions and snippets."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
from functools import partial
import logging
import re
import sys
import traceback

from six import string_types, PY3

from .qt import QKeySequence, QAction, require_qt, _input_dialog
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


def _wrap_callback_args(f, docstring=None):  # pragma: no cover
    """Display a Qt dialog when a function has arguments.

    The user can write function arguments as if it was a snippet.

    """
    def wrapped(checked, *args):
        if args:
            return f(*args)
        if isinstance(f, partial):
            argspec = inspect.getargspec(f.func)
        else:
            argspec = inspect.getargspec(f)
        f_args = argspec.args
        if 'self' in f_args:
            f_args.remove('self')
        # Remove arguments with defaults from the list.
        if len(argspec.defaults or ()):
            f_args = f_args[:-len(argspec.defaults)]
        # Remove arguments supplied in a partial.
        if isinstance(f, partial):
            f_args = f_args[len(f.args):]
            f_args = [arg for arg in f_args if arg not in f.keywords]
        # If there are no remaining args, we can just run the fu nction.
        if not f_args:
            return f()
        # There are args, need to display the dialog.
        s, ok = _input_dialog(getattr(f, '__name__', 'action'), docstring)
        if not ok or not s:
            return
        # Parse user-supplied arguments and call the function.
        args = _parse_snippet(s)
        return f(*args)
    return wrapped


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
        if hasattr(QKeySequence, shortcut):
            shortcut = QKeySequence(getattr(QKeySequence, shortcut))
        else:
            return shortcut.lower()
    assert isinstance(shortcut, QKeySequence)
    s = shortcut.toString() or ''
    return str(s).lower()


def _get_qkeysequence(shortcut):
    """Return a QKeySequence or list of QKeySequence from a shortcut string."""
    if shortcut is None:
        return []
    if isinstance(shortcut, (tuple, list)):
        return [_get_qkeysequence(s) for s in shortcut]
    assert isinstance(shortcut, string_types)
    if hasattr(QKeySequence, shortcut):
        return QKeySequence(getattr(QKeySequence, shortcut))
    sequence = QKeySequence.fromString(shortcut)
    assert not sequence.isEmpty()
    return sequence


def _show_shortcuts(shortcuts, name=None):
    """Display shortcuts."""
    name = name or ''
    print('')
    if name:
        name = ' for ' + name
    print('Keyboard shortcuts' + name)
    for name in sorted(shortcuts):
        shortcut = _get_shortcut_string(shortcuts[name])
        if not name.startswith('_'):
            print('- {0:<40}: {1:s}'.format(name, shortcut))


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

def _alias(name):
    # Get the alias from the character after & if it exists.
    alias = name[name.index('&') + 1] if '&' in name else name
    return alias


@require_qt
def _create_qaction(gui, name, callback, shortcut, docstring=None, alias=''):
    # Create the QAction instance.
    action = QAction(name.capitalize().replace('_', ' '), gui)

    # Show an input dialog if there are args.
    callback = _wrap_callback_args(callback,
                                   docstring=docstring,
                                   )

    action.triggered.connect(callback)
    sequence = _get_qkeysequence(shortcut)
    if not isinstance(sequence, (tuple, list)):
        sequence = [sequence]
    action.setShortcuts(sequence)
    assert docstring
    docstring += ' (alias: {})'.format(alias)
    action.setStatusTip(docstring)
    action.setWhatsThis(docstring)
    return action


class Actions(object):
    """Handle GUI actions.

    This class attaches to a GUI and implements the following features:

    * Add and remove actions
    * Keyboard shortcuts for the actions
    * Display all shortcuts

    """
    def __init__(self, gui, name=None, menu=None, default_shortcuts=None):
        self._actions_dict = {}
        self._aliases = {}
        self._default_shortcuts = default_shortcuts or {}
        self.name = name
        self.menu = menu
        self.gui = gui
        gui.actions.append(self)

    def add(self, callback=None, name=None, shortcut=None, alias=None,
            docstring=None, menu=None, verbose=True):
        """Add an action with a keyboard shortcut."""
        # TODO: add menu_name option and create menu bar
        if callback is None:
            # Allow to use either add(func) or @add or @add(...).
            return partial(self.add, name=name, shortcut=shortcut,
                           alias=alias, menu=menu)
        assert callback

        # Get the name from the callback function if needed.
        name = name or callback.__name__
        alias = alias or _alias(name)
        name = name.replace('&', '')
        shortcut = shortcut or self._default_shortcuts.get(name, None)

        # Skip existing action.
        if name in self._actions_dict:
            return

        # Set the status tip from the function's docstring.
        docstring = docstring or callback.__doc__ or name
        docstring = re.sub(r'[\s]{2,}', ' ', docstring)

        # Create and register the action.
        action = _create_qaction(self.gui, name, callback,
                                 shortcut,
                                 docstring=docstring,
                                 alias=alias,
                                 )
        action_obj = Bunch(qaction=action, name=name, alias=alias,
                           shortcut=shortcut, callback=callback, menu=menu)
        if verbose and not name.startswith('_'):
            logger.log(5, "Add action `%s` (%s).", name,
                       _get_shortcut_string(action.shortcut()))
        self.gui.addAction(action)
        # Add the action to the menu.
        menu = menu or self.menu
        # Do not show private actions in the menu.
        if menu and not name.startswith('_'):
            self.gui.get_menu(menu).addAction(action)
        self._actions_dict[name] = action_obj
        # Register the alias -> name mapping.
        self._aliases[alias] = name

        # Set the callback method.
        if callback:
            setattr(self, name, callback)

    def separator(self, menu=None):
        """Add a separator"""
        self.gui.get_menu(menu or self.menu).addSeparator()

    def disable(self, name=None):
        """Disable one or all actions."""
        if name is None:
            for name in self._actions_dict:
                self.disable(name)
            return
        self._actions_dict[name].qaction.setEnabled(False)

    def enable(self, name=None):
        """Enable one or all actions."""
        if name is None:
            for name in self._actions_dict:
                self.enable(name)
            return
        self._actions_dict[name].qaction.setEnabled(True)

    def get(self, name):
        """Get a QAction instance from its name."""
        return self._actions_dict[name].qaction

    def run(self, name, *args):
        """Run an action as specified by its name."""
        assert isinstance(name, string_types)
        # Resolve the alias if it is an alias.
        name = self._aliases.get(name, name)
        # Get the action.
        action = self._actions_dict.get(name, None)
        if not action:
            raise ValueError("Action `{}` doesn't exist.".format(name))
        if not name.startswith('_'):
            logger.debug("Execute action `%s`.", name)
        return action.callback(*args)

    def remove(self, name):
        """Remove an action."""
        self.gui.removeAction(self._actions_dict[name].qaction)
        del self._actions_dict[name]
        delattr(self, name)

    def remove_all(self):
        """Remove all actions."""
        names = sorted(self._actions_dict.keys())
        for name in names:
            self.remove(name)

    @property
    def shortcuts(self):
        """A dictionary of action shortcuts."""
        return {name: action.shortcut
                for name, action in self._actions_dict.items()}

    def show_shortcuts(self):
        """Print all shortcuts."""
        gui_name = self.gui.name
        actions_name = self.name
        name = ('{} - {}'.format(gui_name, actions_name)
                if actions_name else gui_name)
        _show_shortcuts(self.shortcuts, name)

    def __contains__(self, name):
        return name in self._actions_dict

    def __repr__(self):
        return '<Actions {}>'.format(sorted(self._actions_dict))


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

    def __init__(self, gui):
        self.gui = gui
        self._status_message = gui.status_message

        self.actions = Actions(gui, name='Snippets', menu='Snippets')

        # Register snippet mode shortcut.
        @self.actions.add(shortcut=':')
        def enable_snippet_mode():
            """Enable the snippet mode (type action alias in the status
            bar)."""
            self.mode_on()

        self._create_snippet_actions()
        self.mode_off()

    @property
    def command(self):
        """This is used to write a snippet message in the status bar.

        A cursor is appended at the end.

        """
        msg = self.gui.status_message
        n = len(msg)
        n_cur = len(self.cursor)
        return msg[:n - n_cur]

    @command.setter
    def command(self, value):
        value += self.cursor
        self.gui.unlock_status()
        self.gui.status_message = value
        self.gui.lock_status()

    def _backspace(self):
        """Erase the last character in the snippet command."""
        if self.command == ':':
            return
        logger.log(5, "Snippet keystroke `Backspace`.")
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

            self.actions.add(name='_snippet_{}'.format(i),
                             shortcut=char,
                             callback=_make_func(char))

        self.actions.add(name='_snippet_backspace',
                         shortcut='backspace',
                         callback=self._backspace)
        self.actions.add(name='_snippet_activate',
                         shortcut=('enter', 'return'),
                         callback=self._enter)
        self.actions.add(name='_snippet_disable',
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
            # Try to run the snippet on all attached Actions instances.
            for actions in self.gui.actions:
                try:
                    actions.run(name, *snippet_args[1:])
                    return
                except ValueError:
                    # This Actions instance doesn't contain the requested
                    # snippet, trying the next attached Actions instance.
                    pass
            logger.warn("Couldn't find action `%s`.", name)
        except Exception as e:
            logger.warn("Error when executing snippet: \"%s\".", str(e))
            logger.debug(''.join(traceback.format_exception(*sys.exc_info())))

    def is_mode_on(self):
        return self.command.startswith(':')

    def mode_on(self):
        logger.info("Snippet mode enabled, press `escape` to leave this mode.")
        # Save the current status message.
        self._status_message = self.gui.status_message
        self.gui.lock_status()

        # Silent all actions except the Snippets actions.
        for actions in self.gui.actions:
            if actions != self.actions:
                actions.disable()
        self.actions.enable()

        self.command = ':'

    def mode_off(self):
        self.gui.unlock_status()
        # Reset the GUI status message that was set before the mode was
        # activated.
        self.gui.status_message = self._status_message

        # Re-enable all actions except the Snippets actions.
        self.actions.disable()
        for actions in self.gui.actions:
            if actions != self.actions:
                actions.enable()
        # The `:` shortcut should always be enabled.
        self.actions.enable('enable_snippet_mode')
