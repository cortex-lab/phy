# -*- coding: utf-8 -*-

"""Actions and snippets."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
from functools import partial, wraps
import logging
import re
import sys
import traceback

from .qt import QKeySequence, QAction, require_qt, input_dialog, busy_cursor, _get_icon
from phylib.utils import Bunch

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
    return tuple(map(_parse_list, s.split(' ')))


def _prompt_args(title, docstring, default=None):
    """Display a prompt dialog requesting function arguments.

    'default' is a function returning the default value for the proposed input dialog.

    """
    # There are args, need to display the dialog.
    # Extract Example: `...` in the docstring to put a predefined text
    # in the input dialog.
    logger.debug("Prompting arguments for %s", title)
    r = re.search('Example: `([^`]+)`', docstring)
    docstring_ = docstring[:r.start()].strip() if r else docstring
    try:
        text = str(default()) if default else (r.group(1) if r else None)
    except Exception as e:  # pragma: no cover
        logger.error("Error while handling user input: %s", str(e))
        return
    s, ok = input_dialog(title, docstring_, text)
    if not ok or not s:
        return
    # Parse user-supplied arguments and call the function.
    args = _parse_snippet(s)
    return args


# -----------------------------------------------------------------------------
# Show shortcut utility functions
# -----------------------------------------------------------------------------

def _get_shortcut_string(shortcut):
    """Return a string representation of a shortcut."""
    if not shortcut:
        return ''
    if isinstance(shortcut, (tuple, list)):
        return ', '.join([_get_shortcut_string(s) for s in shortcut])
    if isinstance(shortcut, str):
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
    assert isinstance(shortcut, str)
    if hasattr(QKeySequence, shortcut):
        return QKeySequence(getattr(QKeySequence, shortcut))
    sequence = QKeySequence.fromString(shortcut)
    assert not sequence.isEmpty()
    return sequence


def _show_shortcuts(shortcuts):
    """Display shortcuts."""
    out = []
    for n in sorted(shortcuts):
        shortcut = _get_shortcut_string(shortcuts[n])
        if not n.startswith('_') and not shortcut.startswith('-'):
            out.append('- {0:<40} {1:s}'.format(n, shortcut))
    if out:
        print('Keyboard shortcuts')
        print('\n'.join(out))
        print('')


def _show_snippets(snippets):
    """Display snippets."""
    out = []
    for n in sorted(snippets):
        snippet = snippets[n]
        if not n.startswith('_'):
            out.append('- {0:<40} :{1:s}'.format(n, snippet))
    if out:
        print('Snippets')
        print('\n'.join(out))
        print('')


def show_shortcuts_snippets(actions):
    """Show the shortcuts and snippets of an Actions instance."""
    print(actions.name)
    print('-' * len(actions.name))
    print()
    _show_shortcuts(actions.shortcuts)
    _show_snippets(actions._default_snippets)


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

def _alias(name):
    # Get the alias from the character after & if it exists.
    alias = name[name.index('&') + 1] if '&' in name else name
    alias = alias.replace(' ', '_').lower()
    return alias


def _expected_args(f):
    if isinstance(f, partial):
        argspec = inspect.getfullargspec(f.func)
    else:
        argspec = inspect.getfullargspec(f)
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
    return tuple(f_args)


@require_qt
def _create_qaction(gui, **kwargs):
    # Create the QAction instance.
    name = kwargs.get('name', '')
    name = name[0].upper() + name[1:].replace('_', ' ')
    action = QAction(name, gui)

    # Show an input dialog if there are args.
    callback = kwargs.get('callback', None)
    title = getattr(callback, '__name__', 'action')
    # Number of expected arguments.
    n_args = kwargs.get('n_args', None) or len(_expected_args(callback))

    @wraps(callback)
    def wrapped(is_checked, *args):
        if kwargs.get('checkable', None):
            args = (is_checked,) + args
        if kwargs.get('prompt', None):
            args += _prompt_args(
                title, docstring, default=kwargs.get('prompt_default', None)) or ()
            if not args:  # pragma: no cover
                logger.debug("User cancelled input prompt, aborting.")
                return
        if len(args) < n_args:
            logger.warning(
                "Invalid function arguments: expecting %d but got %d", n_args, len(args))
            return
        try:
            # Set a busy cursor if set_busy is True.
            with busy_cursor(kwargs.get('set_busy', None)):
                return callback(*args)
        except Exception:  # pragma: no cover
            logger.warning("Error when executing action %s.", name)
            logger.debug(''.join(traceback.format_exception(*sys.exc_info())))

    action.triggered.connect(wrapped)
    sequence = _get_qkeysequence(kwargs.get('shortcut', None))
    if not isinstance(sequence, (tuple, list)):
        sequence = [sequence]
    action.setShortcuts(sequence)
    assert kwargs.get('docstring', None)
    docstring = re.sub(r'\s+', ' ', kwargs.get('docstring', None))
    docstring += ' (alias: {})'.format(kwargs.get('alias', None))
    action.setStatusTip(docstring)
    action.setWhatsThis(docstring)
    action.setCheckable(kwargs.get('checkable', None))
    action.setChecked(kwargs.get('checked', None))
    if kwargs.get('icon', None):
        action.setIcon(_get_icon(kwargs['icon']))
    return action


class Actions(object):
    """Group of actions bound to a GUI.

    This class attaches to a GUI and implements the following features:

    * Add and remove actions
    * Keyboard shortcuts for the actions
    * Display all shortcuts

    Constructor
    -----------

    gui : GUI instance
    name : str
        Name of this group of actions.
    menu : str
        Name of the GUI menu that will contain the actions.
    submenu : str
        Name of the GUI submenu that will contain the actions.
    default_shortcuts : dict
        Map action names to keyboard shortcuts (regular strings).
    default_snippets : dict
        Map action names to snippets (regular strings).

    """
    def __init__(
            self, gui, name=None, menu=None, submenu=None, view=None,
            insert_menu_before=None, default_shortcuts=None, default_snippets=None):
        self._actions_dict = {}
        self._aliases = {}
        self._default_shortcuts = default_shortcuts or {}
        self._default_snippets = default_snippets or {}
        assert name
        self.name = name
        self.menu = menu
        self.submenu = submenu
        self.view = view
        self.view_submenu = None
        self.insert_menu_before = insert_menu_before
        self._view_submenus = {}
        self.gui = gui
        gui.actions.append(self)
        # Create the menu when creating the Actions instance.
        if menu:
            gui.get_menu(menu, insert_menu_before)

    def _get_menu(self, menu=None, submenu=None, view=None, view_submenu=None):
        """Return the QMenu depending on a combination of keyword arguments."""
        # Defaults.
        menu = menu or self.menu
        submenu = submenu or self.submenu
        view = view or self.view
        view_submenu = view_submenu or self.view_submenu

        # If the action is a view action, it should be added to the view's menu in the dock widget.
        if view:
            if view_submenu and view_submenu not in self._view_submenus:
                self._view_submenus[view_submenu] = view.dock._menu.addMenu(view_submenu)
            if view_submenu:
                return self._view_submenus[view_submenu]
            else:
                return view.dock._menu

        # Create the submenu if there is one.
        if submenu:
            # Create the submenu.
            self.gui.get_submenu(menu, submenu)
            # Make sure the action gets added to the submenu.
            menu = submenu
        if menu:
            return self.gui.get_menu(menu)

    def add(self, callback=None, name=None, shortcut=None, alias=None, prompt=False, n_args=None,
            docstring=None, menu=None, submenu=None, view=None, view_submenu=None, verbose=True,
            checkable=False, checked=False, set_busy=False, prompt_default=None,
            show_shortcut=True, icon=None, toolbar=False):
        """Add an action with a keyboard shortcut.

        Parameters
        ----------

        callback : function
            Take no argument if checkable is False, or a boolean (checked) if it is True
        name : str
            Action name, the callback's name by default.
        shortcut : str
            The keyboard shortcut for this action.
        alias : str
            Snippet, the name by default.
        prompt : boolean
            Whether this action should display a dialog with an input box where the user can
            write arguments to the callback function.
        n_args : int
            If prompt is True, specify the number of expected arguments.
        set_busy : boolean
            Whether to use a busy cursor while performing the action.
        prompt_default : str
            The default text in the input text box, if prompt is True.
        docstring : str
            The action docstring, to be displayed in the status bar when hovering over the action
            item in the menu. By default, the function's docstring.
        menu : str
            The name of the menu where the action should be added. It is automatically created
            if it doesn't exist.
        submenu : str
            The name of the submenu where the action should be added. It is automatically created
            if it doesn't exist.
        view : QWidget
            A view that belongs to the GUI, if the actions are to be added to the view's menu bar.
        view_submenu : str
            The name of a submenu in the view menu.
        checkable : boolean
            Whether the action is checkable (toggle on/off).
        checked : boolean
            Whether the checkable action is initially checked or not.
        show_shortcut : boolean
            Whether to show the shortcut in the Help action that displays all GUI shortcuts.
        icon : str
            Hexadecimal code of the font-awesome icon.
        toolbar : boolean
            Whether to add the action to the toolbar.

        """
        param_names = sorted(inspect.signature(Actions.add).parameters)
        l = locals()
        kwargs = {param_name: l[param_name] for param_name in param_names if param_name != 'self'}
        if callback is None:
            # Allow to use either add(func) or @add or @add(...).
            kwargs.pop('callback', None)
            return partial(self.add, **kwargs)
        assert callback

        # Get the name from the callback function if needed.
        name = name or callback.__name__
        alias = alias or self._default_snippets.get(name, _alias(name)).split(' ')[0]
        name = name.replace('&', '')
        shortcut = shortcut or self._default_shortcuts.get(name, None)

        # Skip existing action.
        if name in self._actions_dict:
            return

        # Set the status tip from the function's docstring.
        docstring = docstring or callback.__doc__ or name
        docstring = re.sub(r'[ \t\r\f\v]{2,}', ' ', docstring.strip())

        # Create and register the action.
        kwargs.update(name=name, alias=alias, shortcut=shortcut, docstring=docstring)
        action = _create_qaction(self.gui, **kwargs)
        action_obj = Bunch(qaction=action, **kwargs)
        if verbose and not name.startswith('_'):
            logger.log(5, "Add action `%s` (%s).", name, _get_shortcut_string(action.shortcut()))
        self.gui.addAction(action)

        # Do not show private actions in the menu.
        if not name.startswith('_'):
            # Find the menu in which the action should be added.
            qmenu = self._get_menu(
                menu=menu, submenu=submenu, view=view, view_submenu=view_submenu)
            if qmenu:
                qmenu.addAction(action)

        # Add the action to the toolbar.
        if toolbar:
            self.gui._toolbar.show()
            self.gui._toolbar.addAction(action)

        self._actions_dict[name] = action_obj
        # Register the alias -> name mapping.
        self._aliases[alias] = name

        # Set the callback method.
        if callback:
            setattr(self, name.lower().replace(' ', '_').replace(':', ''), callback)

    def separator(self, **kwargs):
        """Add a separator.

        Parameters
        ----------

        menu : str
            The name of the menu where the separator should be added. It is automatically created
            if it doesn't exist.
        submenu : str
            The name of the submenu where the separator should be added. It is automatically
            created if it doesn't exist.
        view : QWidget
            A view that belongs to the GUI, if the separator is to be added to the view's menu bar.
        view_submenu : str
            The name of a submenu in the view menu.

        """
        self._get_menu(**kwargs).addSeparator()

    def disable(self, name=None):
        """Disable all actions, or only one if a name is passed."""
        if name is None:
            for name in self._actions_dict:
                self.disable(name)
            return
        self._actions_dict[name].qaction.setEnabled(False)

    def enable(self, name=None):
        """Enable all actions, or only one if a name is passed.."""
        if name is None:
            for name in self._actions_dict:
                self.enable(name)
            return
        self._actions_dict[name].qaction.setEnabled(True)

    def get(self, name):
        """Get a QAction instance from its name."""
        return self._actions_dict[name].qaction if name in self._actions_dict else None

    def run(self, name, *args):
        """Run an action as specified by its name."""
        assert isinstance(name, str)
        # Resolve the alias if it is an alias.
        name = self._aliases.get(name, name)
        # Get the action.
        action = self._actions_dict.get(name, None)
        if not action:
            raise ValueError("Action `{}` doesn't exist.".format(name))
        if not name.startswith('_'):
            logger.debug("Execute action `%s`.", name)
        try:
            return action.callback(*args)
        except TypeError as e:
            logger.warning("Invalid action arguments: " + str(e))
            return

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
        """A dictionary mapping action names to keyboard shortcuts."""
        out = {}
        for name in sorted(self._actions_dict):
            action = self._actions_dict[name]
            if not action.show_shortcut:
                continue
            # Discard actions without shortcut and without an alias.
            if not action.shortcut and not action.alias:
                continue
            # Only show alias for actions with no shortcut.
            alias_str = ' (:%s)' % action.alias if action.alias != name else ''
            shortcut = action.shortcut or '-'
            shortcut = shortcut if isinstance(action.shortcut, str) else ', '.join(shortcut)
            out[name] = '%s%s' % (shortcut, alias_str)
        return out

    def show_shortcuts(self):
        """Display all shortcuts in the console."""
        show_shortcuts_snippets(self)

    def __contains__(self, name):
        """Whether the Actions group contains a specified action."""
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

    When the snippet mode is enabled (with `:`), this object adds a hidden Qt action
    for every keystroke. These actions are removed when the snippet mode is disabled.

    Constructor
    -----------

    gui : GUI instance

    """

    # HACK: Unicode characters do not seem to work on Python 2
    cursor = '\u200A\u258C'

    # Allowed characters in snippet mode.
    # A Qt shortcut will be created for every character.
    _snippet_chars = r"abcdefghijklmnopqrstuvwxyz0123456789 ,.;?!_-+~=*/\(){}[]<>&|"

    def __init__(self, gui):
        self.gui = gui
        self._status_message = gui.status_message

        self.actions = Actions(gui, name='Snippets', menu='&File')

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
        """This is used to write a snippet message in the status bar. A cursor is appended at
        the end."""
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

            # Lowercase letters.
            self.actions.add(
                name='_snippet_{}'.format(i),
                shortcut=char,
                callback=_make_func(char))

            # Uppercase letters.
            if char in self._snippet_chars[:26]:
                self.actions.add(
                    name='_snippet_{}_upper'.format(i),
                    shortcut='shift+' + char,
                    callback=_make_func(char.upper()))

        self.actions.add(
            name='_snippet_backspace', shortcut='backspace', callback=self._backspace)
        self.actions.add(
            name='_snippet_activate', shortcut=('enter', 'return'), callback=self._enter)
        self.actions.add(
            name='_snippet_disable', shortcut='escape', callback=self.mode_off)

    def run(self, snippet):
        """Execute a snippet command.

        May be overridden.

        """
        assert snippet[0] == ':'
        snippet = snippet[1:]
        snippet_args = _parse_snippet(snippet)
        name = snippet_args[0]

        logger.debug("Processing snippet `%s`.", snippet)
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
            logger.warning("Couldn't find action `%s`.", name)
        except Exception as e:
            logger.warning("Error when executing snippet: \"%s\".", str(e))
            logger.debug(''.join(traceback.format_exception(*sys.exc_info())))

    def is_mode_on(self):
        """Whether the snippet mode is enabled."""
        return self.command.startswith(':')

    def mode_on(self):
        """Enable the snippet mode."""
        logger.debug("Snippet mode enabled, press `escape` to leave this mode.")
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
        """Disable the snippet mode."""
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
