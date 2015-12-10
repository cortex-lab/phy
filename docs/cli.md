# CLI

When you install phy, a command-line tool named `phy` is installed:

```bash
$ phy
Usage: phy [OPTIONS] COMMAND [ARGS]...

  By default, the `phy` command does nothing. Add subcommands with plugins
  using `attach_to_cli()` and the `click` library.

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.
```

This command doesn't do anything by default, but it serves as an entry-point for your applications.

## Adding a subcommand

A subcommand is called with `phy subcommand ...` from the command-line. To create a subcommand, create a new plugin, and implement the `attach_to_cli(cli)` method. This uses the [click](http://click.pocoo.org/5/) library.

Here is an example. Create a file in `~/.phy/plugins/hello.py` and write the following:

```
from phy import IPlugin
import click


class MyPlugin(IPlugin):
    def attach_to_cli(self, cli):
        @cli.command('hello')
        @click.argument('name')
        def hello(name):
            print("Hello %s!" % name)
```

Then, type the following in a system shell:

```bash
$ phy
Usage: phy [OPTIONS] COMMAND [ARGS]...

  By default, the `phy` command does nothing. Add subcommands with plugins
  using `attach_to_cli()` and the `click` library.

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  hello

$ phy hello
Usage: phy hello [OPTIONS] NAME

Error: Missing argument "name".

$ phy hello world
Hello world!
```

When the `phy` CLI is created, the `attach_to_cli(cli)` method of all discovered plugins are called. Refer to the click documentation to create subcommands with phy.

## Creating a graphical application

You can use this system to create a graphical application that is launched with `phy some_subcommand`. Moreover, your graphical application can itself accept user-defined plugins.

Here is a complete example. Write the following in `~/.phy/plugins/mygui.py`:

```
import click
from phy import IPlugin
from phy.gui import GUI, HTMLWidget, create_app, run_app, load_gui_plugins
from phy.utils import Bunch


class MyGUI(GUI):
    def __init__(self, name, plugins=None):
        super(MyGUI, self).__init__()

        # We create a widget.
        view = HTMLWidget()
        view.set_body("Hello %s!" % name)
        view.show()
        self.add_view(view)

        # We load all plugins attached to that GUI.
        session = Bunch(name=name)
        load_gui_plugins(self, plugins, session)


class MyGUIPlugin(IPlugin):
    def attach_to_cli(self, cli):

        @cli.command(name='mygui')
        @click.argument('name')
        def mygui(name):

            # Create the Qt application.
            create_app()

            # Show the GUI.
            gui = MyGUI(name)
            gui.show()

            # Start the Qt event loop.
            run_app()

            # Close the GUI.
            gui.close()
            del gui
```

Now, you can call `phy mygui world` to open a GUI showing `Hello world!`.

## GUI plugins

Your users can now create plugins for your graphical application, by creating a plugin with the `attach_to_gui(gui, session)` method. In this method, you can add actions, add views, and do anything provided by the GUI API.

The `session` object is any Python object passed to the plugins by the GUI. Generally, it is a `Bunch` instance (just a Python dictionary with the additional `bunch.name` syntax) containing any data that you want to pass to the plugins.

Here is a complete example. There are three steps.

### Creating the plugin

First, create a file in `~/.phy/plugins/mygui_plugin.py` with the following:

```
from phy import IPlugin
from phy.gui import Actions


class MyGUIPlugin(IPlugin):
    def attach_to_gui(self, gui, session):
        actions = Actions(gui)

        @actions.add(shortcut='a')
        def myaction():
            print("Hello %s!" % session.name)
```

### Activating the plugin

Next, add the following line in `~/.phy/phy_config.py`:

```
c.MyGUI.plugins = ['MyGUIPlugin']
```

This is the list of the plugin names to activate automatically when creating a `MyGUI` instance. When you create a GUI from Python, you can also pass the list of plugins to activate as follows: `gui = MyGUI(name, plugins=[...])`.

### Testing the plugin

Finally, launch the GUI with `phy mygui world` and press `a` in the GUI. It should print `Hello world!` in the console.
