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
            """Display a greeting message."""
            print("Hello %s!" % name)
```

Then, type the following in a system shell:

```bash
$ phy
Usage: phy [OPTIONS] COMMAND [ARGS]...
[...]
Commands:
  hello          Display a greeting message.

$ phy hello world
Hello world!
```

When the `phy` CLI is created, the `attach_to_cli(cli)` method of all discovered plugins are called. Refer to the click documentation to create subcommands with phy.
