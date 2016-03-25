# Configuration and plugin system

phy uses part of the **traitlets** package for its config system. **This is still a work in progress**.

## Configuration file

The configuration file is a Python file stored at `~/.phy/phy_config.py`. It should always begin with `c = get_config()` with no import (this function is automatically injected in the namespace by the config system).

Then, you can set configuration options as follows:

`c.SomeClass.some_param = some_value`

## Plugin system

A plugin is a Python class deriving from `phy.IPlugin`. To ensure that phy knows about your plugin, just make sure that your class is imported in the Python namespace.

Here are two common methods:

* Implement your plugin in a Python file and put this file in `~/.phy/plugins/`: it will be automatically discovered by phy.
* Edit `c.Plugins.dirs = ['/path/to/folder']` in your `phy_config.py` file: all Python scripts there will be automatically imported.

Here is a minimal plugin template:

```
from phy import IPlugin

class MyPlugin(IPlugin):
    pass
```
