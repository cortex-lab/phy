# Developing with phy

This section gives some information to develop new features in phy, as plugins or as pull requests to the main repository.

## Code overview

### Event system

example of on_cluster, singleton instance

### Cache system

### GUI state

global and local


## Writing custom OpenGL view

Matplotlib views can be developed for phy, as explained in the Customization section.

However, for better performance, all built-in views in phy are not based on matplotlib, but on OpenGL. OpenGL is a graphics interface for graphics cards, which provide hardware acceleration for fast display of large amounts of data.

In phy, OpenGL views are written on top of a thin layer, a fork of `glumpy.gloo` (object-oriented interface to OpenGL). On top of that, the `phy.plot` module proposes a minimal plotting API. This interface is complex as it suffers from the limitations of OpenGL. As such, writing custom OpenGL views for phy is not straightforward.


### GLSL

### Visuals

### Transforms

### Interacts and layouts

### Plot canvas and batch processing


## Writing a custom HTML-based table view

write a custom html table widget


## Creating a custom GUI

actions, keyboard shortcuts, views
