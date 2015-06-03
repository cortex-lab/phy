# -*- coding: utf-8 -*-
from __future__ import print_function

"""Base classes for GUIs."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..ext.six import string_types
from ..utils.logging import debug
import phy.plot.view_models.kwik as vmk


#------------------------------------------------------------------------------
# Widget creator
#------------------------------------------------------------------------------

class WidgetCreator(object):
    """Manage the creation of widgets.

    A widget must implement:

    * `name`
    * `show()`
    * `connect` (for `close` event)

    """
    _widget_classes = {}

    def __init__(self):
        self._widgets = []

    def _create_widget(self, widget_class, **kwargs):
        """Create a new widget of a given class.

        Must be overriden.

        """
        pass

    @property
    def widget_classes(self):
        return self._widget_classes

    def get(self, name=None):
        """Return the list of widgets of a given type."""
        if name is None:
            return self._widgets
        return [widget for widget in self._widgets if widget.name == name]

    def add(self, widget_class, show=True, **kwargs):
        """Add a new widget."""
        # widget_class can also be a name, but in this case it must be
        # registered in self._widget_classes.
        if isinstance(widget_class, string_types):
            widget_class = self.widget_classes.get(widget_class)
        widget = self._create_widget(widget_class, **kwargs)

        if widget not in self._widgets:
            self._widgets.append(widget)

        @widget.connect
        def on_close(event):
            self._widgets.remove(widget)

        if show:
            widget.show()

        return widget


#------------------------------------------------------------------------------
# View creator
#------------------------------------------------------------------------------

class ViewCreator(WidgetCreator):
    """Create views from a model."""

    _widget_model_classes = {
        'waveforms': vmk.WaveformViewModel,
        'features': vmk.FeatureViewModel,
        'correlograms': vmk.CorrelogramViewModel,
        'traces': vmk.TraceViewModel,
        'wizard': vmk.WizardViewModel,
        'stats': vmk.StatsViewModel,
    }

    def __init__(self, session):
        super(ViewCreator, self).__init__()
        self.session = session

    def _create_widget(self, vm_class, save_size_pos=True, **kwargs):
        """Create a new view model instance."""

        # Load parameters from the settings.
        params = vm_class.get_params(self.session.settings)
        params.update(kwargs)

        vm = vm_class(model=self.session.model,
                      store=self.session.cluster_store,
                      **params)

        self.session.connect(vm.on_open)

        @vm.connect
        def on_close(event):
            self.session.unconnect(vm.on_open)
            self._save_vm_params(vm, save_size_pos)

        return vm

    def _save_vm_params(self, vm, save_size_pos=True):
        """Save the parameters exported by a view model instance."""
        to_save = vm.exported_params(save_size_pos)
        for key, value in to_save.items():
            name = '{}_{}'.format(vm.name, key)
            self.session.settings[name] = value
            debug("Save {0}={1} for {2}.".format(name, value, vm.name))

    def save_view_params(self, save_size_pos=True):
        """Save all view parameters to user settings."""
        for vm in self._vms:
            self._save_vm_params(vm, save_size_pos=save_size_pos)


#------------------------------------------------------------------------------
# GUI creator
#------------------------------------------------------------------------------

class GUICreator(WidgetCreator):
    def __init__(self, session):
        super(GUICreator, self).__init__()
        self.session = session

    def add(self, config=None, show=True):
        """Add a new manual clustering GUI.

        Parameters
        ----------

        config : list
            A list of tuples `(name, kwargs)` describing the views in the GUI.
        show : bool
            Whether to show the newly-created GUI.

        Returns
        -------

        gui : ClusterManualGUI
            The GUI.

        """
        gui = ClusterManualGUI(self.session, config=config)
        self._guis.append(gui)

        @gui.main_window.on_close
        def on_close():
            if gui in self._guis:
                self._guis.remove(gui)
            self.session.view_creator.save_view_params()
            gs = gui._dock.save_geometry_state()
            self.session.settings['gui_state'] = gs
            self.session.settings['gui_view_count'] = gui._dock.view_counts()
            self.session.settings.save()

        if show:
            gui.show()

        return gui

    @property
    def guis(self):
        """List of GUIs."""
        return self._guis

    @property
    def gui(self):
        """The GUI if there is only one."""
        if len(self._guis) != 1:
            return
        return self._guis[0]
