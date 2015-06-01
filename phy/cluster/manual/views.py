# -*- coding: utf-8 -*-
from __future__ import print_function

"""View creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ...ext.six import string_types
from ...plot.view_models.base import HTMLViewModel
from ...plot.view_models.kwik import (WaveformViewModel,
                                      FeatureViewModel,
                                      CorrelogramViewModel,
                                      TraceViewModel,
                                      )
from ...utils.logging import debug
from .static import _get_html


#------------------------------------------------------------------------------
# Wizard view model
#------------------------------------------------------------------------------

class WizardViewModel(HTMLViewModel):
    def _get_html(self, cluster_ids):
        params = self._wizard.get_panel_params()
        html = _get_html('wizard.html', **params)
        return html

    def _create_view(self, **kwargs):
        kwargs['html'] = self._get_html
        return super(WizardViewModel, self)._create_view(**kwargs)


#------------------------------------------------------------------------------
# View creator
#------------------------------------------------------------------------------

class ViewCreator(object):
    """Create views from a model."""

    # Mapping view names/view model classes.
    view_model_classes = {
        'waveforms': WaveformViewModel,
        'features': FeatureViewModel,
        'correlograms': CorrelogramViewModel,
        'traces': TraceViewModel,
        'wizard': WizardViewModel,
    }

    def __init__(self, session):
        self.session = session
        self._vms = []

    def _create_vm(self, vm_class, save_size_pos=True, **kwargs):
        """Create a new view model instance."""
        name = vm_class._view_name
        # Load the default options for that view.
        param_names = vm_class.imported_params()
        params = {key: self.session.settings[name + '_' + key]
                  for key in param_names
                  if (name + '_' + key) in self.session.settings}
        params.update(kwargs)

        vm = vm_class(model=self.session.model,
                      store=self.session.cluster_store,
                      wizard=self.session.wizard,
                      **params)

        self.session.connect(vm.on_open)

        @vm.connect
        def on_close(event):
            self.session.unconnect(vm.on_open)
            self._save_vm_params(vm, save_size_pos)
            vm.on_close()

        return vm

    def add(self, vm_cls, show=True, **kwargs):
        """Add a new view."""
        if isinstance(vm_cls, string_types):
            # If a string, the class can be either specified with the
            # `view_model` keyword argument, or it is one of the predefined
            # view models.
            vm_cls = (kwargs.get('view_model', None) or
                      self.view_model_classes.get(vm_cls))
        vm = self._create_vm(vm_cls, **kwargs)
        if vm not in self._vms:
            self._vms.append(vm)

        @vm.connect
        def on_close(event):
            self._vms.remove(vm)

        if show:
            vm.show()
        return vm

    def get(self, name=None):
        """Return the list of views of a given type."""
        if name is None:
            return self._vms
        return [vm for vm in self._vms if vm.name == name]

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
