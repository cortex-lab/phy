# -*- coding: utf-8 -*-
from __future__ import print_function

"""View creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ...ext.six import string_types
from ...plot.view_models.base import BaseViewModel
from ...plot.view_models.kwik import (WaveformViewModel,
                                      FeatureViewModel,
                                      CorrelogramViewModel,
                                      TraceViewModel,
                                      )


#------------------------------------------------------------------------------
# View creator
#------------------------------------------------------------------------------

class ViewCreator(object):
    view_model_classes = {
        'waveforms': WaveformViewModel,
        'features': FeatureViewModel,
        'correlograms': CorrelogramViewModel,
        'traces': TraceViewModel,
        # TODO: wizard
    }

    def __init__(self, session):
        self.session = session
        self._vms = []

    def _create_vm(self, name, save_size_pos=True, **kwargs):
        vm_class = self.view_model_classes[name]

        # Load the default options for that view.
        param_names = (BaseViewModel._imported_params +
                       vm_class._imported_params)
        params = {key: self.session.settings[name + '_' + key]
                  for key in param_names
                  if (name + '_' + key) in self.session.settings}
        params.update(kwargs)

        vm = vm_class(model=self.session.model,
                      store=self.session.cluster_store,
                      **params)

        self.session.connect(vm.on_open)

        @vm.view.connect
        def on_close(event):
            self.session.unconnect(vm.on_open)
            self._save_vm_params(vm, save_size_pos)
            vm.on_close()

        return vm

    def _save_vm_params(self, vm, save_size_pos=True):
        to_save = vm.exported_params(save_size_pos)
        for key, value in to_save.items():
            self.session.settings['{}_{}'.format(vm.name, key)] = value

    def save_view_params(self, save_size_pos=True):
        for vm in self._vms:
            self._save_vm_params(vm, save_size_pos=save_size_pos)

    def add(self, vm_or_name, show=True, **kwargs):
        if isinstance(vm_or_name, string_types):
            vm = self._create_vm(vm_or_name, **kwargs)
        else:
            vm = vm_or_name
        if vm not in self._vms:
            self._vms.append(vm)

        @vm.view.connect
        def on_close(event):
            self._vms.remove(vm)

        if show:
            vm.view.show()
        return vm

    def get(self, name):
        cls = self.view_model_classes[name]
        return [vm for vm in self._vms if isinstance(vm, cls)]
