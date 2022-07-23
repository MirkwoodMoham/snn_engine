from copy import copy
from dataclasses import asdict
import torch
from typing import Callable, Optional, Union

from .signaling import SignalModel, StepSignal, DiscretePulseSignal

from network import IzhikevichPresets, SingleNeuronPlot, SpikingNeuronNetwork
from gpu import RegisteredVBO
from app.content.plots import SingleNeuronPlotWidget


class NeuronInterface:

    def __init__(self, neuron_id, network: SpikingNeuronNetwork,
                 plot_widget: Optional[SingleNeuronPlotWidget] = None):

        self._id = neuron_id
        self.network = network

        self.plot = SingleNeuronPlot(self.plot_length)

        self.plot_widget = plot_widget

        self.max_v = 0
        self.max_prev_i = 0

        self.first_plot_run = True

        self.b_current_injection = False
        self.current_injection_function: Optional[Union[SignalModel, Callable]] = None

        self.current_scale_reset_threshold_up = 0.8
        self.current_scale_reset_threshold_down = 0
        self.voltage_scale_reset_threshold_up = 5
        self.voltage_scale_reset_threshold_down = 0

        self.preset_dct = {'initial': None}
        self.preset_dct.update(asdict(self.presets))
        initial_preset = self.current_model
        self.preset_dct['initial'] = initial_preset
        self.preset_dct['custom'] = copy(initial_preset)

        self.network.GPU.N_states.preset_model(**self.preset_dct['initial'])

        self.set_current_injection(0, activate=True, mode='pulse_current', phase=0)

        self.custom_presets = []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i):
        self._id = i
        initial_preset = self.current_model
        if i in self.custom_presets:
            self.preset_dct['initial'] = initial_preset
            self.preset_dct['custom'] = copy(initial_preset)

    @property
    def current_model(self):
        model = {}
        for k in list(self.preset_dct.values())[1]:
            model[k] = getattr(self, k).item()
        return model

    def use_preset(self, key):
        print(self.preset_dct[key])
        for k, v in self.preset_dct[key].items():
            setattr(self, k, v)

    @property
    def presets(self):
        return self.network.GPU.N_states.presets

    @property
    def group(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_group_id_col]

    @property
    def type(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_neuron_type_col]

    def register_vbos(self):
        self.plot.init_cuda_attributes(self.network.GPU.device)
        self.plot.line.colors_gpu.tensor[:, 3] = 0

    def link_plot_widget(self, plot_widget: SingleNeuronPlotWidget):
        self.plot_widget = plot_widget
        self.plot_widget.view.add(self.plot)

    @property
    def v(self):
        return self.network.GPU.N_states.v[self.id]

    @v.setter
    def v(self, v):
        self.network.GPU.N_states.v[self.id] = v

    @property
    def i(self):
        return self.network.GPU.N_states.i[self.id]

    @i.setter
    def i(self, v):
        self.network.GPU.N_states.i[self.id] = v

    def set_current_injection(self, t, activate: bool, mode='step_current',
                              amplitude=15, frequency=100, effective_injection_period=1,
                              phase=10):

        self.b_current_injection = activate
        if mode == 'step_current':
            self.current_injection_function = StepSignal(
                t_start=t, amplitude=amplitude, phase=phase)
        elif mode == 'pulse_current':
            self.current_injection_function = DiscretePulseSignal(
                amplitude, phase, frequency, pulse_length=effective_injection_period)

    @property
    def i_prev(self):
        return self.network.GPU.N_states.i_prev[self.id]

    @i_prev.setter
    def i_prev(self, v):
        self.network.GPU.N_states.i_prev[self.id] = v

    @property
    def plot_length(self):
        return self.network.plotting_config.voltage_plot_length

    def _rescale_plot(self):

        if self.first_plot_run is True:
            self.first_plot_run = False

        max_v = torch.max(self.plot.line.pos_gpu.tensor[0: self.plot_length, 1]).item()
        if max_v > self.voltage_scale_reset_threshold_up:
            self.plot_widget.y_axis_right.scale *= (max_v / 2)

        max_i = torch.max(self.plot.line.pos_gpu.tensor[self.plot_length:, 1]).item()
        if max_i > self.current_scale_reset_threshold_up:
            self.plot_widget.y_axis.scale *= (max_i + 1)

        if (max_v > self.voltage_scale_reset_threshold_up) or (max_i > self.current_scale_reset_threshold_up):
            self.plot_widget.cam_reset()

    def update_plot(self, t, t_mod):

        if t_mod == 0:
            self.plot.line.colors_gpu.tensor[:, 3] = 0

        if (t_mod == (self.plot_length - 1)) is True:
            self._rescale_plot()

        self.plot.line.pos_gpu.tensor[t_mod, 1] = self.v / self.plot_widget.y_axis_right.scale
        self.plot.line.pos_gpu.tensor[t_mod + self.plot_length, 1] = \
            self.i_prev / self.plot_widget.y_axis.scale

        if self.b_current_injection is True:
            self.i = self.current_injection_function(t, t_mod)

        if t_mod > 0:
            self.plot.line.colors_gpu.tensor[t_mod, 3] = 1
            self.plot.line.colors_gpu.tensor[t_mod + self.plot_length, 3] = 1


class IzhikevichNeuronsInterface(NeuronInterface):

    def __init__(self, neuron_id, network: SpikingNeuronNetwork,
                 plot_widget: Optional[SingleNeuronPlotWidget] = None):
        super().__init__(neuron_id, network, plot_widget=plot_widget)

    @property
    def pt(self):
        return self.network.GPU.N_states.pt[self.id]

    @pt.setter
    def pt(self, v):
        self.network.GPU.N_states.pt[self.id] = v

    @property
    def a(self):
        return self.network.GPU.N_states.a[self.id]

    @a.setter
    def a(self, v):
        self.preset_dct['custom']['a'] = v
        # print(self.network.GPU.N_states.tensor[:, self.id])
        self.network.GPU.N_states.a[self.id] = v

    @property
    def b(self):
        return self.network.GPU.N_states.b[self.id]

    @b.setter
    def b(self, v):
        self.preset_dct['custom']['b'] = v
        self.network.GPU.N_states.b[self.id] = v

    @property
    def c(self):
        return self.network.GPU.N_states.c[self.id]

    @c.setter
    def c(self, v):
        self.preset_dct['custom']['c'] = v
        self.network.GPU.N_states.c[self.id] = v

    @property
    def d(self):
        return self.network.GPU.N_states.d[self.id]

    @d.setter
    def d(self, v):
        self.preset_dct['custom']['d'] = v
        self.network.GPU.N_states.d[self.id] = v
