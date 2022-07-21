import numpy as np
from scipy import signal
import torch
from typing import Callable, Optional

from network import SingleNeuronPlot, SpikingNeuronNetwork
from gpu import RegisteredVBO
from app.plot_widgets import SingleNeuronPlotWidget
from vispy.scene import ViewBox


class StepSignal:

    def __init__(self, t_start, amplitude, phase):

        self.amplitude = amplitude
        self.step_time = t_start + phase

    def __call__(self, t, t_mod):
        if t < self.step_time:
            return 0
        else:
            return self.amplitude


class PulseSignal:
    def __init__(self, amplitude, phase, frequency, pulse_length):
        self.amplitude = amplitude
        self.period = 1000 / frequency
        self.duty = round(pulse_length/self.period, 2)
        self.phase = phase

    def __call__(self, t, t_mod):
        return signal.square(((t + self.phase) * 2 * np.pi)/self.period, self.duty) * self.amplitude


class NeuronInterface:

    def __init__(self, neuron_id, network: SpikingNeuronNetwork,
                 plot_widget: Optional[SingleNeuronPlotWidget] = None):

        self.id = neuron_id
        self.network = network

        self.plot = SingleNeuronPlot(self.plot_length)
        self.vbo_array: Optional[RegisteredVBO] = None

        self.plot_widget = plot_widget

        self.max_v = 0
        self.max_prev_i = 0

        self.first_plot_run = True

        self.b_current_injection = False
        self.current_injection_function: Optional[Callable] = None

        self.current_scale_reset_threshold_up = 0.8
        self.current_scale_reset_threshold_down = 0
        self.voltage_scale_reset_threshold_up = 5
        self.voltage_scale_reset_threshold_down = 0

        self.set_current_injection(0, activate=True, mode='pulse_current', phase=50)

    @property
    def group(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_group_id_col]

    @property
    def type(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_neuron_type_col]

    def register_vbo(self):
        self.vbo_array = RegisteredVBO(self.plot.vbo, shape=self.plot.plot_data.pos.shape,
                                       device=self.network.GPU.device)

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
                              amplitude=25, frequency=100, effective_injection_period=5,
                              phase=0):

        self.b_current_injection = activate
        if mode == 'step_current':
            self.current_injection_function = StepSignal(
                t_start=t, amplitude=amplitude, phase=phase)
        elif mode == 'pulse_current':
            self.current_injection_function = PulseSignal(
                amplitude, phase, frequency, pulse_length=effective_injection_period
            )

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

        max_v = torch.max(self.vbo_array.tensor[0: self.plot_length, 1]).item()
        if max_v > self.voltage_scale_reset_threshold_up:
            self.plot_widget.y_axis_right.scale *= (max_v / 2)

        max_i = torch.max(self.vbo_array.tensor[self.plot_length:, 1]).item()
        if max_i > self.current_scale_reset_threshold_up:
            self.plot_widget.y_axis.scale *= (max_i + 1)

        if (max_v > self.voltage_scale_reset_threshold_up) or (max_i > self.current_scale_reset_threshold_up):
            self.plot_widget.cam_reset()

    def update_plot(self, t, t_mod):

        if (t_mod == (self.plot_length - 1)) is True:
            self._rescale_plot()

        self.vbo_array.tensor[t_mod, 1] = self.v / self.plot_widget.y_axis_right.scale
        self.vbo_array.tensor[t_mod + self.plot_length, 1] = \
            self.i_prev / self.plot_widget.y_axis.scale

        if self.b_current_injection is True:
            self.i = self.current_injection_function(t, t_mod)


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
        print(self.network.GPU.N_states.tensor[:, self.id])
        self.network.GPU.N_states.a[self.id] = v

    @property
    def b(self):
        return self.network.GPU.N_states.b[self.id]

    @b.setter
    def b(self, v):
        self.network.GPU.N_states.b[self.id] = v

    @property
    def c(self):
        return self.network.GPU.N_states.c[self.id]

    @c.setter
    def c(self, v):
        self.network.GPU.N_states.c[self.id] = v

    @property
    def d(self):
        return self.network.GPU.N_states.d[self.id]

    @d.setter
    def d(self, v):
        self.network.GPU.N_states.d[self.id] = v


