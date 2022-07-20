from typing import Optional

from network import SingleNeuronPlot, SpikingNeuronNetwork
from gpu import RegisteredVBO


class NeuronInterface:

    def __init__(self, neuron_id, network: SpikingNeuronNetwork):

        self.id = neuron_id
        self.network = network

        self.plot = SingleNeuronPlot(network.plotting_config.voltage_plot_length)
        self.vbo_array: Optional[RegisteredVBO] = None

    @property
    def group(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_group_id_col]

    @property
    def type(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_neuron_type_col]

    def register_vbo(self):
        self.vbo_array = RegisteredVBO(self.plot.vbo, shape=self.plot.plot_data.pos.shape,
                                       device=self.network.GPU.device)


class IzhikevichNeuronsInterface(NeuronInterface):

    def __init__(self, neuron_id, network: SpikingNeuronNetwork):
        super().__init__(neuron_id, network)

    @property
    def pt(self):
        return self.network.GPU.N_states.pt[self.id]

    @pt.setter
    def pt(self, v):
        self.network.GPU.N_states.pt[self.id] = v

    @property
    def v(self):
        return self.network.GPU.N_states.v[self.id]

    @v.setter
    def v(self, v):
        self.network.GPU.N_states.v[self.id] = v

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

    @property
    def i(self):
        return self.network.GPU.N_states.i[self.id]

    @i.setter
    def i(self, v):
        self.network.GPU.N_states.i[self.id] = v

    def update_plot(self, t):
        self.vbo_array.tensor[t, 1] = self.v/100
        self.vbo_array.tensor[t + self.network.plotting_config.voltage_plot_length, 1] = self.i
