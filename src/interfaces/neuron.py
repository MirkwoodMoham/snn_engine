from network import SpikingNeuronNetwork


class NeuronInterface:

    def __init__(self, neuron_id, network: SpikingNeuronNetwork):

        self.id = neuron_id
        self.network = network

    @property
    def group(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_group_id_col]

    @property
    def type(self):
        return self.network.GPU.N_G[self.id, self.network.network_config.N_G_neuron_type_col]


class IzhikevichNeuronsInterface(NeuronInterface):

    def __init__(self, neuron_id, network: SpikingNeuronNetwork):
        super().__init__(neuron_id, network)

    @property
    def pt(self):
        return self.network.GPU.N_states.pt[self.id]

    @pt.setter
    def pt(self, v):
        print(self.network.GPU.N_states._tensor[:, self.id])
        self.network.GPU.N_states.pt[self.id] = v
        print(self.network.GPU.N_states._tensor[:, self.id])

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
        print(self.network.GPU.N_states._tensor[:, self.id])
        self.network.GPU.N_states.a[self.id] = v
        print(self.network.GPU.N_states._tensor[:, self.id])

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
