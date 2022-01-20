import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
from vispy.scene import visuals

from .gpu_array import (
    GPUArrayConfig,
    RegisteredGPUArray
)
from .network_config import (
    NetworkConfig
)
from .network_structures import (
    NeuronTypes,
    NeuronTypeGroup
)
from .rendered_objects import (
    NetworkScatterPlot,
    RenderedObject,
    SelectorBox,
    default_box
)

from gpu import func2, snn_construction_gpu


class NetworkDataShapes:

    # noinspection PyPep8Naming
    def __init__(self, N, S, D, G, T, n_N_states, config):
        self.N_pos = (N, config.N_pos_n_cols)
        self.N_rep = (N, S)
        self.N_G = (N, config.N_G_n_cols)

        # Network Representation
        self.N_rep = (N, S)  # float
        self.N_weights = self.N_rep  # np.float32
        self.N_delays = (D + 1, N)  # int
        self.N_fired = (1, N)  # int
        self.firing_times = (15, N)  # float
        self.firing_idcs = self.firing_times  # dtype=np.int32
        self.firing_counts = 2 * T  # dtype=np.int32

        # pt, u, v, a, b, c, d, I
        self.N_states = (n_N_states, N)  # dtype=np.float32
        self.N_stpd = (N, S * 2)  # dtype=np.float32

        # GROUPS (location-based)

        self.G_pos = (G, 3)  # position of each location group; dtype=np.int32
        self.G_rep = (G, G)
        self.G_delay_counts = (G, D + 1)  # number of groups per delays; dtype=np.int32
        self.G_neuron_counts = (2 + 2 * D, G)  # dtype=np.int32
        # self.G_neuron_typed_ccount = (1, 2 * G + 1)  # dtype=np.int32

        syn_count_shape = (2 * (D + 1), G)
        # expected (cumulative) count of synapses per source types and delay (sources types: inhibitory or excitatory)
        self.G_exp_ccsyn_per_src_type_and_delay = syn_count_shape  # dtype=np.int32

        # expected cumulative sum of excitatory synapses per delay and per sink type
        # (sink types: inhibitory, excitatory)
        self.G_exp_exc_syn_ccount_per_snk_type_and_delay = syn_count_shape  # dtype=np.int32

        self.G_conn_probs = (2 * G, D)  # dtype=np.float32
        self.local_autapse_idcs = (3 * D, G)  # dtype=np.int32

        self.G_props = (config.n_group_properties, G)  # dtype=np.int32;  selected_p, thalamic input (on/off), ...


# noinspection PyPep8Naming
class NetworkGPUArrays:

    def __init__(self,
                 config: NetworkConfig,
                 pos_vbo: int,
                 type_group_dct: dict,
                 device: int,
                 n_N_states: int,
                 T: int):

        self.device = torch.device(device)
        # nbcuda.select_device(device)

        N = config.N
        S = config.S
        D = config.D
        G = config.G

        sh = NetworkDataShapes(N=N, S=S, D=D, G=G, T=T, n_N_states=n_N_states, config=config)

        def t_i_zeros(shape):
            return torch.zeros(shape, dtype=torch.int32, device=self.device)

        def t_f_zeros(shape):
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

        self.N_pos = self._set_N_pos(shape=sh.N_pos,
                                     vbo=pos_vbo,
                                     type_group_dct=type_group_dct)

        self.N_G = t_i_zeros(sh.N_G)
        t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in type_group_dct.values():
            self.N_G[g.start_idx:g.end_idx + 1, config.N_G_neuron_type_col] = g.type.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        self.G_neuron_counts = t_i_zeros(sh.G_neuron_counts)
        snn_construction_gpu.fill_N_G_group_id_and_G_neuron_count_per_type(
            N, G,
            N_pos=self.N_pos.data_ptr(),
            # N_pos_shape=config.N_pos_shape,
            N_G=self.N_G.data_ptr(),
            N_G_n_cols=config.N_G_n_cols,
            N_G_neuron_type_col=config.N_G_neuron_type_col,
            N_G_group_id_col=config.N_G_group_id_col,
            G_shape=config.G_shape,
            G_neuron_counts=self.G_neuron_counts.data_ptr())

        self.G_neuron_typed_ccount = self.G_neuron_counts[:2, :].ravel().cumsum(0)

        self.G_pos = self._set_G_pos(G=G,
                                     shape=sh.G_pos,
                                     N_pos_shape=config.N_pos_shape,
                                     G_shape=config.G_shape)

        self.validate_N_G(config=config)

        G_pos_distance = torch.cdist(self.G_pos, self.G_pos)
        self.G_delay_distance = ((D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

        self.G_exp_ccsyn_per_src_type_and_delay = t_i_zeros(sh.G_exp_ccsyn_per_src_type_and_delay)

        self.G_group_delay_counts = t_i_zeros(sh.G_delay_counts)
        for d in range(D):
            self.G_group_delay_counts[:, d + 1] = self.G_delay_distance.eq(d).sum(dim=1)

        self.G_rep = torch.sort(self.G_delay_distance, dim=1, stable=True).indices

        self.G_conn_probs = self._set_G_conn_probs(sh.G_conn_probs)

        snn_construction_gpu.fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay_python(
            S=S,
            G=G,
            D=D,
            G_delay_distance=self.G_delay_distance.data_ptr(),
            G_conn_probs=self.G_conn_probs.data_ptr(),
            G_neuron_counts=self.G_neuron_counts.data_ptr(),
            G_synapse_count_per_delay=self.G_exp_ccsyn_per_src_type_and_delay.data_ptr())

        self.validate_G_neuron_counts(config=config, type_group_dct=type_group_dct)

        print()

    def _set_G_conn_probs(self, shape):
        """
        Set the connection probabilities such that the expected value for the total synapse count
        is approximately S
        """
        gprob = torch.zeros(shape, dtype=torch.float32, device=self.device)

        def conn_prob():
            pass

        for delay_count in self.G_group_delay_counts.unique(dim=0):
            delay_count_cpu = delay_count.cpu().numpy()
            print()

        return gprob

    def _set_N_pos(self, shape, vbo, type_group_dct):

        N_pos = RegisteredGPUArray.from_vbo(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * 4, 4),
                                       dtype=np.float32, device=self.device))

        for g in type_group_dct.values():
            if g.type.value == NeuronTypes.INHIBITORY.value:
                orange = torch.Tensor([1, .5, .2])
                N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        return N_pos

    def _set_G_pos(self, G, shape, N_pos_shape, G_shape):
        groups = torch.arange(G, device=self.device)
        z = (groups / (G_shape[0] * G_shape[1])).floor()
        r = groups - z * (G_shape[0] * G_shape[1])
        y = (r / G_shape[0]).floor()
        x = r - y * G_shape[0]

        gpos = torch.zeros(shape, dtype=torch.float32, device=self.device)

        gpos[:, 0] = x * (N_pos_shape[0] / G_shape[0])
        gpos[:, 1] = y * (N_pos_shape[1] / G_shape[1])
        gpos[:, 2] = z * (N_pos_shape[2] / G_shape[2])

        return gpos

    def validate_N_G(self, config: NetworkConfig):
        cond0 = (self.N_G[:-1, config.N_G_neuron_type_col]
                 .masked_select(self.N_G[:, config.N_G_neuron_type_col].diff() < 0).size(dim=0) > 0)
        cond1 = (self.N_G[:-1, config.N_G_group_id_col]
                 .masked_select(self.N_G[:, config.N_G_group_id_col].diff() < 0).size(dim=0) != 1)
        if cond0 or cond1:
            df = pd.DataFrame(self.N_pos.tensor[:, :3].cpu().numpy())
            df[['0g', '1g', '2g']] = config.grid_pos
            df['N_G'] = self.N_G[:, 1].cpu().numpy()
            print(self.G_pos)
            print(df)
            raise AssertionError

    def validate_G_neuron_counts(
            self, config: NetworkConfig, type_group_dct: Dict[int, NeuronTypeGroup]):

        print(self.G_neuron_counts)




class SpikingNeuronNetwork:
    # noinspection PyPep8Naming
    def __init__(self, config: NetworkConfig, T: int = 2000):

        RenderedObject._grid_unit_shape = config.grid_unit_shape

        self.N = config.N
        self.S = config.S
        self.D = config.D
        self.G = config.G
        self.T = T

        self.n_N_states = 8

        self.config = config

        self.type_group_dct: Dict[int, NeuronTypeGroup] = {}
        n_inhN = int(.2 * self.N)
        self.add_type_group(ID=0, count=n_inhN, neuron_type=NeuronTypes.INHIBITORY)
        self.add_type_group(ID=1, count=self.N - n_inhN, neuron_type=NeuronTypes.EXCITATORY)
        self.sort_pos()
        print('\n', self.config.pos[self.type_group_dct[0].start_idx:self.type_group_dct[0].end_idx+1], '\n')
        print(self.config.pos[self.type_group_dct[1].start_idx:self.type_group_dct[1].end_idx+1], '\n')
        self._scatter_plot = NetworkScatterPlot(self.config)

        self.GPU: Optional[NetworkGPUArrays] = None

        self._outer_grid = None
        self._selector_box = None
        print()

        print()

    # noinspection PyPep8Naming
    # @property
    # def CPU(self):
    #     if not self._CPU:
    #         self._CPU = NetworkCPUArrays(self.config, T=self.T, n_N_states=self.n_N_states)
    #     return self._CPU

    @property
    def type_groups(self):
        return self.type_group_dct.values()

    # noinspection PyPep8Naming
    def add_type_group(self, ID, count, neuron_type):
        NeuronTypeGroup.from_count(ID, count, self.S, neuron_type, self.type_group_dct)

    @property
    def scatter_plot(self) -> NetworkScatterPlot:
        return self._scatter_plot

    @property
    def outer_grid(self) -> visuals.Box:
        if self._outer_grid is None:
            self._outer_grid: visuals.Box = default_box(shape=self.config.N_pos_shape,
                                                        scale=[.99, .99, .99],
                                                        segments=self.config.G_shape)
            # self._outer_grid.set_gl_state('translucent', blend=True, depth_test=True)
            # self._outer_grid.mesh.set_gl_state('translucent', blend=True, depth_test=True)
            self._outer_grid.visible = False

        return self._outer_grid

    @property
    def selector_box(self) -> visuals.Box:
        if self._selector_box is None:
            self._selector_box = SelectorBox(self.config.grid_unit_shape)
        return self._selector_box

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device):
        self.GPU = NetworkGPUArrays(
            config=self.config,
            pos_vbo=self._scatter_plot.vbo,
            type_group_dct=self.type_group_dct,
            device=device,
            n_N_states=self.n_N_states,
            T=self.T)

    def sort_pos(self):
        """
        Sort neuron positions w.r.t. location-based groups and neuron types.
        """

        for g in self.type_group_dct.values():

            grid_pos = self.config.grid_pos[g.start_idx: g.end_idx + 1]

            p0 = grid_pos[:, 0].argsort(kind='stable')
            p1 = grid_pos[p0][:, 1].argsort(kind='stable')
            p2 = grid_pos[p0][p1][:, 2].argsort(kind='stable')

            self.config.grid_pos[g.start_idx: g.end_idx + 1] = self.config.grid_pos[g.start_idx: g.end_idx + 1][p0][p1][p2]
            self.config.pos[g.start_idx: g.end_idx + 1] = self.config.pos[g.start_idx: g.end_idx + 1][p0][p1][p2]
