import numpy as np
import pandas as pd
import time
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
    NeuronTypeGroup,
    NeuronTypeGroupConnection
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
    def __init__(self, N, S, D, G, T, n_N_states, config, n_neuron_types=2):
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
        self.N_stpd = (N, S * n_neuron_types)  # dtype=np.float32

        # GROUPS (location-based)

        self.G_pos = (G, 3)  # position of each location group; dtype=np.int32
        self.G_rep = (G, G)
        self.G_delay_counts = (G, D + 1)  # number of groups per delays; dtype=np.int32
        # G_neuron_counts-layout:
        #   - columns: location-based group-ids
        #   - row 0 to max(#neuron-types) - 1: #neurons in this group and type
        #   - row max(#neuron-types) to last row:
        #   for x in neuron-types:
        #       row0 = max(#neuron-types) + (D * (neuron-type) - 1)
        #       row1 = row0 + D
        #       rows from row0 to row1:
        #           #neurons for this group per delay of type 1
        #
        # Example:
        #
        # max(delay) = D - 1
        # #(neurons) = 20
        # #(neurons of neuron-type 1) = 4
        # #(neurons of neuron-type 2) = 16
        # n_neuron_types = 2
        #
        # tensor([[ 0,  0,  0,  2,  0,  0,  1,  1],  # sum = 4
        #         [ 4,  3,  1,  1,  3,  0,  3,  1],  # sum = 16
        #         [ 0,  0,  0,  2,  0,  0,  1,  1],  # row0-0
        #         [ 4,  4,  4,  2,  4,  4,  3,  3],  # row1-0
        #         [ 4,  3,  1,  1,  3,  0,  3,  1],  # row0-1
        #         [12, 13, 15, 15, 13, 16, 13, 15]]) # row1-1
        #
        # (group ids are 0-indexed)
        # (neuron type ids are 1-indexed)
        # #(neurons with (delay-from-group4==1)) = G_delay_counts[1:4]
        # #(neurons-type-x with (delay-from-group_y==z)) =
        #   G_delay_counts[n_neuron_types + (D * (x-1)) + z  :y]

        self.G_neuron_counts = (n_neuron_types + n_neuron_types * D, G)  # dtype=np.int32
        # self.G_neuron_typed_ccount = (1, 2 * G + 1)  # dtype=np.int32

        syn_count_shape = (n_neuron_types * (D + 1), G)
        # expected (cumulative) count of synapses per source types and delay (sources types: inhibitory or excitatory)
        self.G_exp_ccsyn_per_src_type_and_delay = syn_count_shape  # dtype=np.int32

        # expected cumulative sum of excitatory synapses per delay and per sink type
        # (sink types: inhibitory, excitatory)
        self.G_exp_exc_ccsyn_per_snk_type_and_delay = syn_count_shape  # dtype=np.int32

        self.G_conn_probs = (n_neuron_types * G, D)  # dtype=np.float32
        self.local_autapse_idcs = (3 * D, G)  # dtype=np.int32

        self.G_props = (config.n_group_properties, G)  # dtype=np.int32;  selected_p, thalamic input (on/off), ...


# noinspection PyPep8Naming
class NetworkGPUArrays:

    def __init__(self,
                 config: NetworkConfig,
                 pos_vbo: int,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 n_N_states: int,
                 T: int,
                 max_batch_size_mb: int):

        self.device = torch.device(device)
        # nbcuda.select_device(device)

        N = config.N
        S = config.S
        D = config.D
        G = config.G

        self._type_groups: list[NeuronTypeGroup] = list(type_group_dct.values())

        sh = NetworkDataShapes(N=N, S=S, D=D, G=G, T=T, n_N_states=n_N_states, config=config)

        self.N_pos = self._set_N_pos(shape=sh.N_pos, vbo=pos_vbo)

        self.N_G = self.t_i_zeros(sh.N_G)
        # t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in self._type_groups:
            self.N_G[g.start_idx:g.end_idx + 1, config.N_G_neuron_type_col] = g.ntype.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        self.G_neuron_counts = self.t_i_zeros(sh.G_neuron_counts)
        snn_construction_gpu.fill_N_G_group_id_and_G_neuron_count_per_type(
            N, G,
            N_pos=self.N_pos.data_ptr(),
            N_pos_shape=config.N_pos_shape,
            N_G=self.N_G.data_ptr(),
            N_G_n_cols=config.N_G_n_cols,
            N_G_neuron_type_col=config.N_G_neuron_type_col,
            N_G_group_id_col=config.N_G_group_id_col,
            G_shape=config.G_shape,
            G_neuron_counts=self.G_neuron_counts.data_ptr())

        self.G_neuron_typed_ccount = self.t_i_zeros((2, G))
        self.G_neuron_typed_ccount[0, :] = self.G_neuron_counts[0, :].ravel().cumsum(dim=0)
        self.G_neuron_typed_ccount[1, :] = self.G_neuron_counts[1, :].ravel().cumsum(dim=0)

        self.G_pos = self._set_G_pos(G=G,
                                     shape=sh.G_pos,
                                     N_pos_shape=config.N_pos_shape,
                                     G_shape=config.G_shape)

        self.validate_N_G(config=config)

        G_pos_distance = torch.cdist(self.G_pos, self.G_pos)
        self.G_delay_distance = ((D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

        self.G_group_delay_counts = self.t_i_zeros(sh.G_delay_counts)
        for d in range(D):
            self.G_group_delay_counts[:, d + 1] = (self.G_group_delay_counts[:, d]
                                                   + self.G_delay_distance.eq(d).sum(dim=1))
        self.G_group_delay_counts[:, 0] = torch.arange(G, device=self.device)

        self.G_rep = torch.sort(self.G_delay_distance, dim=1, stable=True).indices

        snn_construction_gpu.fill_G_neuron_count_per_delay(
            S=S, D=D, G=G,
            G_delay_distance=self.G_delay_distance.data_ptr(),
            G_neuron_counts=self.G_neuron_counts.data_ptr())

        self.validate_G_neuron_counts(D=D, G=G)

        self.G_conn_probs = self.t_f_zeros(sh.G_conn_probs)
        self.G_exp_ccsyn_per_src_type_and_delay = self.t_i_zeros(sh.G_exp_ccsyn_per_src_type_and_delay)
        self.G_exp_exc_ccsyn_per_snk_type_and_delay = self.t_i_zeros(sh.G_exp_exc_ccsyn_per_snk_type_and_delay)
        self._fill_syn_counts(S=S, D=D, G=G, type_group_conn_dct=type_group_conn_dct)

        # start = time.process_time()
        # # your code here
        # self.N_rep = torch.multinomial(self.t_f_zeros((N, N)) + 1, N)
        # print('GPU', time.process_time() - start)
        # start = time.process_time()
        # # your code here
        # self.N_rep = torch.multinomial(torch.zeros((N, N), dtype=torch.float32) + 1, N)
        # print('CPU:', time.process_time() - start)

        self.N_rep = self._set_N_rep(sh.N_rep, N, S, D, G, type_group_conn_dct)

        print()

    def t_i_zeros(self, shape):
        return torch.zeros(shape, dtype=torch.int32, device=self.device)

    def t_f_zeros(self, shape):
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _set_N_rep(
            self,
            N_rep_shape,
            N, S, D, G,
            type_group_conn_dct: dict[tuple[int, int], NeuronTypeGroupConnection]):

        n_rep = self.t_i_zeros(N_rep_shape)

        for gc in type_group_conn_dct.values():
            cn_row = gc.src_type_value - 1
            ct_row = (gc.snk_type_value - 1) * D + 2
            slice_ = self.G_neuron_counts[cn_row, :]
            counts = torch.repeat_interleave(self.G_neuron_counts[ct_row: ct_row+D, :].T,
                                             slice_, dim=0)

            if (gc.src.ntype == NeuronTypes.INHIBITORY) and (gc.snk.ntype == NeuronTypes.EXCITATORY):
                ccsyn = self.G_exp_ccsyn_per_src_type_and_delay[0: D + 1, :]
            elif (gc.src.ntype == NeuronTypes.EXCITATORY) and (gc.snk.ntype == NeuronTypes.INHIBITORY):
                ccsyn = self.G_exp_exc_ccsyn_per_snk_type_and_delay[0: D + 1, :]
            elif (gc.src.ntype == NeuronTypes.EXCITATORY) and (gc.snk.ntype == NeuronTypes.EXCITATORY):
                ccsyn = self.G_exp_exc_ccsyn_per_snk_type_and_delay[D + 1: 2 * (D + 1), :]
            else:
                raise ValueError

            autapse_indices = self.t_i_zeros((D, G))
            sort_keys = self.t_i_zeros(N_rep_shape)

            ccsyn = (torch.repeat_interleave(ccsyn.T, slice_, dim=0).diff(dim=1))

            # n_rep[
            #     gc.N_rep_loc[0]: gc.conn_shape[0],
            #     gc.N_rep_loc[1]: gc.conn_shape[1]
            # ] = (torch.arange(D, device=self.device)
            #      .unsqueeze(0)
            #      .repeat_interleave(gc.conn_shape[0], dim=0)
            #      .repeat_interleave(ccsyn, dim=1)
            #      )

            for d in range(D):
                pass

            print(counts, '\n')
            print(ccsyn, '\n')
            print()

        return 0

    def _set_N_pos(self, shape, vbo):

        N_pos = RegisteredGPUArray.from_vbo(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * 4, 4),
                                       dtype=np.float32, device=self.device))

        for g in self._type_groups:
            if g.ntype.value == NeuronTypes.INHIBITORY.value:
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

    def _fill_syn_counts(self, S, D, G, type_group_conn_dct: dict[tuple[int, int], NeuronTypeGroupConnection]):
        snn_construction_gpu.fill_G_exp_ccsyn_per_src_type_and_delay(
            S=S, D=D, G=G,
            G_neuron_counts=self.G_neuron_counts.data_ptr(),
            G_conn_probs=self.G_conn_probs.data_ptr(),
            G_exp_ccsyn_per_src_type_and_delay=self.G_exp_ccsyn_per_src_type_and_delay.data_ptr())

        exp_result = (self.t_f_zeros(G) + 1) * S

        for ntype_group in self._type_groups:
            ntype = ntype_group.ntype.value
            first_row = self.G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype - 1), :]
            if first_row.sum() != 0:
                print(first_row)
                raise AssertionError
            last_row = self.G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype - 1) + D, :]
            if ((last_row - exp_result).abs()).sum() != 0:
                print(last_row)
                print((last_row - exp_result).abs())
                raise AssertionError

        exc_syn_counts = []

        for gc in type_group_conn_dct.values():
            if gc.src_type_value == NeuronTypes.EXCITATORY.value:
                exc_syn_counts.append(len(gc))
        assert np.array(exc_syn_counts).cumsum()[-1] == S

        max_median_inh_targets_delay = 0
        max_inh_target_row = self.G_neuron_counts[2, :]
        max_median_exc_targets_delay = 0
        max_exc_target_row = self.G_neuron_counts[2 + D, :]

        last_row_inh = None
        last_row_exc = None

        for d in range(D):
            if d > 0:
                inh_targets = self.G_neuron_counts[2 + d, :]
                exc_targets = self.G_neuron_counts[2 + D + d, :]
                if max_median_inh_targets_delay < inh_targets.median():
                    max_median_inh_targets_delay = d
                    max_inh_target_row = inh_targets
                if max_median_exc_targets_delay < exc_targets.median():
                    max_median_exc_targets_delay = d
                    max_exc_target_row = exc_targets

            row_inh = d + 1
            row_exc = D + 2 + d

            exc_ccsyn = self.G_exp_ccsyn_per_src_type_and_delay[row_exc, :]
            self.G_exp_exc_ccsyn_per_snk_type_and_delay[row_inh, :] = exc_ccsyn * (exc_syn_counts[0]/S) + .5
            self.G_exp_exc_ccsyn_per_snk_type_and_delay[row_exc, :] = exc_ccsyn * (exc_syn_counts[1]/S) + .5
            if d == (D-1):
                last_row_inh = self.G_exp_exc_ccsyn_per_snk_type_and_delay[row_inh, :]
                last_row_exc = self.G_exp_exc_ccsyn_per_snk_type_and_delay[row_exc, :]

        inh_too_low_mask = last_row_inh < exc_syn_counts[0]
        exc_too_low_mask = last_row_exc < exc_syn_counts[1]

        if any(inh_too_low_mask) or any(exc_too_low_mask):
            print(self.G_exp_ccsyn_per_src_type_and_delay)
            print(self.G_exp_exc_ccsyn_per_snk_type_and_delay)
            raise AssertionError

    def validate_N_G(self, config: NetworkConfig):

        if (self.N_G[:, config.N_G_neuron_type_col] == 0).sum() > 0:
            raise AssertionError

        cond0 = (self.N_G[:-1, config.N_G_neuron_type_col]
                 .masked_select(self.N_G[:, config.N_G_neuron_type_col].diff() < 0).size(dim=0) > 0)
        cond1 = (self.N_G[:-1, config.N_G_group_id_col]
                 .masked_select(self.N_G[:, config.N_G_group_id_col].diff() < 0).size(dim=0) != 1)
        if cond0 or cond1:

            idcs1 = (self.N_G[:, config.N_G_group_id_col].diff() < 0).nonzero()
            df = pd.DataFrame(self.N_pos.tensor[:, :3].cpu().numpy())
            df[['0g', '1g', '2g']] = config.grid_pos
            df['N_G'] = self.N_G[:, 1].cpu().numpy()
            print(self.G_pos)
            print(df)
            df10 = df.iloc[int(idcs1[0]) - 2: int(idcs1[0]) + 3, :]
            df11 = df.iloc[int(idcs1[-1]) - 2: int(idcs1[-1]) + 3, :]
            print(df10)
            print(df11)
            raise AssertionError

    def validate_G_neuron_counts(self, D, G):

        max_ntype = 0
        for ntype_group in self._type_groups:
            ntype = ntype_group.ntype.value
            if self.G_neuron_counts[ntype - 1, :].sum() != len(ntype_group):
                raise AssertionError
            max_ntype = max(max_ntype, ntype)

        for ntype_group in self._type_groups:
            ntype = ntype_group.ntype.value
            min_row = max_ntype + D * (ntype - 1)
            max_row = min_row + D
            expected_result = (self.t_i_zeros(G) + 1) * len(ntype_group)
            if ((self.G_neuron_counts[min_row: max_row, :].sum(dim=0)
                 - expected_result).sum() != 0):
                print(self.G_neuron_counts)
                raise AssertionError

    @staticmethod
    def to_dataframe(tensor: torch.Tensor):
        return pd.DataFrame(tensor.cpu().numpy())


class SpikingNeuronNetwork:
    # noinspection PyPep8Naming
    def __init__(self, config: NetworkConfig, max_batch_size_mb: int, T: int = 2000):

        RenderedObject._grid_unit_shape = config.grid_unit_shape

        self.N = config.N
        self.S = config.S
        self.D = config.D
        self.G = config.G
        self.T = T

        self.n_N_states = 8

        self.config = config
        self.max_batch_size_mb = max_batch_size_mb

        self.type_group_dct: Dict[int, NeuronTypeGroup] = {}
        self.type_group_conn_dict: Dict[tuple[int, int], NeuronTypeGroupConnection] = {}
        self.next_group_id = 0

        g_inh = self.add_type_group(count=int(.2 * self.N), neuron_type=NeuronTypes.INHIBITORY)
        g_exc = self.add_type_group(count=self.N - len(g_inh), neuron_type=NeuronTypes.EXCITATORY)
        print()
        c_ihn_exc = self.add_type_group_conn(g_inh, g_exc, w0=-.49, exp_syn_counts=self.S)
        c_exc_inh = self.add_type_group_conn(g_exc, g_inh, w0=.51, exp_syn_counts=int((len(g_inh) / self.N) * self.S))
        c_exc_exc = self.add_type_group_conn(g_exc, g_exc, w0=.5, exp_syn_counts=self.S - len(c_exc_inh))

        self.sort_pos()

        self._scatter_plot = NetworkScatterPlot(self.config)

        self.GPU: Optional[NetworkGPUArrays] = None

        self._outer_grid = None
        self._selector_box = None
        self.validate()

    def validate(self):
        NeuronTypeGroup.validate(self.type_group_dct, N=self.N)
        NeuronTypeGroupConnection.validate(self.type_group_conn_dict, S=self.S)

    @property
    def type_groups(self):
        return self.type_group_dct.values()

    # noinspection PyPep8Naming
    def add_type_group(self, count, neuron_type):
        g = NeuronTypeGroup.from_count(self.next_group_id, count, self.S, neuron_type, self.type_group_dct)
        self.next_group_id += 1
        return g

    def add_type_group_conn(self, src, snk, w0, exp_syn_counts):
        c = NeuronTypeGroupConnection(src, snk, w0=w0, S=self.S,
                                      exp_syn_counts=exp_syn_counts,
                                      max_batch_size_mb=self.max_batch_size_mb,
                                      conn_dict=self.type_group_conn_dict)
        return c

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
            type_group_conn_dct=self.type_group_conn_dict,
            device=device,
            n_N_states=self.n_N_states,
            T=self.T,
            max_batch_size_mb=self.max_batch_size_mb
        )

    def sort_pos(self):
        """
        Sort neuron positions w.r.t. location-based groups and neuron types.
        """

        for g in self.type_groups:

            grid_pos = self.config.grid_pos[g.start_idx: g.end_idx + 1]

            p0 = grid_pos[:, 0].argsort(kind='stable')
            p1 = grid_pos[p0][:, 1].argsort(kind='stable')
            p2 = grid_pos[p0][p1][:, 2].argsort(kind='stable')

            self.config.grid_pos[g.start_idx: g.end_idx + 1] = grid_pos[p0][p1][p2]
            self.config.pos[g.start_idx: g.end_idx + 1] = self.config.pos[g.start_idx: g.end_idx + 1][p0][p1][p2]
            if self.N <= 100:
                print('\n', self.config.pos[g.start_idx:g.end_idx+1])
        print('\n')
