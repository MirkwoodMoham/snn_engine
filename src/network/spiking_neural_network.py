# from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
# import time
import torch
from typing import Dict, Optional
from vispy.scene import visuals

from .gpu_arrays import (
    GPUArrayConfig,
    RegisteredGPUArray,
    GPUArrayCollection
)
from .network_config import (
    NetworkConfig,
    PlottingConfig
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
    DefaultBox,
    VoltagePlot,
    FiringScatterPlot
)
from .network_states import IzhikevichModel, LocationGroupProperties

# noinspection PyUnresolvedReferences
from gpu import snn_construction_gpu, snn_simulation_gpu


class NetworkDataShapes:

    # noinspection PyPep8Naming
    def __init__(self,
                 config: NetworkConfig,
                 T: int,
                 n_N_states: int,
                 plotting_config: PlottingConfig,
                 n_neuron_types=2):

        self.N_pos = (config.N, config.vispy_scatter_plot_stride)
        self.N_rep = (config.N, config.S)
        self.N_G = (config.N, config.N_G_n_cols)

        # Network Representation
        self.N_rep = (config.N, config.S)  # float
        self.N_weights = self.N_rep  # np.float32
        self.N_delays = (config.D + 1, config.N)  # int
        self.N_fired = (1, config.N)  # int
        self.firing_times = (15, config.N)  # float
        self.firing_idcs = self.firing_times  # dtype=np.int32
        self.firing_counts = 2 * T  # dtype=np.int32

        # pt, u, v, a, b, c, d, I
        self.N_states = (n_N_states, config.N)  # dtype=np.float32
        self.N_stpd = (config.N, config.S * n_neuron_types)  # dtype=np.float32

        # GROUPS (location-based)

        self.G_pos = (config.G, 3)  # position of each location group; dtype=np.int32
        self.G_rep = (config.G, config.G)
        self.G_delay_counts = (config.G, config.D + 1)  # number of groups per delays; dtype=np.int32
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

        self.G_neuron_counts = (n_neuron_types + n_neuron_types * config.D, config.G)  # dtype=np.int32
        # self.G_neuron_typed_ccount = (1, 2 * G + 1)  # dtype=np.int32

        syn_count_shape = (n_neuron_types * (config.D + 1), config.G)
        # expected (cumulative) count of synapses per source types and delay (sources types: inhibitory or excitatory)
        self.G_exp_ccsyn_per_src_type_and_delay = syn_count_shape  # dtype=np.int32

        # expected cumulative sum of excitatory synapses per delay and per sink type
        # (sink types: inhibitory, excitatory)
        self.G_exp_exc_ccsyn_per_snk_type_and_delay = syn_count_shape  # dtype=np.int32

        self.G_conn_probs = (n_neuron_types * config.G, config.D)  # dtype=np.float32
        # self.relative_autapse_idcs = (3 * D, G)  # dtype=np.int32

        # dtype=np.int32;  selected_p, thalamic input (on/off), ...
        self.G_props = (LocationGroupProperties.__len__(), config.G)

        self.voltage_plot = (plotting_config.n_voltage_plots * plotting_config.voltage_plot_length, 2)
        self.firings_scatter_plot = (plotting_config.n_scatter_plots * plotting_config.scatter_plot_length,
                                     config.vispy_scatter_plot_stride)

        self.voltage_plot_map = (plotting_config.n_voltage_plots, 1)
        self.firings_scatter_plot_map = (plotting_config.n_scatter_plots, 1)

        assert config.G == self.G_props[1]
        assert config.N == self.N_states[1]


# noinspection PyPep8Naming
class NetworkGPUArrays(GPUArrayCollection):

    class PlottingGPUArrays(GPUArrayCollection):
        def __init__(self, device, shapes: NetworkDataShapes,
                     voltage_plot_vbo, firing_scatter_plot_vbo, bprint_allocated_memory):

            super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

            nbytes_float32 = 4

            self.voltage = RegisteredGPUArray.from_vbo(
                voltage_plot_vbo,
                config=GPUArrayConfig(shape=shapes.voltage_plot, strides=(shapes.voltage_plot[1] * nbytes_float32,
                                                                          nbytes_float32),
                                      dtype=np.float32, device=self.device))

            self.firings = RegisteredGPUArray.from_vbo(
                firing_scatter_plot_vbo, config=GPUArrayConfig(shape=shapes.firings_scatter_plot,
                                                               strides=(shapes.firings_scatter_plot[1] * nbytes_float32,
                                                                        nbytes_float32),
                                                               dtype=np.float32, device=self.device))

            self.voltage_map = self.izeros(shapes.voltage_plot_map)
            self.voltage_map[:, 0] = torch.arange(shapes.voltage_plot_map[0])

            self.firings_map = self.izeros(shapes.firings_scatter_plot_map)
            self.firings_map[:, 0] = torch.arange(shapes.firings_scatter_plot_map[0])

            print(self.voltage.to_dataframe)

    def __init__(self,
                 config: NetworkConfig,
                 pos_vbo: int,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 T: int,
                 voltage_plot_vbo: int,
                 firing_scatter_plot_vbo: int,
                 plotting_config: PlottingConfig,
                 model=IzhikevichModel,
                 ):

        super(NetworkGPUArrays, self).__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._type_group_dct = type_group_dct
        self._type_group_conn_dct = type_group_conn_dct

        shapes = NetworkDataShapes(config=config, T=T, n_N_states=model.__len__(), plotting_config=plotting_config)

        self.plotting_arrays = self.PlottingGPUArrays(device=device, shapes=shapes, voltage_plot_vbo=voltage_plot_vbo,
                                                      firing_scatter_plot_vbo=firing_scatter_plot_vbo,
                                                      bprint_allocated_memory=self.bprint_allocated_memory)

        self.curand_states = self._curand_states()
        self.N_pos = self._N_pos(shape=shapes.N_pos, vbo=pos_vbo)

        (self.N_G,
         self.G_neuron_counts,
         self.G_neuron_typed_ccount) = self._N_G_and_G_neuron_counts_1of2(shapes)

        self.G_pos = self._G_pos(shape=shapes.G_pos)
        self.G_delay_distance = self._G_delay_distance(self.G_pos)
        self._G_neuron_counts_2of2(self.G_delay_distance, self.G_neuron_counts)
        self.G_group_delay_counts = self._G_group_delay_counts(shapes.G_delay_counts, self.G_delay_distance)

        self.G_rep = torch.sort(self.G_delay_distance, dim=1, stable=True).indices.int()

        (self.G_conn_probs,
         self.G_exp_ccsyn_per_src_type_and_delay,
         self.G_exp_exc_ccsyn_per_snk_type_and_delay) = self._fill_syn_counts(shapes=shapes,
                                                                              G_neuron_counts=self.G_neuron_counts)

        (self.N_rep,
         self.N_delays) = self._set_N_rep(shapes=shapes, curand_states=self.curand_states)

        self.N_weights = self._set_N_weights(shapes.N_weights)

        self.N_states = model(shape=shapes.N_states, device=self.device,
                              types_tensor=self.N_G[:, self.config.N_G_neuron_type_col])

        self.G_props = LocationGroupProperties(
            shape=shapes.G_props, device=self.device, config=self.config)

        self.print_allocated_memory('end')

        self.Fired = self.fzeros(self.N)
        self.Firing_times = self.fzeros((15, self.N))
        self.Firing_idcs = self.izeros((15, self.N))
        self.Firing_counts = self.izeros((1, T * 2))
        # print()
        # print(self.N_states)

        self.Simulation = snn_simulation_gpu.SnnSimulation(
            N=self.N,
            G=self.config.G,
            S=self.S,
            D=self.config.D,
            T=T,
            n_voltage_plots=plotting_config.n_voltage_plots,
            voltage_plot_length=plotting_config.voltage_plot_length,
            voltage_plot_data=self.plotting_arrays.voltage.data_ptr(),
            voltage_plot_map=self.plotting_arrays.voltage_map.data_ptr(),
            n_scatter_plots=plotting_config.n_scatter_plots,
            scatter_plot_length=plotting_config.scatter_plot_length,
            scatter_plot_data=self.plotting_arrays.firings.data_ptr(),
            scatter_plot_map=self.plotting_arrays.firings_map.data_ptr(),
            curand_states_p=self.curand_states,
            N_pos=self.N_pos.data_ptr(),
            N_G=self.N_G.data_ptr(),
            G_props=self.G_props.data_ptr(),
            N_rep=self.N_rep.data_ptr(),
            N_delays=self.N_delays.data_ptr(),
            N_states=self.N_states.data_ptr(),
            N_weights=self.N_weights.data_ptr(),
            fired=self.Fired.data_ptr(),
            firing_times=self.Firing_times.data_ptr(),
            firing_idcs=self.Firing_idcs.data_ptr(),
            firing_counts=self.Firing_counts.data_ptr()
        )

        # print(self.Firing_idcs)

        # for i in range(2):
        #     self.update()

    def update(self):
        self.plotting_arrays.voltage.map()
        self.plotting_arrays.firings.map()
        self.Simulation.update(False)
        # self.plotting_arrays.voltage.unmap()
        # self.print_sim_state()

        # print()

    def print_sim_state(self):
        print('Fired:\n', self.Fired)
        print('Firing_idcs:\n', self.Firing_idcs)
        print('Firing_times:\n', self.Firing_times)
        print('Firing_counts:\n', self.Firing_counts)

    @property
    def config(self) -> NetworkConfig:
        return self._config

    @property
    def N(self):
        return self.config.N

    @property
    def S(self):
        return self.config.S

    @property
    def type_groups(self) -> list[NeuronTypeGroup]:
        return list(self._type_group_dct.values())

    @property
    def type_group_conns(self) -> list[NeuronTypeGroupConnection]:
        return list(self._type_group_conn_dct.values())

    def _N_G_and_G_neuron_counts_1of2(self, shapes):
        N_G = self.izeros(shapes.N_G)
        # t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in self.type_groups:
            N_G[g.start_idx:g.end_idx + 1, self.config.N_G_neuron_type_col] = g.ntype.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        G_neuron_counts = self.izeros(shapes.G_neuron_counts)
        snn_construction_gpu.fill_N_G_group_id_and_G_neuron_count_per_type(
            N=self.config.N, G=self.config.G,
            N_pos=self.N_pos.data_ptr(),
            N_pos_shape=self.config.N_pos_shape,
            N_G=N_G.data_ptr(),
            N_G_n_cols=self.config.N_G_n_cols,
            N_G_neuron_type_col=self.config.N_G_neuron_type_col,
            N_G_group_id_col=self.config.N_G_group_id_col,
            G_shape=self.config.G_shape,
            G_neuron_counts=G_neuron_counts.data_ptr())

        G_neuron_typed_ccount = self.izeros((2 * self.config.G + 1))
        G_neuron_typed_ccount[1:] = G_neuron_counts[: 2, :].ravel().cumsum(dim=0)

        return N_G, G_neuron_counts, G_neuron_typed_ccount

    def _G_group_delay_counts(self, shape, G_delay_distance):
        G_group_delay_counts = self.izeros(shape)
        for d in range(self.config.D):
            G_group_delay_counts[:, d + 1] = (G_group_delay_counts[:, d] + G_delay_distance.eq(d).sum(dim=1))
        return G_group_delay_counts

    def _G_neuron_counts_2of2(self, G_delay_distance, G_neuron_counts):
        snn_construction_gpu.fill_G_neuron_count_per_delay(
            S=self.config.S, D=self.config.D, G=self.config.G,
            G_delay_distance=G_delay_distance.data_ptr(),
            G_neuron_counts=G_neuron_counts.data_ptr())
        self.validate_G_neuron_counts()

    def _G_delay_distance(self, G_pos):
        G_pos_distance = torch.cdist(G_pos, G_pos)
        return ((self.config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

    def _curand_states(self):
        cu = snn_construction_gpu.CuRandStates(self.config.N).ptr()
        self.print_allocated_memory('curand_states')
        return cu

    def _N_pos(self, shape, vbo):
        nbytes_float32 = 4
        N_pos = RegisteredGPUArray.from_vbo(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes_float32, nbytes_float32),
                                       dtype=np.float32, device=self.device))

        for g in self.type_groups:
            if g.ntype.value == NeuronTypes.INHIBITORY.value:
                orange = torch.Tensor([1, .5, .2])
                N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        return N_pos

    def _G_pos(self, shape):
        groups = torch.arange(self._config.G, device=self.device)
        z = (groups / (self._config.G_shape[0] * self._config.G_shape[1])).floor()
        r = groups - z * (self._config.G_shape[0] * self._config.G_shape[1])
        y = (r / self._config.G_shape[0]).floor()
        x = r - y * self._config.G_shape[0]

        gpos = torch.zeros(shape, dtype=torch.float32, device=self.device)

        gpos[:, 0] = x * (self._config.N_pos_shape[0] / self._config.G_shape[0])
        gpos[:, 1] = y * (self._config.N_pos_shape[1] / self._config.G_shape[1])
        gpos[:, 2] = z * (self._config.N_pos_shape[2] / self._config.G_shape[2])

        self.validate_N_G(gpos)
        return gpos

    def _fill_syn_counts(self, shapes, G_neuron_counts):

        S, D, G = self.config.S, self.config.D, self.config.G

        G_conn_probs = self.fzeros(shapes.G_conn_probs)
        G_exp_ccsyn_per_src_type_and_delay = self.izeros(shapes.G_exp_ccsyn_per_src_type_and_delay)
        G_exp_exc_ccsyn_per_snk_type_and_delay = self.izeros(shapes.G_exp_exc_ccsyn_per_snk_type_and_delay)

        snn_construction_gpu.fill_G_exp_ccsyn_per_src_type_and_delay(
            S=S, D=D, G=G,
            G_neuron_counts=G_neuron_counts.data_ptr(),
            G_conn_probs=G_conn_probs.data_ptr(),
            G_exp_ccsyn_per_src_type_and_delay=G_exp_ccsyn_per_src_type_and_delay.data_ptr())

        exp_result = (self.fzeros(G) + 1) * S

        for ntype_group in self.type_groups:
            ntype = ntype_group.ntype.value
            first_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype - 1), :]
            if first_row.sum() != 0:
                print(first_row)
                raise AssertionError
            last_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype - 1) + D, :]
            if ((last_row - exp_result).abs()).sum() != 0:
                print(last_row)
                print((last_row - exp_result).abs())
                raise AssertionError

        exc_syn_counts = []

        for gc in self.type_group_conns:
            if gc.src_type_value == NeuronTypes.EXCITATORY.value:
                exc_syn_counts.append(len(gc))
        assert np.array(exc_syn_counts).cumsum()[-1] == S

        max_median_inh_targets_delay = -1
        # max_inh_target_row = G_neuron_counts[2, :]
        max_median_exc_targets_delay = -1
        # max_exc_target_row = G_neuron_counts[2 + D, :]

        last_row_inh = None
        last_row_exc = None

        autapse_mask = torch.zeros(G, dtype=torch.bool, device=self.device)
        exp_inh = self.izeros(G)
        exp_exc = self.izeros(G)
        row_exc_max = D + 2

        def add_mask(row, v, mask=None):
            if mask is not None:
                G_exp_exc_ccsyn_per_snk_type_and_delay[row, :][mask] = (
                        G_exp_exc_ccsyn_per_snk_type_and_delay[row, :] + v)[mask]
            else:
                G_exp_exc_ccsyn_per_snk_type_and_delay[row, :] += v

        def row_diff(row):
            return (G_exp_exc_ccsyn_per_snk_type_and_delay[row, :]
                    - G_exp_exc_ccsyn_per_snk_type_and_delay[row - 1, :])

        def inh_targets(delay):
            return G_neuron_counts[2 + delay, :]

        def exc_targets(delay):
            return G_neuron_counts[2 + D + delay, :]

        for d in range(D):

            row_inh = d + 1
            row_exc = D + 2 + d

            if d > 0:
                if max_median_inh_targets_delay < inh_targets(d).median():
                    max_median_inh_targets_delay = d
                    # max_inh_target_row = inh_targets(d)
                if max_median_exc_targets_delay < exc_targets(d).median():
                    max_median_exc_targets_delay = d
                    # max_exc_target_row = exc_targets(d)
                    row_exc_max = row_exc

            exc_ccsyn = G_exp_ccsyn_per_src_type_and_delay[row_exc, :]
            exp_inh[:] = exc_ccsyn * (exc_syn_counts[0]/S) + .5
            add_mask(row_inh, exp_inh)
            exp_exc[:] = exc_ccsyn * (exc_syn_counts[1]/S) + .5
            add_mask(row_exc, exp_exc)

            if d == 0:
                autapse_mask[:] = (exp_exc == exc_targets(d)) & (exp_exc > 0)
            add_mask(row_exc, -1, autapse_mask)

            exp_inh_count = row_diff(row_inh)
            exp_exc_count = row_diff(row_exc)

            inh_count_too_high_mask = (inh_targets(d) - exp_inh_count) < 0
            if inh_count_too_high_mask.any():
                if (d > 0) & ((row_diff(row_inh-1) < inh_targets(d-1)) & inh_count_too_high_mask).all():
                    add_mask(row_inh - 1, 1, mask=inh_count_too_high_mask)
                else:
                    add_mask(row_inh, - 1, mask=inh_count_too_high_mask)
                    # add_mask(row_inh + 1, 1, mask=inh_count_too_high_mask & (row_diff(row_inh + 1) < 0))

            exc_count_too_high_mask = (exc_targets(d) - exp_exc_count) < 0
            if exc_count_too_high_mask.any():
                if (d > 1) & ((row_diff(row_exc - 1) < exc_targets(d-1)) & exc_count_too_high_mask).all():
                    add_mask(row_exc - 1, 1, mask=exc_count_too_high_mask)
                else:
                    add_mask(row_exc, - 1, mask=exc_count_too_high_mask)
                    # add_mask(row_exc + 1, 1, mask=exc_count_too_high_mask)

            if d == (D-1):
                if (max_median_exc_targets_delay == 0) or (row_exc_max == D + 2):
                    raise AssertionError
        # print(G_exp_exc_ccsyn_per_snk_type_and_delay)
        for d in range(max_median_exc_targets_delay, D):
            row_inh = d + 1
            row_exc = D + 2 + d
            add_mask(row_exc, 1, mask=autapse_mask)

            exp_inh_count = row_diff(row_inh)
            exp_exc_count = row_diff(row_exc)

            if (exp_inh_count < 0).any():
                raise ValueError(f'({row_exc}){(exp_inh_count < 0).sum()}')
            if (exp_exc_count < 0).any():
                raise ValueError(f'({row_exc}){(exp_exc_count < 0).sum()}')

            if ((inh_targets(d) - exp_inh_count) < 0).any():
                raise ValueError(f'({row_exc}){((inh_targets(d) - exp_inh_count) < 0).sum()}')
            if ((exc_targets(d) - exp_exc_count) < 0).any():
                raise ValueError(f'({row_exc}){((exc_targets(d) - exp_exc_count) < 0).sum()}')

            if d == (D-1):
                if (max_median_exc_targets_delay == 0) or (row_exc_max == D + 2):
                    raise AssertionError
                last_row_inh = G_exp_exc_ccsyn_per_snk_type_and_delay[row_inh, :]
                last_row_exc = G_exp_exc_ccsyn_per_snk_type_and_delay[row_exc, :]

        # print(G_exp_exc_ccsyn_per_snk_type_and_delay)

        inh_neq_exp_mask = last_row_inh != exc_syn_counts[0]
        exc_neq_exp_mask = last_row_exc != exc_syn_counts[1]

        if any(inh_neq_exp_mask) or any(exc_neq_exp_mask):
            print('G_neuron_counts:\n', G_neuron_counts, '\n')
            print('G_exp_ccsyn_per_src_type_and_delay:\n', G_exp_ccsyn_per_src_type_and_delay, '\n')
            print('G_exp_exc_ccsyn_per_snk_type_and_delay:\n', G_exp_exc_ccsyn_per_snk_type_and_delay)
            raise AssertionError

        return G_conn_probs, G_exp_ccsyn_per_src_type_and_delay, G_exp_exc_ccsyn_per_snk_type_and_delay

    def _set_N_rep(self, shapes, curand_states):

        N, S, D, G = self.config.N, self.config.S, self.config.D, self.config.G

        torch.cuda.empty_cache()
        self.print_allocated_memory('syn_counts')

        N_delays = self.izeros(shapes.N_delays)
        N_rep = self.izeros(shapes.N_rep)
        torch.cuda.empty_cache()
        self.print_allocated_memory('N_rep')
        sort_keys = self.izeros(shapes.N_rep)
        self.print_allocated_memory('sort_keys')

        def cc_syn_(gc_):
            t = self.izeros((D + 1, G))
            if (gc_.src.ntype == NeuronTypes.INHIBITORY) and (gc_.snk.ntype == NeuronTypes.EXCITATORY):
                t[:, :] = self.G_exp_ccsyn_per_src_type_and_delay[0: D + 1, :]
            elif (gc_.src.ntype == NeuronTypes.EXCITATORY) and (gc_.snk.ntype == NeuronTypes.INHIBITORY):
                t[:, :] = self.G_exp_exc_ccsyn_per_snk_type_and_delay[0: D + 1, :]
            elif (gc_.src.ntype == NeuronTypes.EXCITATORY) and (gc_.snk.ntype == NeuronTypes.EXCITATORY):
                t[:, :] = self.G_exp_exc_ccsyn_per_snk_type_and_delay[D + 1: 2 * (D + 1), :]
            else:
                raise ValueError
            return t

        for i, gc in enumerate(self.type_group_conns):
            # cn_row = gc.src_type_value - 1
            ct_row = (gc.snk_type_value - 1) * D + 2
            # slice_ = self.G_neuron_counts[cn_row, :]
            # counts = torch.repeat_interleave(self.G_neuron_counts[ct_row: ct_row+D, :].T, slice_, dim=0)

            ccn_idx_src = G * (gc.src_type_value - 1)
            ccn_idx_snk = G * (gc.snk_type_value - 1)

            G_autapse_indices = self.izeros((D, G))
            G_relative_autapse_indices = self.izeros((D, G))
            cc_syn = cc_syn_(gc)

            snn_construction_gpu.fill_N_rep(
                N=N, S=S, D=D, G=G,
                curand_states=curand_states,
                N_G=self.N_G.data_ptr(),
                cc_src=self.G_neuron_typed_ccount[ccn_idx_src: ccn_idx_src + G + 1].data_ptr(),
                cc_snk=self.G_neuron_typed_ccount[ccn_idx_snk: ccn_idx_snk + G + 1].data_ptr(),
                G_rep=self.G_rep.data_ptr(),
                G_neuron_counts=self.G_neuron_counts[ct_row: ct_row+D, :].data_ptr(),
                G_group_delay_counts=self.G_group_delay_counts.data_ptr(),
                G_autapse_indices=G_autapse_indices.data_ptr(),
                G_relative_autapse_indices=G_relative_autapse_indices.data_ptr(),
                has_autapses=ccn_idx_src == ccn_idx_snk,
                gc_location=gc.location,
                gc_conn_shape=gc.conn_shape,
                cc_syn=cc_syn.data_ptr(),
                N_delays=N_delays.data_ptr(),
                sort_keys=sort_keys.data_ptr(),
                N_rep=N_rep.data_ptr(),
                verbose=False)

            if G_autapse_indices[1:, :].sum() != -(G_autapse_indices.shape[0] - 1) * G_autapse_indices.shape[1]:
                raise AssertionError

            if (G_relative_autapse_indices[1:, :].sum()
                    != -(G_relative_autapse_indices.shape[0] - 1) * G_relative_autapse_indices.shape[1]):
                raise AssertionError
            # if i == 1:
            #     print()

            self.print_allocated_memory(f'{gc.id}')
            # print()
        del G_autapse_indices
        del G_relative_autapse_indices
        torch.cuda.empty_cache()
        # df_unsorted = self.to_dataframe(N_rep)
        # print(df_unsorted)
        # print(sort_keys)
        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=sort_keys.data_ptr(), N_rep=N_rep.data_ptr())
        # df_sorted = self.to_dataframe(N_rep)

        # print(df_sorted)
        for i, gc in enumerate(self.type_group_conns):

            ct_row = (gc.snk_type_value - 1) * D + 2
            # slice_ = self.G_neuron_counts[cn_row, :]
            # counts = torch.repeat_interleave(self.G_neuron_counts[ct_row: ct_row+D, :].T, slice_, dim=0)

            ccn_idx_src = G * (gc.src_type_value - 1)
            ccn_idx_snk = G * (gc.snk_type_value - 1)
            cc_syn = cc_syn_(gc)

            snn_construction_gpu.reindex_N_rep(
                N=N, S=S, D=D, G=G,
                N_G=self.N_G.data_ptr(),
                cc_src=self.G_neuron_typed_ccount[ccn_idx_src: ccn_idx_src + G + 1].data_ptr(),
                cc_snk=self.G_neuron_typed_ccount[ccn_idx_snk: ccn_idx_snk + G + 1].data_ptr(),
                G_rep=self.G_rep.data_ptr(),
                G_neuron_counts=self.G_neuron_counts[ct_row: ct_row+D, :].data_ptr(),
                G_group_delay_counts=self.G_group_delay_counts.data_ptr(),
                gc_location=gc.location,
                gc_conn_shape=gc.conn_shape,
                cc_syn=cc_syn.data_ptr(),
                N_delays=N_delays.data_ptr(),
                sort_keys=sort_keys.data_ptr(),
                N_rep=N_rep.data_ptr(),
                verbose=False)

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=sort_keys.data_ptr(), N_rep=N_rep.data_ptr())
        # print(N_rep)
        del sort_keys
        self.print_allocated_memory(f'sorted')

        return N_rep, N_delays

    def _set_N_weights(self, shape):
        weights = self.fzeros(shape)
        for gc in self.type_group_conns:
            weights[gc.location[0]: gc.location[0] + gc.conn_shape[0],
                    gc.location[1]: gc.location[1] + gc.conn_shape[1]] = gc.w0
        return weights

    def validate_N_G(self, G_pos):

        cfg = self._config
        if (self.N_G[:, cfg.N_G_neuron_type_col] == 0).sum() > 0:
            raise AssertionError

        cond0 = (self.N_G[:-1, cfg.N_G_neuron_type_col]
                 .masked_select(self.N_G[:, cfg.N_G_neuron_type_col].diff() < 0).size(dim=0) > 0)
        cond1 = (self.N_G[:-1, cfg.N_G_group_id_col]
                 .masked_select(self.N_G[:, cfg.N_G_group_id_col].diff() < 0).size(dim=0) != 1)
        if cond0 or cond1:

            idcs1 = (self.N_G[:, cfg.N_G_group_id_col].diff() < 0).nonzero()
            df = pd.DataFrame(self.N_pos.tensor[:, :3].cpu().numpy())
            df[['0g', '1g', '2g']] = cfg.grid_pos
            df['N_G'] = self.N_G[:, 1].cpu().numpy()
            print(G_pos)
            print(df)
            df10 = df.iloc[int(idcs1[0]) - 2: int(idcs1[0]) + 3, :]
            df11 = df.iloc[int(idcs1[-1]) - 2: int(idcs1[-1]) + 3, :]
            print(df10)
            print(df11)
            raise AssertionError

    def validate_G_neuron_counts(self):
        D, G = self.config.D, self.config.G
        max_ntype = 0
        for ntype_group in self.type_groups:
            ntype = ntype_group.ntype.value
            if self.G_neuron_counts[ntype - 1, :].sum() != len(ntype_group):
                raise AssertionError
            max_ntype = max(max_ntype, ntype)

        for ntype_group in self.type_groups:
            ntype = ntype_group.ntype.value
            min_row = max_ntype + D * (ntype - 1)
            max_row = min_row + D
            expected_result = (self.izeros(G) + 1) * len(ntype_group)
            if ((self.G_neuron_counts[min_row: max_row, :].sum(dim=0)
                 - expected_result).sum() != 0):
                print(self.G_neuron_counts)
                raise AssertionError


class SpikingNeuronNetwork:
    # noinspection PyPep8Naming
    def __init__(self,
                 network_config: NetworkConfig,
                 plotting_config: PlottingConfig,
                 max_batch_size_mb: int,
                 T: int = 2000,
                 model=IzhikevichModel,
                 ):

        RenderedObject._grid_unit_shape = network_config.grid_unit_shape

        self.T = T
        self._network_config: NetworkConfig = network_config
        self._plotting_config: PlottingConfig = plotting_config
        self.model = model
        self.max_batch_size_mb = max_batch_size_mb

        self.type_group_dct: Dict[int, NeuronTypeGroup] = {}
        self.type_group_conn_dict: Dict[tuple[int, int], NeuronTypeGroupConnection] = {}
        self.next_group_id = 0

        g_inh = self.add_type_group(count=int(.2 * self.config.N), neuron_type=NeuronTypes.INHIBITORY)
        g_exc = self.add_type_group(count=self.config.N - len(g_inh), neuron_type=NeuronTypes.EXCITATORY)

        self.add_type_group_conn(g_inh, g_exc, w0=-.49, exp_syn_counts=self.config.S)
        c_exc_inh = self.add_type_group_conn(g_exc, g_inh, w0=.51,
                                             exp_syn_counts=max(int((len(g_inh) / self.config.N) * self.config.S), 1))
        self.add_type_group_conn(g_exc, g_exc, w0=.5, exp_syn_counts=self.config.S - len(c_exc_inh))

        self.sort_pos()

        self._network_scatter_plot = NetworkScatterPlot(self.config)

        self.GPU: Optional[NetworkGPUArrays] = None

        self._outer_grid: Optional[visuals.Box] = None
        self._selector_box: Optional[SelectorBox] = None
        self._voltage_plot: Optional[VoltagePlot] = None
        self._firing_scatter_plot: Optional[VoltagePlot] = None

        self.validate()

    @property
    def config(self):
        return self._network_config

    @property
    def plotting_config(self):
        return self._plotting_config

    def validate(self):
        NeuronTypeGroup.validate(self.type_group_dct, N=self.config.N)
        NeuronTypeGroupConnection.validate(self.type_group_conn_dict, S=self.config.S)

    @property
    def type_groups(self):
        return self.type_group_dct.values()

    # noinspection PyPep8Naming
    def add_type_group(self, count, neuron_type):
        g = NeuronTypeGroup.from_count(self.next_group_id, count, self.config.S, neuron_type, self.type_group_dct)
        self.next_group_id += 1
        return g

    def add_type_group_conn(self, src, snk, w0, exp_syn_counts):
        c = NeuronTypeGroupConnection(src, snk, w0=w0, S=self.config.S,
                                      exp_syn_counts=exp_syn_counts,
                                      max_batch_size_mb=self.max_batch_size_mb,
                                      conn_dict=self.type_group_conn_dict)
        return c

    @property
    def scatter_plot(self) -> NetworkScatterPlot:
        return self._network_scatter_plot

    @property
    def outer_grid(self) -> visuals.Box:
        if self._outer_grid is None:
            self._outer_grid: visuals.Box = DefaultBox(shape=self.config.N_pos_shape,
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

    @property
    def voltage_plot(self):
        if self._voltage_plot is None:
            self._voltage_plot = VoltagePlot(n_plots=self.plotting_config.n_voltage_plots,
                                             plot_length=self.plotting_config.voltage_plot_length)
        return self._voltage_plot

    @property
    def firing_scatter_plot(self):
        if self._firing_scatter_plot is None:
            self._firing_scatter_plot = FiringScatterPlot(n_plots=self.plotting_config.n_scatter_plots,
                                                          plot_length=self.plotting_config.scatter_plot_length)
        return self._firing_scatter_plot

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device):
        self.GPU = NetworkGPUArrays(
            config=self.config,
            pos_vbo=self._network_scatter_plot.pos_vbo,
            type_group_dct=self.type_group_dct,
            type_group_conn_dct=self.type_group_conn_dict,
            device=device,
            T=self.T,
            voltage_plot_vbo=self.voltage_plot.pos_vbo,
            firing_scatter_plot_vbo=self.firing_scatter_plot.pos_vbo,
            plotting_config=self.plotting_config,
            model=self.model)

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
            if self.config.N <= 100:
                print('\n', self.config.pos[g.start_idx:g.end_idx+1])
        if self.config.N <= 100:
            print()

    def update(self):
        self.GPU.update()
