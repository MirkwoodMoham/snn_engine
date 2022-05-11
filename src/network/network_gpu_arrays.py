import numpy as np
import pandas as pd
import torch

from network.network_array_shapes import NetworkArrayShapes
from network.network_config import BufferCollection, NetworkConfig, PlottingConfig
from network.network_states import IzhikevichModel, LocationGroupProperties
from network.network_structures import NeuronTypeGroup, NeuronTypeGroupConnection, NeuronTypes
from .network_grid import NetworkGrid
from .neurons import Neurons

# noinspection PyUnresolvedReferences
from gpu import (
    snn_construction_gpu,
    snn_simulation_gpu,
    GPUArrayConfig,
    RegisteredGPUArray,
    GPUArrayCollection
)


# noinspection PyPep8Naming
class NetworkGPUArrays(GPUArrayCollection):

    class PlottingGPUArrays(GPUArrayCollection):
        def __init__(self, device, shapes: NetworkArrayShapes,
                     voltage_plot_vbo, firing_scatter_plot_vbo, bprint_allocated_memory):

            super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

            nbytes_float32 = 4

            self.voltage = RegisteredGPUArray.from_buffer(
                voltage_plot_vbo,
                config=GPUArrayConfig(shape=shapes.voltage_plot, strides=(shapes.voltage_plot[1] * nbytes_float32,
                                                                          nbytes_float32),
                                      dtype=np.float32, device=self.device))

            self.firings = RegisteredGPUArray.from_buffer(
                firing_scatter_plot_vbo, config=GPUArrayConfig(shape=shapes.firings_scatter_plot,
                                                               strides=(shapes.firings_scatter_plot[1] * nbytes_float32,
                                                                        nbytes_float32),
                                                               dtype=np.float32, device=self.device))

            self.voltage_map = self.izeros(shapes.voltage_plot_map)
            self.voltage_map[:] = torch.arange(shapes.voltage_plot_map)

            self.firings_map = self.izeros(shapes.firings_scatter_plot_map)
            self.firings_map[:] = torch.arange(shapes.firings_scatter_plot_map)

            # print(self.voltage.to_dataframe)

    def __init__(self,
                 config: NetworkConfig,
                 grid: NetworkGrid,
                 neurons: Neurons,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 T: int,
                 plotting_config: PlottingConfig,
                 buffers: BufferCollection,
                 model=IzhikevichModel,
                 ):

        super(NetworkGPUArrays, self).__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._plotting_config: PlottingConfig = plotting_config
        self._type_group_dct = type_group_dct
        self._type_group_conn_dct = type_group_conn_dct

        shapes = NetworkArrayShapes(config=config, T=T, n_N_states=model.__len__(), plotting_config=plotting_config,
                                    n_neuron_types=len(NeuronTypes))

        # self.selector_box = self._selector_box(())

        self.plotting_arrays = self.PlottingGPUArrays(device=device, shapes=shapes, voltage_plot_vbo=buffers.voltage,
                                                      firing_scatter_plot_vbo=buffers.firings,
                                                      bprint_allocated_memory=self.bprint_allocated_memory)

        self.curand_states = self._curand_states()
        self.N_pos: RegisteredGPUArray = self._N_pos(shape=shapes.N_pos, vbo=buffers.N_pos)

        (self.N_G,
         self.G_neuron_counts,
         self.G_neuron_typed_ccount) = self._N_G_and_G_neuron_counts_1of2(shapes, grid, neurons)

        self.neuron_ids = torch.arange(config.N).to(device=self.device)
        self.group_indices = self._set_group_indices()

        self.G_pos: RegisteredGPUArray = self._G_pos(shape=shapes.G_pos, vbo=buffers.selected_group_boxes_vbo)
        self.G_delay_distance = self._G_delay_distance(self.G_pos)
        self._G_neuron_counts_2of2(self.G_delay_distance, self.G_neuron_counts)
        self.G_group_delay_counts = self._G_group_delay_counts(shapes.G_delay_counts, self.G_delay_distance)

        self.G_rep = torch.sort(self.G_delay_distance, dim=1, stable=True).indices.int()

        (self.G_conn_probs,
         self.G_exp_ccsyn_per_src_type_and_delay,
         self.G_exp_exc_ccsyn_per_snk_type_and_delay) = self._fill_syn_counts(shapes=shapes,
                                                                              G_neuron_counts=self.G_neuron_counts)

        (self.N_rep,
         self.N_delays) = self._N_rep_and_N_delays(shapes=shapes, curand_states=self.curand_states)

        self.N_rep_groups_cpu = self._N_rep_groups_cpu()

        self.N_weights = self._N_weights(shapes.N_weights)
        self.print_allocated_memory('weights')

        self.N_states = model(shape=shapes.N_states, device=self.device,
                              types_tensor=self.N_G[:, self._config.N_G_neuron_type_col])

        self.G_props = LocationGroupProperties(shape=shapes.G_props, device=self.device, config=self._config,
                                               grid=grid,
                                               select_ibo=buffers.selected_group_boxes_ibo)

        self.Fired = self.fzeros(self._config.N)
        self.Firing_times = self.fzeros((15, self._config.N))
        self.Firing_idcs = self.izeros((15, self._config.N))
        self.Firing_counts = self.izeros((1, T * 2))
        # print()
        # print(self.N_states)

        self.Simulation = snn_simulation_gpu.SnnSimulation(
            N=self._config.N,
            G=self._config.G,
            S=self._config.S,
            D=self._config.D,
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

        self.G_swap_tensor = self._G_swap_tensor()
        self.swap_group_synapses(torch.from_numpy(grid.forward_groups).to(device=self.device))
        self.N_states.use_preset('rs', self.selected_neuron_mask(self._config.sensory_groups))

    def update(self):
        self.plotting_arrays.voltage.map()
        self.plotting_arrays.firings.map()
        self.Simulation.update(False)

    def print_sim_state(self):
        print('Fired:\n', self.Fired)
        print('Firing_idcs:\n', self.Firing_idcs)
        print('Firing_times:\n', self.Firing_times)
        print('Firing_counts:\n', self.Firing_counts)

    @property
    def type_groups(self) -> list[NeuronTypeGroup]:
        return list(self._type_group_dct.values())

    @property
    def type_group_conns(self) -> list[NeuronTypeGroupConnection]:
        return list(self._type_group_conn_dct.values())

    def _N_G_and_G_neuron_counts_1of2(self, shapes, grid: NetworkGrid, neurons: Neurons):
        N_G = self.izeros(shapes.N_G)
        # t_neurons_ids = torch.arange(self.N_G.shape[0], device='cuda')  # Neuron Id
        for g in self.type_groups:
            N_G[g.start_idx:g.end_idx + 1, self._config.N_G_neuron_type_col] = g.ntype.value  # Set Neuron Type

        # rows[0, 1]: inhibitory count, excitatory count,
        # rows[2 * D]: number of neurons per delay (post_synaptic type: inhibitory, excitatory)
        G_neuron_counts = self.izeros(shapes.G_neuron_counts)
        snn_construction_gpu.fill_N_G_group_id_and_G_neuron_count_per_type(
            N=self._config.N, G=self._config.G,
            N_pos=self.N_pos.data_ptr(),
            N_pos_shape=self._config.N_pos_shape,
            N_G=N_G.data_ptr(),
            N_G_n_cols=self._config.N_G_n_cols,
            N_G_neuron_type_col=self._config.N_G_neuron_type_col,
            N_G_group_id_col=self._config.N_G_group_id_col,
            G_shape=grid.segmentation,
            G_neuron_counts=G_neuron_counts.data_ptr())

        G_neuron_typed_ccount = self.izeros((2 * self._config.G + 1))
        G_neuron_typed_ccount[1:] = G_neuron_counts[: 2, :].ravel().cumsum(dim=0)
        self.validate_N_G(N_G, neurons)
        return N_G, G_neuron_counts, G_neuron_typed_ccount

    def _G_group_delay_counts(self, shape, G_delay_distance):
        G_group_delay_counts = self.izeros(shape)
        for d in range(self._config.D):
            G_group_delay_counts[:, d + 1] = (G_group_delay_counts[:, d] + G_delay_distance.eq(d).sum(dim=1))
        return G_group_delay_counts

    def _G_neuron_counts_2of2(self, G_delay_distance, G_neuron_counts):
        snn_construction_gpu.fill_G_neuron_count_per_delay(
            S=self._config.S, D=self._config.D, G=self._config.G,
            G_delay_distance=G_delay_distance.data_ptr(),
            G_neuron_counts=G_neuron_counts.data_ptr())
        self.validate_G_neuron_counts()

    def _G_delay_distance(self, G_pos: RegisteredGPUArray):
        G_pos_distance = torch.cdist(G_pos.tensor, G_pos.tensor)
        return ((self._config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

    def _curand_states(self):
        cu = snn_construction_gpu.CuRandStates(self._config.N).ptr()
        self.print_allocated_memory('curand_states')
        return cu

    @property
    def _N_pos_face_color(self):
        return self.N_pos.tensor[:, 7:11]

    @property
    def _N_pos_edge_color(self):
        return self.N_pos.tensor[:, 3:7]

    def _N_pos(self, shape, vbo):
        nbytes_float32 = 4
        N_pos = RegisteredGPUArray.from_buffer(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes_float32, nbytes_float32),
                                       dtype=np.float32, device=self.device))

        for g in self.type_groups:
            if g.ntype == NeuronTypes.INHIBITORY:
                orange = torch.Tensor([1, .5, .2])
                N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        return N_pos

    def _selector_box(self, shape, vbo):
        nbytes_float32 = 4
        b = RegisteredGPUArray.from_buffer(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes_float32, nbytes_float32),
                                       dtype=np.float32, device=self.device))
        return b

    def _G_pos(self, shape, vbo) -> RegisteredGPUArray:
        # groups = torch.arange(self._config.G, device=self.device)
        # z = (groups / (self._config.G_shape[0] * self._config.G_shape[1])).floor()
        # r = groups - z * (self._config.G_shape[0] * self._config.G_shape[1])
        # y = (r / self._config.G_shape[0]).floor()
        # x = r - y * self._config.G_shape[0]
        #
        # gpos = torch.zeros(shape, dtype=torch.float32, device=self.device)
        #
        # gpos[:, 0] = x * (self._config.N_pos_shape[0] / self._config.G_shape[0])
        # gpos[:, 1] = y * (self._config.N_pos_shape[1] / self._config.G_shape[1])
        # gpos[:, 2] = z * (self._config.N_pos_shape[2] / self._config.G_shape[2])

        G_pos = RegisteredGPUArray.from_buffer(
            vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * 4, 4),
                                       dtype=np.float32, device=self.device))

        # self.validate_N_G()
        return G_pos

    def _fill_syn_counts(self, shapes, G_neuron_counts):

        S, D, G = self._config.S, self._config.D, self._config.G

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
            first_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype_group.ntype - 1), :]
            if first_row.sum() != 0:
                print(first_row)
                raise AssertionError
            last_row = G_exp_ccsyn_per_src_type_and_delay[(D + 1) * (ntype_group.ntype - 1) + D, :]
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

    def _N_rep_and_N_delays(self, shapes, curand_states):

        N, S, D, G = self._config.N, self._config.S, self._config.D, self._config.G

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

            # self.print_allocated_memory(f'{gc.id}')
            # print()
        del G_autapse_indices
        del G_relative_autapse_indices
        torch.cuda.empty_cache()

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=sort_keys.data_ptr(), N_rep=N_rep.data_ptr())

        for i, gc in enumerate(self.type_group_conns):

            ct_row = (gc.snk_type_value - 1) * D + 2

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

    def _N_rep_groups_cpu(self):
        N_rep_groups = self.N_rep.clone()
        for ntype in NeuronTypes:
            for g in range(self._config.G):
                col = 2 * (ntype - 1)
                mask = ((self.N_rep >= self.group_indices[g, col])
                        & (self.N_rep <= self.group_indices[g, col + 1]))
                N_rep_groups[mask] = g
        return N_rep_groups.cpu()

    def _N_weights(self, shape):
        weights = self.fzeros(shape)
        for gc in self.type_group_conns:
            weights[gc.location[0]: gc.location[0] + gc.conn_shape[0],
                    gc.location[1]: gc.location[1] + gc.conn_shape[1]] = gc.w0
        return weights

    def set_weight(self, w, x0=0, x1=None, y0=0, y1=None):
        self.N_weights[x0: x1 if x1 is not None else self._config.N,
                       y0: y1 if y1 is not None else self._config.S] = w

    def set_src_group_weights(self, groups, w):
        selected = self.selected_neuron_mask(groups)
        self.N_weights[selected, :] = w

    def _set_group_indices(self):
        indices = self.izeros((self._config.G, 4)) - 1
        for ntype in NeuronTypes:
            for g in range(self._config.G):
                ids = self.neuron_ids[
                    ((self.N_G[:, self._config.N_G_neuron_type_col] == ntype)
                     & (self.N_G[:, self._config.N_G_group_id_col] == g))]
                col = 2 * (ntype - 1)
                if len(ids) > 0:
                    indices[g, col] = ids[0]
                    indices[g, col + 1] = ids[-1]
        return indices

    def validate_N_G(self, N_G, neurons: Neurons):
        if (N_G[:, self._config.N_G_neuron_type_col] == 0).sum() > 0:
            raise AssertionError

        cond0 = (N_G[:-1, self._config.N_G_neuron_type_col]
                 .masked_select(N_G[:, self._config.N_G_neuron_type_col].diff() < 0).size(dim=0) > 0)
        cond1 = (N_G[:-1, self._config.N_G_group_id_col]
                 .masked_select(N_G[:, self._config.N_G_group_id_col].diff() < 0).size(dim=0) != 1)
        if cond0 or cond1:

            idcs1 = (N_G[:, self._config.N_G_group_id_col].diff() < 0).nonzero()
            df = pd.DataFrame(self.N_pos.tensor[:, :3].cpu().numpy())
            df[['0g', '1g', '2g']] = neurons.shape
            df['N_G'] = N_G[:, 1].cpu().numpy()
            # print(G_pos)
            print(df)
            df10 = df.iloc[int(idcs1[0]) - 2: int(idcs1[0]) + 3, :]
            df11 = df.iloc[int(idcs1[-1]) - 2: int(idcs1[-1]) + 3, :]
            print(df10)
            print(df11)
            raise AssertionError

    def validate_G_neuron_counts(self):
        D, G = self._config.D, self._config.G
        max_ntype = 0
        for ntype_group in self.type_groups:
            if self.G_neuron_counts[ntype_group.ntype - 1, :].sum() != len(ntype_group):
                raise AssertionError
            max_ntype = max(max_ntype, ntype_group.ntype)

        for ntype_group in self.type_groups:
            min_row = max_ntype + D * (ntype_group.ntype - 1)
            max_row = min_row + D
            expected_result = (self.izeros(G) + 1) * len(ntype_group)
            if ((self.G_neuron_counts[min_row: max_row, :].sum(dim=0)
                 - expected_result).sum() != 0):
                print(self.G_neuron_counts)
                raise AssertionError

    def selected_neuron_mask(self, groups):
        self.N_states.selected[:] = 0
        for g in groups:
            self.N_states.selected += (self.N_G[:, self._config.N_G_group_id_col] == g)
        self.N_states.selected = self.N_states.selected > 0
        return self.N_states.selected

    def select(self, groups):
        return self.neuron_ids[self.selected_neuron_mask(groups)]

    def actualize_plot_map(self, groups):
        selected_neurons = self.neuron_ids[self.selected_neuron_mask(groups)]

        n_selected = len(selected_neurons)

        n_voltage_plots = min(n_selected, self._plotting_config.n_voltage_plots)
        self.plotting_arrays.voltage_map[: n_voltage_plots] = selected_neurons[: n_voltage_plots]

        n_scatter_plots = min(n_selected, self._plotting_config.n_scatter_plots)
        self.plotting_arrays.firings_map[: n_scatter_plots] = selected_neurons[: n_scatter_plots]

        if n_selected < self._plotting_config.n_voltage_plots:
            pass
        if n_selected < self._plotting_config.n_scatter_plots:
            pass

    def redirect_synapses(self, groups, direction, rate):
        pass

    def _G_swap_tensor(self):
        max_neurons_per_group = self.G_neuron_counts[:len(NeuronTypes)].sum(axis=0).max().item()
        return self.izeros((10 * max_neurons_per_group, max_neurons_per_group)) - 1

    def swap_group_synapses(self, groups):
        n_groups = groups.shape[1]
        if n_groups > 10:
            raise ValueError
        neurons = self.select(groups=groups[0])
        self.Simulation.swap_groups(neurons.data_ptr(), groups.data_ptr(), 3, 2)