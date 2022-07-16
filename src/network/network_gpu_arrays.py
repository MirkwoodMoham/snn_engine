import numpy as np
import pandas as pd
import torch

from network.network_array_shapes import NetworkArrayShapes
from network.network_config import BufferCollection, NetworkConfig, PlottingConfig
from network.network_states import IzhikevichModel, LocationGroupProperties, LocationGroupFlags, G2GInfoArrays
from network.network_structures import NeuronTypeGroup, NeuronTypeGroupConnection, NeuronTypes
from .network_grid import NetworkGrid
from .neurons import Neurons

# noinspection PyUnresolvedReferences
from gpu import (
    snn_construction_gpu,
    snn_simulation_gpu,
    GPUArrayConfig,
    RegisteredVBO,
    GPUArrayCollection
)
# from app import App


class PlottingGPUArrays(GPUArrayCollection):

    def __init__(self, plotting_config: PlottingConfig,
                 device, shapes: NetworkArrayShapes,
                 buffers: BufferCollection,
                 bprint_allocated_memory,
                 app):

        super().__init__(device=device, bprint_allocated_memory=bprint_allocated_memory)

        self.buffers = buffers
        self.registered_buffers = []
        self.shapes = shapes

        self.voltage = None
        self.voltage_group_line_pos = None
        self.voltage_group_line_colors = None

        self.firings = None
        self.firings_group_line_pos = None
        self.firings_group_line_colors = None

        if app.neuron_plot_window is not None:
            app.neuron_plot_window.scatter_plot_sc.set_current()
        if self.buffers.voltage is not None:
            self.init_voltage_plot_arrays()

        if app.neuron_plot_window is not None:
            app.neuron_plot_window.scatter_plot_sc.set_current()
        if self.buffers.firings is not None:
            self.init_firings_plot_arrays()

        app.main_window.scene_3d.set_current()

        self.voltage_map = self.izeros(shapes.voltage_plot_map)
        self.voltage_map[:] = torch.arange(shapes.voltage_plot_map)

        self.firings_map = self.izeros(shapes.firings_scatter_plot_map)
        self.firings_map[:] = torch.arange(shapes.firings_scatter_plot_map)

        self.voltage_plot_slots = torch.arange(shapes.voltage_plot_map + 1, device=self.device)
        self.firings_plot_slots = torch.arange(shapes.firings_scatter_plot_map + 1, device=self.device)

        # print(self.voltage.to_dataframe)
        if buffers.group_firing_counts_plot_single0 is not None:
            self.group_firing_counts_plot_single0 = RegisteredVBO(buffers.group_firing_counts_plot_single0,
                                                                  (plotting_config.scatter_plot_length * 2, 2),
                                                                  self.device)
            self.registered_buffers.append(self.group_firing_counts_plot_single0)

        if buffers.group_firing_counts_plot_single1 is not None:
            self.group_firing_counts_plot_single1 = RegisteredVBO(buffers.group_firing_counts_plot_single1,
                                                                  (plotting_config.scatter_plot_length * 2, 2),
                                                                  self.device)
            self.registered_buffers.append(self.group_firing_counts_plot_single1)

    # noinspection DuplicatedCode
    def init_voltage_plot_arrays(self):
        self.voltage = RegisteredVBO(self.buffers.voltage, self.shapes.voltage_plot, self.device)

        self.voltage_group_line_pos = RegisteredVBO(self.buffers.voltage_group_line_pos,
                                                    self.shapes.plot_group_line_pos,
                                                    self.device)
        self.voltage_group_line_colors = RegisteredVBO(self.buffers.voltage_group_line_colors,
                                                       self.shapes.plot_group_line_colors,
                                                       self.device)
        self.registered_buffers.append(self.voltage)
        self.registered_buffers.append(self.voltage_group_line_pos)
        self.registered_buffers.append(self.voltage_group_line_colors)

    # noinspection DuplicatedCode
    def init_firings_plot_arrays(self):
        self.firings = RegisteredVBO(self.buffers.firings, self.shapes.firings_scatter_plot, self.device)

        self.firings_group_line_pos = RegisteredVBO(self.buffers.firings_group_line_pos,
                                                    self.shapes.plot_group_line_pos, self.device)

        self.firings_group_line_colors = RegisteredVBO(self.buffers.firings_group_line_colors,
                                                       self.shapes.plot_group_line_colors,
                                                       self.device)
        self.registered_buffers.append(self.firings)
        self.registered_buffers.append(self.firings_group_line_pos)
        self.registered_buffers.append(self.firings_group_line_colors)


# noinspection PyPep8Naming
class NetworkGPUArrays(GPUArrayCollection):

    def __init__(self,
                 config: NetworkConfig,
                 grid: NetworkGrid,
                 neurons: Neurons,
                 type_group_dct: dict,
                 type_group_conn_dct: dict,
                 device: int,
                 T: int,
                 shapes: NetworkArrayShapes,
                 plotting_config: PlottingConfig,
                 buffers: BufferCollection,
                 app,
                 model=IzhikevichModel,
                 ):

        super().__init__(device=device, bprint_allocated_memory=config.N > 1000)

        self._config: NetworkConfig = config
        self._plotting_config: PlottingConfig = plotting_config
        self._type_group_dct = type_group_dct
        self._type_group_conn_dct = type_group_conn_dct

        self.registered_buffers = []

        self.plotting_arrays = PlottingGPUArrays(plotting_config,
                                                 device=device, shapes=shapes, buffers=buffers,
                                                 bprint_allocated_memory=self.bprint_allocated_memory,
                                                 app=app)

        self.registered_buffers += self.plotting_arrays.registered_buffers
        self.curand_states = self._curand_states()
        self.N_pos: RegisteredVBO = self._N_pos(shape=shapes.N_pos, vbo=buffers.N_pos)

        (self.N_G,
         self.G_neuron_counts,
         self.G_neuron_typed_ccount) = self._N_G_and_G_neuron_counts_1of2(shapes, grid, neurons)

        self.neuron_ids = torch.arange(config.N).to(device=self.device)
        self.group_ids = torch.arange(config.G).to(device=self.device)

        self.group_indices = self._set_group_indices()

        self.G_pos: RegisteredVBO = RegisteredVBO(buffers.selected_group_boxes_vbo, shapes.G_pos, self.device)
        self.registered_buffers.append(self.G_pos)

        self.G_flags = LocationGroupFlags(self._config.G, device=self.device, grid=grid,
                                          select_ibo=buffers.selected_group_boxes_ibo)
        self.registered_buffers.append(self.G_flags.selected_array)

        self.g2g_info_arrays = G2GInfoArrays(self._config, self.group_ids,
                                             self.G_flags, self.G_pos,
                                             device=device, bprint_allocated_memory=self.bprint_allocated_memory)

        self._G_neuron_counts_2of2(self.g2g_info_arrays.G_delay_distance, self.G_neuron_counts)
        self.G_group_delay_counts = self._G_group_delay_counts(shapes.G_delay_counts,
                                                               self.g2g_info_arrays.G_delay_distance)

        self.G_rep = torch.sort(self.g2g_info_arrays.G_delay_distance, dim=1, stable=True).indices.int()

        (self.G_conn_probs,
         self.G_exp_ccsyn_per_src_type_and_delay,
         self.G_exp_exc_ccsyn_per_snk_type_and_delay) = self._fill_syn_counts(shapes=shapes,
                                                                              G_neuron_counts=self.G_neuron_counts)
        self.N_rep_buffer = self.izeros(shapes.N_rep)
        self.print_allocated_memory('N_rep_buffer')

        (self.N_rep,
         self.N_delays) = self._N_rep_and_N_delays(shapes=shapes, curand_states=self.curand_states)

        self.N_rep_pre_synaptic_idx = self.izeros(shapes.N_rep_inv)
        self.N_rep_pre_synaptic_counts = self.izeros(self._config.N + 1)
        self.print_allocated_memory('N_rep_inv')

        self.N_rep_groups_cpu = self._N_rep_groups_cpu()

        self.N_weights = self._N_weights(shapes.N_weights)
        self.print_allocated_memory('weights')

        self.N_states = model(self._config.N, device=self.device,
                              types_tensor=self.N_G[:, self._config.N_G_neuron_type_col])

        self.G_props = LocationGroupProperties(self._config.G, device=self.device, config=self._config, grid=grid)

        self.Fired = self.fzeros(self._config.N)
        self.last_Fired = self.izeros(self._config.N) - self._config.D
        self.Firing_times = self.fzeros(shapes.Firing_times)
        self.Firing_idcs = self.izeros(shapes.Firing_idcs)
        self.Firing_counts = self.izeros(shapes.Firing_counts)

        self.G_firing_count_hist = self.izeros((self._plotting_config.scatter_plot_length, self._config.G))

        self.G_stdp_config0 = self.izeros((self._config.G, self._config.G))
        self.G_stdp_config1 = self.izeros((self._config.G, self._config.G))

        self.Simulation = self._init_sim(T, plotting_config)

        self.N_relative_G_indices = self._N_relative_G_indices()
        self.G_swap_tensor = self._G_swap_tensor()
        self.print_allocated_memory('G_swap_tensor')
        self.swap_group_synapses(torch.from_numpy(grid.forward_groups).to(device=self.device).type(torch.int64))

        self.actualize_N_rep_pre_synaptic_idx(shapes)

        self.N_states.use_preset('rs', self.selected_neuron_mask(self._config.sensory_groups))

        self.Simulation.set_stdp_config(0)

        self.N_rep_buffer = self.N_rep_buffer.reshape(shapes.N_rep)
        # self.N_rep_buffer[:] = self.N_rep_groups_cpu.to(self.device)

        self.Simulation.calculate_avg_group_weight()

        self.output_tensor = self.fzeros(6)

        self.debug = False

    def _init_sim(self, T, plotting_config):

        sim = snn_simulation_gpu.SnnSimulation(
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
            G_group_delay_counts=self.G_group_delay_counts.data_ptr(),
            G_flags=self.G_flags.data_ptr(),
            G_props=self.G_props.data_ptr(),
            N_rep=self.N_rep.data_ptr(),
            N_rep_buffer=self.N_rep_buffer.data_ptr(),
            N_rep_pre_synaptic_idx=self.N_rep_pre_synaptic_idx.data_ptr(),
            N_rep_pre_synaptic_counts=self.N_rep_pre_synaptic_counts.data_ptr(),
            N_delays=self.N_delays.data_ptr(),
            N_states=self.N_states.data_ptr(),
            N_weights=self.N_weights.data_ptr(),
            fired=self.Fired.data_ptr(),
            last_fired=self.last_Fired.data_ptr(),
            firing_times=self.Firing_times.data_ptr(),
            firing_idcs=self.Firing_idcs.data_ptr(),
            firing_counts=self.Firing_counts.data_ptr(),
            G_firing_count_hist=self.G_firing_count_hist.data_ptr(),
            G_stdp_config0=self.g2g_info_arrays.G_stdp_config0.data_ptr(),
            G_stdp_config1=self.g2g_info_arrays.G_stdp_config1.data_ptr(),
            G_avg_weight_inh=self.g2g_info_arrays.G_avg_weight_inh.data_ptr(),
            G_avg_weight_exc=self.g2g_info_arrays.G_avg_weight_exc.data_ptr(),
            G_syn_count_inh=self.g2g_info_arrays.G_syn_count_inh.data_ptr(),
            G_syn_count_exc=self.g2g_info_arrays.G_syn_count_exc.data_ptr()
        )

        return sim

    # noinspection PyUnusedLocal
    def actualize_N_rep_pre_synaptic_idx(self, shapes):

        self.N_rep_buffer = self.N_rep_buffer.reshape(shapes.N_rep_inv)

        self.Simulation.actualize_N_rep_pre_synaptic()

        if self._config.N <= 2 * 10 ** 5:
            aa = self.to_dataframe(self.N_rep_buffer)
            ab = self.to_dataframe(self.N_rep_pre_synaptic_counts)
            ac = self.to_dataframe(self.N_rep_pre_synaptic_idx)
            ad = self.to_dataframe(self.N_rep)

            # noinspection PyTypeChecker
            assert len(self.N_rep_pre_synaptic_idx[self.N_rep.flatten()[
                self.N_rep_pre_synaptic_idx.type(torch.int64)] != self.N_rep_buffer]) == 0

        self.N_rep_buffer[:] = -1

    # noinspection PyUnusedLocal
    def look_up(self, tuples, input_tensor, output_tensor=None, precision=6):
        if output_tensor is None:
            if len(self.output_tensor) != len(tuples):
                self.output_tensor = self.fzeros(len(tuples))
            output_tensor = self.output_tensor
        output_tensor[:] = torch.nan
        for i, e in enumerate(tuples):
            output_tensor[i] = input_tensor[e]
        print(output_tensor)

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

    def _N_G_and_G_neuron_counts_1of2(self, shapes: NetworkArrayShapes, grid: NetworkGrid, neurons: Neurons):
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

    def _G_delay_distance(self, G_pos: RegisteredVBO):
        # return None, None
        G_pos_distance = torch.cdist(G_pos.tensor, G_pos.tensor)
        return G_pos_distance, ((self._config.D - 1) * G_pos_distance / G_pos_distance.max()).round().int()

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
        N_pos = RegisteredVBO(vbo, shape, self.device)
        for g in self.type_groups:
            if g.ntype == NeuronTypes.INHIBITORY:
                orange = torch.Tensor([1, .5, .2])
                N_pos.tensor[g.start_idx:g.end_idx + 1, 7:10] = orange  # Inhibitory Neurons -> Orange
        self.registered_buffers.append(N_pos)
        return N_pos

    # def _G_pos(self, shape, vbo) -> RegisteredVBO:
    #     # groups = torch.arange(self._config.G, device=self.device)
    #     # z = (groups / (self._config.G_shape[0] * self._config.G_shape[1])).floor()
    #     # r = groups - z * (self._config.G_shape[0] * self._config.G_shape[1])
    #     # y = (r / self._config.G_shape[0]).floor()
    #     # x = r - y * self._config.G_shape[0]
    #     #
    #     # gpos = torch.zeros(shape, dtype=torch.float32, device=self.device)
    #     #
    #     # gpos[:, 0] = x * (self._config.N_pos_shape[0] / self._config.G_shape[0])
    #     # gpos[:, 1] = y * (self._config.N_pos_shape[1] / self._config.G_shape[1])
    #     # gpos[:, 2] = z * (self._config.N_pos_shape[2] / self._config.G_shape[2])
    #
    #     G_pos = RegisteredVBO.from_buffer(
    #         vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * 4, 4),
    #                                    dtype=np.float32, device=self.device))
    #
    #     # self.validate_N_G()
    #     return G_pos

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

    def _N_rep_and_N_delays(self, shapes: NetworkArrayShapes, curand_states):

        N, S, D, G = self._config.N, self._config.S, self._config.D, self._config.G

        torch.cuda.empty_cache()
        self.print_allocated_memory('syn_counts')

        N_delays = self.izeros(shapes.N_delays)
        N_rep_t = self.izeros(shapes.N_rep_inv)
        torch.cuda.empty_cache()
        self.print_allocated_memory('N_rep')
        self.N_rep_buffer[:] = 0

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
                sort_keys=self.N_rep_buffer.data_ptr(),
                N_rep=N_rep_t.data_ptr(),
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

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=self.N_rep_buffer.data_ptr(), N_rep=N_rep_t.data_ptr())

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
                sort_keys=self.N_rep_buffer.data_ptr(),
                N_rep=N_rep_t.data_ptr(),
                verbose=False)

        snn_construction_gpu.sort_N_rep(N=N, S=S, sort_keys=self.N_rep_buffer.data_ptr(), N_rep=N_rep_t.data_ptr())

        # N_rep = torch.empty(shapes.N_rep, dtype=torch.int32, device=self.device)
        self.N_rep_buffer = self.N_rep_buffer.reshape(shapes.N_rep)
        self.N_rep_buffer[:] = -1
        self.N_rep_buffer[:] = N_rep_t.T

        N_rep_t = N_rep_t.reshape(shapes.N_rep)
        N_rep_t[:] = self.N_rep_buffer
        self.N_rep_buffer[:] = -1
        self.print_allocated_memory(f'transposed')

        assert len(N_rep_t[N_rep_t == -1]) == 0

        return N_rep_t, N_delays

    def _N_rep_groups_cpu(self):
        N_rep_groups = self.N_rep.clone()
        # for ntype in NeuronTypes:
        #     for g in range(self._config.G):
        #         col = 2 * (ntype - 1)
        #         N_rep_groups[((self.N_rep >= self.group_indices[g, col])
        #                      & (self.N_rep <= self.group_indices[g, col + 1]))] = g

        snn_construction_gpu.fill_N_rep_groups(
            N=self._config.N,
            S=self._config.S,
            N_G=self.N_G.data_ptr(),
            N_rep=self.N_rep.data_ptr(),
            N_rep_groups=N_rep_groups.data_ptr(),
            N_G_n_cols=self._config.N_G_n_cols,
            N_G_group_id_col=self._config.N_G_group_id_col
        )
        self.print_allocated_memory('N_rep_groups')
        return N_rep_groups.cpu()

    def _N_weights(self, shape):
        weights = self.fzeros(shape)
        for gc in self.type_group_conns:
            weights[gc.location[1]: gc.location[1] + gc.conn_shape[1],
                    gc.location[0]: gc.location[0] + gc.conn_shape[0]] = gc.w0
        return weights

    def set_weight(self, w, x0=0, x1=None, y0=0, y1=None):
        self.N_weights[x0: x1 if x1 is not None else self._config.N,
                       y0: y1 if y1 is not None else self._config.S] = w

    def set_src_group_weights(self, groups, w):
        selected = self.selected_neuron_mask(groups)
        self.N_weights[:, selected] = w

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

    def select_groups(self, mask):
        return self.group_ids[mask]

    @staticmethod
    def actualize_group_separator_lines(plot_slots_tensor, pos_tensor, color_tensor, separator_mask, n_plots):

        separator_mask_ = separator_mask[: min(n_plots + 1, plot_slots_tensor[-1] + 1)].clone()
        separator_mask_[-1] = True
        separators = (plot_slots_tensor[: len(separator_mask_)][separator_mask_]
                      .repeat_interleave(2).to(torch.float32))

        separators = separators[: min(len(separators), pos_tensor.shape[0])]
        pos_tensor[:len(separators), 1] = separators
        color_tensor[:, 3] = 0
        color_tensor[:len(separators), 3] = 1

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

        neuron_groups = self.izeros(max(n_voltage_plots, n_scatter_plots) + 1)
        neuron_groups[: -1] = self.N_G[:, 1][selected_neurons[: max(n_voltage_plots, n_scatter_plots)]]
        neuron_groups_prev = self.izeros(neuron_groups.shape)
        neuron_groups_prev[0] = -1
        neuron_groups_prev[1:] = neuron_groups[0: -1]
        neuron_groups[-1] = -1

        separator_mask = neuron_groups != neuron_groups_prev

        self.actualize_group_separator_lines(
            plot_slots_tensor=self.plotting_arrays.voltage_plot_slots,
            separator_mask=separator_mask,
            pos_tensor=self.plotting_arrays.voltage_group_line_pos.tensor,
            color_tensor=self.plotting_arrays.voltage_group_line_colors.tensor,
            n_plots=n_voltage_plots)

        self.actualize_group_separator_lines(
            plot_slots_tensor=self.plotting_arrays.firings_plot_slots,
            separator_mask=separator_mask,
            pos_tensor=self.plotting_arrays.firings_group_line_pos.tensor,
            color_tensor=self.plotting_arrays.firings_group_line_colors.tensor,
            n_plots=n_scatter_plots)

    def redirect_synapses(self, groups, direction, rate):
        pass

    @property
    def group_counts(self):
        return self.G_neuron_counts[:len(NeuronTypes)].sum(axis=0)

    def _N_relative_G_indices(self):
        all_groups = self.N_G[:, 1].type(torch.int64)
        inh_start_indices = self.G_neuron_typed_ccount[all_groups]
        start_indices = self.G_neuron_typed_ccount[all_groups + self._config.G]
        inh_neurons = self.N_G[:, 0] == 1
        start_indices[inh_neurons] = inh_start_indices[inh_neurons]
        start_indices[~inh_neurons] -= (self.G_neuron_typed_ccount[all_groups + 1][~inh_neurons]
                                        - inh_start_indices[~inh_neurons])
        return (self.neuron_ids - start_indices).type(torch.int32)

    def _G_swap_tensor(self):
        max_neurons_per_group = self.group_counts.max().item()
        m = self._config.swap_tensor_shape_multiplicators
        return self.izeros((m[0], m[1] * max_neurons_per_group)) - 1

    # noinspection PyUnusedLocal
    def swap_group_synapses(self, groups, self_swap=True):

        # groups = groups[:3:, [0]]
        # groups = groups.flip(0)

        n_groups = groups.shape[1]
        if n_groups > self._config.swap_tensor_shape_multiplicators[1]:
            raise ValueError

        swap_delay = self.g2g_info_arrays.G_delay_distance[groups.ravel()[0], groups.ravel()[n_groups]].item()
        # noinspection PyUnresolvedReferences
        if not bool((self.g2g_info_arrays.G_delay_distance[
                         groups.ravel()[:groups.size().numel() - n_groups],
                         groups.ravel()[n_groups:groups.size().numel()]] == swap_delay).all()):
            raise ValueError

        chain_length = groups.shape[0] - 2

        group_neuron_counts_typed = (self.G_neuron_counts[:len(NeuronTypes)][:, groups.ravel()]
                                     .reshape((2, groups.shape[0], groups.shape[1])))

        group_neuron_counts_total = group_neuron_counts_typed[0] + group_neuron_counts_typed[1]

        inh_sums = group_neuron_counts_typed[0].sum(axis=1)
        total_sums = group_neuron_counts_total.sum(axis=1)

        swap_rates0 = self.fzeros((chain_length, n_groups)) + 1.
        swap_rates1 = self.fzeros((chain_length, n_groups)) + 1.
        # group_indices_offset = 0
        # max_neurons_per_group = int(self.G_swap_tensor.shape[0] / 2)
        neuron_group_indices = self.izeros(self.G_swap_tensor.shape[1]) - 1
        neuron_group_counts = self.izeros((2, self.G_swap_tensor.shape[1]))
        neuron_group_indices_aranged = torch.arange(n_groups, device=self.device)

        for i in range(chain_length):
            program = groups[i:i + 3]

            n_inh_core_neurons = inh_sums[i + 1]
            neuron_group_indices[:n_inh_core_neurons] = (
                torch.repeat_interleave(neuron_group_indices_aranged, group_neuron_counts_typed[0][i + 1].ravel()))
            neuron_group_indices[n_inh_core_neurons:total_sums[i+1]] = (
                torch.repeat_interleave(neuron_group_indices_aranged, group_neuron_counts_typed[1][i + 1].ravel()))
            self._single_group_swap(
                program,
                group_neuron_counts_total[i:i + 3], group_neuron_counts_typed[:, i:i + 3],
                neuron_group_counts, neuron_group_indices,
                swap_rates0[i])

            self._single_group_swap(
                program[[1, 1, 2], :].clone(),
                group_neuron_counts_total[[i+1, i+1, i+2], :].clone(),
                group_neuron_counts_typed[:, [i+1, i+1, i+2]].clone(),
                neuron_group_counts, neuron_group_indices,
                swap_rates1[i])

            neuron_group_indices[:] = -1

        assert len(self.N_rep[self.N_rep < 0]) == 0
        return

    def _single_group_swap(self, program,
                           group_neuron_counts_total, group_neuron_counts_typed,
                           neuron_group_counts, neuron_group_indices,
                           swap_rates):

        # swap_delay0 = self.G_delay_distance[program[0, 0], program[1, 0]].item()

        neurons = self.select(groups=program[1])
        n_neurons = len(neurons)

        assert (neuron_group_indices[neuron_group_indices >= 0] + program[1, 0] - self.N_G[neurons][:, 1]).sum() == 0

        print_idx = min(1900000, n_neurons - 1)
        # print_idx = 0
        self.Simulation.swap_groups(
            neurons.data_ptr(), n_neurons,
            program.data_ptr(), program.shape[1],
            neuron_group_indices.data_ptr(),
            self.G_swap_tensor.data_ptr(), self.G_swap_tensor.shape[1],
            swap_rates.data_ptr(), swap_rates.data_ptr(),
            group_neuron_counts_typed[0].data_ptr(), group_neuron_counts_typed[1].data_ptr(),
            group_neuron_counts_total.data_ptr(),
            self.g2g_info_arrays.G_delay_distance.data_ptr(),
            self.N_relative_G_indices.data_ptr(), self.G_neuron_typed_ccount.data_ptr(), neuron_group_counts.data_ptr(),
            print_idx)
        # noinspection PyUnusedLocal
        a, b = self.swap_validation(print_idx, neurons)
        # assert (neuron_group_counts[0].any() == False)
        self.N_rep_groups_cpu[:, neurons] = self.G_swap_tensor[:, :n_neurons].cpu()

        self.G_swap_tensor[:] = -1

        neuron_group_counts[:] = 0

    def swap_validation(self, j, neurons):
        a = self.to_dataframe(self.G_swap_tensor)
        b = a.iloc[:, j:j + 3].copy()
        b.columns = [1, 2, 3]
        b[2] = self.N_rep[:, neurons[j]].cpu().numpy()
        b[3] = self.N_G[b[2], 1].cpu().numpy()
        b[0] = self.N_rep_groups_cpu[:, neurons[j]].numpy()
        b[4] = b[0] != b[3]

        return a, b

    @property
    def active_sensory_groups(self):
        return self.select_groups(self.G_flags.b_sensory_input.type(torch.bool))

    @property
    def active_output_groups(self):
        return self.select_groups(self.G_flags.b_output_group.type(torch.bool))

    def unregister_registered_buffers(self):
        for rb in self.registered_buffers:
            rb.reg.unregister(None)

    def update(self):

        if self.Simulation.t % 100 == 0:
            print('t =', self.Simulation.t)
        if self.Simulation.t % 1000 == 0:
            # if False:
            self.Simulation.calculate_avg_group_weight()
            a = self.to_dataframe(self.g2g_info_arrays.G_avg_weight_inh)
            b = self.to_dataframe(self.g2g_info_arrays.G_avg_weight_exc)
            r = 6
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)],
            # self.G_stdp_config0.type(torch.float32))
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_inh)
            # self.look_up([(80, 72), (88, 80), (96, 88), (104, 96), (112, 104)], self.G_avg_weight_exc)
            # print()
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.g2g_info_arrays.G_avg_weight_inh)
            self.look_up([(75, 67), (83, 75), (91, 83), (99, 91), (107, 99), (115, 107), (123, 115)],
                         self.g2g_info_arrays.G_avg_weight_exc)
            print()

        n_updates = self._config.sim_updates_per_frame

        t_mod = self.Simulation.t % self._plotting_config.scatter_plot_length

        if self.debug is False:

            for i in range(n_updates):
                if self.debug is False:
                    self.Simulation.update(False)
                    # if self.Simulation.t >= 2100:
                    #     self.debug = True
        else:
            a = self.to_dataframe(self.Firing_idcs)
            b = self.to_dataframe(self.Firing_times)
            c = self.to_dataframe(self.Firing_counts)
            self.Simulation.update(True)

        # print(self.G_firing_count_hist.flatten()[67 + (self.Simulation.t-1) * self._config.G])

        self.plotting_arrays.group_firing_counts_plot_single1.tensor[
        t_mod: t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 123] / 100

        offset1 = self._plotting_config.scatter_plot_length

        self.plotting_arrays.group_firing_counts_plot_single1.tensor[
        offset1 + t_mod: offset1 + t_mod + n_updates, 1] = \
            self.G_firing_count_hist[t_mod: t_mod + n_updates, 125] / 100

        if t_mod + n_updates + 1 >= self._plotting_config.scatter_plot_length:
            self.G_firing_count_hist[:] = 0
