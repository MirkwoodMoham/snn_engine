import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
from vispy.scene import visuals

from .network_config import (
    NetworkConfig,
    PlottingConfig, BufferCollection
)
from .network_gpu_arrays import NetworkGPUArrays
from .network_structures import (
    NeuronTypes,
    NeuronTypeGroup,
    NeuronTypeGroupConnection
)
from .boxes import (
    BoxSystem,
    InputCells,
    OutputCells,
    SelectorBox,
)
from .plots import (
    VoltagePlot,
    FiringScatterPlot
)
from .neurons import Neurons
from .network_states import IzhikevichModel
from .network_grid import NetworkGrid
from rendering import Box


# noinspection PyPep8Naming
class NetworkCPUArrays:

    def __init__(self, config: NetworkConfig, gpu_arrays: NetworkGPUArrays):

        self._config = config
        self.gpu = gpu_arrays

        self.N_rep: np.array = gpu_arrays.N_rep.cpu()
        self.N_G: np.array = gpu_arrays.N_G.cpu()

        self.group_indices: np.array = gpu_arrays.group_indices.cpu()

        self.N_rep_groups: np.array = self.gpu.N_rep_groups_cpu

    @staticmethod
    def to_dataframe(tensor: torch.Tensor):
        return pd.DataFrame(tensor.numpy())


class SpikingNeuronNetwork:
    # noinspection PyPep8Naming
    def __init__(self,
                 network_config: NetworkConfig,
                 plotting_config: PlottingConfig,
                 max_batch_size_mb: int,
                 T: int = 2000,
                 model=IzhikevichModel,
                 ):

        # RenderedObjectNode._grid_unit_shape = network_config.grid_unit_shape

        self.T = T
        self.network_config: NetworkConfig = network_config
        self._plotting_config: PlottingConfig = plotting_config
        self.grid = NetworkGrid(self.network_config)
        print('\n', self.network_config, '\n')
        self.model = model
        self.max_batch_size_mb = max_batch_size_mb

        self.type_group_dct: Dict[int, NeuronTypeGroup] = {}
        self.type_group_conn_dict: Dict[tuple[int, int], NeuronTypeGroupConnection] = {}
        self.next_group_id = 0

        g_inh = self.add_type_group(count=int(.2 * self.network_config.N), neuron_type=NeuronTypes.INHIBITORY)
        g_exc = self.add_type_group(count=self.network_config.N - len(g_inh), neuron_type=NeuronTypes.EXCITATORY)

        self.add_type_group_conn(g_inh, g_exc, w0=-.49, exp_syn_counts=self.network_config.S)
        c_exc_inh = self.add_type_group_conn(
            g_exc, g_inh, w0=.51,
            exp_syn_counts=max(int((len(g_inh) / self.network_config.N) * self.network_config.S), 1))
        self.add_type_group_conn(g_exc, g_exc, w0=.5, exp_syn_counts=self.network_config.S - len(c_exc_inh))

        self._neurons = Neurons(self.network_config, self.grid.segmentation, self.type_groups)

        self.GPU: Optional[NetworkGPUArrays] = None
        self.CPU: Optional[NetworkCPUArrays] = None

        self.rendered_3d_objs = [self._neurons]

        self.outer_grid: Optional[visuals.Box] = None
        self.selector_box: Optional[SelectorBox] = None
        self.voltage_plot: Optional[VoltagePlot] = None
        self.firing_scatter_plot: Optional[VoltagePlot] = None
        self.selected_group_boxes: Optional[BoxSystem] = None
        self.input_cells: Optional[InputCells] = None
        self.output_cells: Optional[OutputCells] = None

        self._all_rendered_objects_initialized = False

        self.validate()

    @property
    def plotting_config(self):
        return self._plotting_config

    def validate(self):
        NeuronTypeGroup.validate(self.type_group_dct, N=self.network_config.N)
        NeuronTypeGroupConnection.validate(self.type_group_conn_dict, S=self.network_config.S)

    @property
    def type_groups(self):
        return self.type_group_dct.values()

    # noinspection PyPep8Naming
    def add_type_group(self, count, neuron_type):
        g = NeuronTypeGroup.from_count(self.next_group_id, count, self.network_config.S,
                                       neuron_type, self.type_group_dct)
        self.next_group_id += 1
        return g

    def add_type_group_conn(self, src, snk, w0, exp_syn_counts):
        c = NeuronTypeGroupConnection(src, snk, w0=w0, S=self.network_config.S,
                                      exp_syn_counts=exp_syn_counts,
                                      max_batch_size_mb=self.max_batch_size_mb,
                                      conn_dict=self.type_group_conn_dict)
        return c

    def update(self):
        self.GPU.update()

    # noinspection PyStatementEffect,PyTypeChecker
    def initialize_rendered_objs(self):
        self.voltage_plot = VoltagePlot(n_plots=self.plotting_config.n_voltage_plots,
                                        plot_length=self.plotting_config.voltage_plot_length)
        self.firing_scatter_plot = FiringScatterPlot(n_plots=self.plotting_config.n_scatter_plots,
                                                     plot_length=self.plotting_config.scatter_plot_length)
        self.outer_grid: visuals.Box = Box(shape=self.network_config.N_pos_shape,
                                           scale=[.99, .99, .99],
                                           segments=self.network_config.grid_segmentation,
                                           depth_test=True,
                                           use_parent_transform=False)
        self.outer_grid.visible = False
        self.outer_grid.set_gl_state(polygon_offset_fill=True, cull_face=False,
                                     polygon_offset=(1, 1), depth_test=False, blend=True)

        self.selector_box = SelectorBox(self.network_config, self.grid)
        g = self.network_config.G
        self.selected_group_boxes = BoxSystem(network_config=self.network_config,
                                              grid=self.grid,
                                              connect=np.zeros((g + 1, 2)) + g)

        self.input_cells = InputCells(
            data=np.array([0., 1., 0.]),
            pos=np.array([[int(self.network_config.N_pos_shape[0]/2 + 1) * self.grid.unit_shape[1],
                           0.,
                           self.network_config.N_pos_shape[2] - self.grid.unit_shape[2]]]),
            network=self,
            state_colors_attr='input_face_colors',
            compatible_groups=self.network_config.sensory_groups,
        )
        self.output_cells = OutputCells(
            data=np.array([0., -1., 1.]),
            pos=np.array([[int(self._neurons._shape[0]/2 + 1) * self.grid.unit_shape[1],
                           self._neurons._shape[1] - self.grid.unit_shape[1],
                           self._neurons._shape[2] - self.grid.unit_shape[2]]]),
            state_colors_attr='output_face_colors',
            network=self,
            data_color_coding=np.array([
                [1., 0., 0., .6],
                [0., 1., 0., .6],
                # [0., 0., 0., .0],
            ]),
            compatible_groups=self.network_config.output_groups,
            face_dir='+z',
        )

        self.rendered_3d_objs.append(self.outer_grid)
        self.rendered_3d_objs.append(self.selector_box)
        self.rendered_3d_objs.append(self.selected_group_boxes)
        self.rendered_3d_objs.append(self.output_cells)
        self.rendered_3d_objs.append(self.input_cells)

        self._all_rendered_objects_initialized = True

    def add_rendered_objects(self, view_3d, vplot_view, fplot_view):
        if not self._all_rendered_objects_initialized:
            self.initialize_rendered_objs()
        vplot_view.add(self.voltage_plot)
        fplot_view.add(self.firing_scatter_plot)
        for o in self.rendered_3d_objs:
            view_3d.add(o)

    # noinspection PyPep8Naming
    def initialize_GPU_arrays(self, device):
        if not self._all_rendered_objects_initialized:
            raise AssertionError('not self._all_rendered_objects_initialized')

        buffers = BufferCollection(
            N_pos=self._neurons.vbo,
            voltage=self.voltage_plot.vbo,
            firings=self.firing_scatter_plot.vbo,
            selected_group_boxes_vbo=self.selected_group_boxes.vbo,
            selected_group_boxes_ibo=self.selected_group_boxes.ibo,
        )
        self.GPU = NetworkGPUArrays(
            config=self.network_config,
            grid=self.grid,
            neurons=self._neurons,
            type_group_dct=self.type_group_dct,
            type_group_conn_dct=self.type_group_conn_dict,
            device=device,
            T=self.T,
            plotting_config=self.plotting_config,
            model=self.model,
            buffers=buffers)

        self.selector_box.init_cuda_attributes(self.GPU.device, self.GPU.G_props)
        self.selected_group_boxes.init_cuda_attributes(self.GPU.device, self.GPU.G_props)
        self.output_cells.init_cuda_attributes(self.GPU.device, self.GPU.G_props)
        self.input_cells.init_cuda_attributes(self.GPU.device, self.GPU.G_props)

        self.input_cells.src_weight = self.network_config.InitValues.Weights.SensorySource

        self.CPU = NetworkCPUArrays(self.network_config, self.GPU)

