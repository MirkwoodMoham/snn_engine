import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from typing import Optional

from vispy.geometry import create_box, create_plane
from vispy.visuals import MeshVisual
from vispy.visuals.transforms import STTransform

from network.network_config import NetworkConfig
# from network.spiking_neural_network import SpikingNeuronNetwork
from network.network_states import LocationGroupProperties
from rendering import (
    Translate,
    BoxSystemLineVisual,
    RenderedObjectNode,
    Scale,
    CudaBox,
    RenderedCudaObjectNode,
    NormalArrow,
    initial_normal_vertices
)
from gpu import RegisteredGPUArray


# noinspection PyAbstractClass
class SelectorBox(RenderedCudaObjectNode):
    count: int = 0

    def __init__(self, network_config: NetworkConfig, parent=None, name=None):
        self.name = name or f'{self.__class__.__name__}{SelectorBox.count}'

        self._select_children: list[NormalArrow] = []
        # super().__init__(name=name or f'{self.__class__.__name__}{SelectorBox.count}',
        #                  parent=parent, selectable=True)
        self.network_config = network_config
        self.original_color = (1, 0.65, 0, 0.5)
        self._visual: CudaBox = CudaBox(select_parent=self,
                                        name=self.name + '.obj',
                                        shape=self.shape,
                                        # color=np.array([1, 0.65, 0, 0.5]),
                                        color=(1, 0.65, 0, 0.1),
                                        edge_color=self.original_color,
                                        # scale=[1.1, 1.1, 1.1],
                                        depth_test=False,
                                        border_width=2,
                                        parent=None)

        super().__init__([self._visual], selectable=True, parent=parent)

        self.unfreeze()

        self.transform = STTransform()
        self.transform.translate = (self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2)
        self.transform.scale = [1.1, 1.1, 1.1]

        for normal in self._select_children:
            normal.transform = self.transform

        SelectorBox.count += 1
        self.interactive = True
        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / min(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)
        self.states_gpu: Optional[LocationGroupProperties] = None

        # noinspection PyPep8Naming
        G = self.network_config.G

        self.selected_masks = np.zeros((G, 4), dtype=np.int32, )

        self.group_numbers = np.arange(G)  # .reshape((G, 1))

        self.selection_property = 'thalamic_input'

        self.freeze()

    @property
    def g_pos(self):
        return self.network_config.g_pos[:self.network_config.G, :]

    @property
    def g_pos_end(self):
        return self.network_config.g_pos_end[:self.network_config.G, :]

    @property
    def shape(self):
        return self.network_config.grid_unit_shape

    @property
    def color(self):
        return self.visual._border.color

    @color.setter
    def color(self, v):
        self.visual._border.color = v

    @property
    def vbo_glir_id(self):
        return self.visual._border._vertices.id

    @property
    def selection_vertices(self):
        return (self.visual._initial_selection_vertices
                * self.transform.scale[:3]
                + self.transform.translate[:3])

    @property
    def edge_lengths(self):
        return np.array(self.shape) * self.transform.scale[:3]

    def transform_changed(self):
        g_pos = self.g_pos
        g_pos_end = self.g_pos_end
        v = self.selection_vertices
        self.selected_masks[:, 0] = (g_pos[:, 0] >= v[0, 0]) & (g_pos_end[:, 0] <= v[1, 0])
        self.selected_masks[:, 1] = (g_pos[:, 1] >= v[0, 1]) & (g_pos_end[:, 1] <= v[2, 1])
        self.selected_masks[:, 2] = (g_pos[:, 2] >= v[0, 2]) & (g_pos_end[:, 2] <= v[3, 2])
        # self.selected_masks[:, 3] = self.network_config.G
        # self.selected_masks[:, 3] = np.where(~self.selected_masks[:, :3].all(axis=1),
        #                                      self.selected_masks[:, 3], self.group_numbers)
        #
        # self.states_gpu.selected = torch.from_numpy(self.selected_masks[:, [3]]).to(self._cuda_device)
        self.states_gpu.selected = torch.from_numpy(self.selected_masks[:, :3].all(axis=1)).to(self._cuda_device)

    def on_select_callback(self, v: bool):
        self.states_gpu.selection_property = self.selection_property if v is True else None
        self.swap_select_color(v)
        for c in self.visual.normals:
            c.visual.visible = v

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, property_tensor):
        super().init_cuda_attributes(device)
        self.states_gpu = property_tensor
        self.transform_connected = True

        for c in self.visual.normals:
            c.visual.visible = False

    # @property
    # def color_vbo_glir_id(self):
    #     return self._obj._border.shared_program.vert['base_color'].id

    # def init_cuda_arrays(self):
    #     nbytes = 4
    #     shape = (24, 4)
    #     print('shape:', shape)
    #     b = RegisteredGPUArray.from_buffer(
    #         self.color_vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes, nbytes),
    #                                               dtype=np.float32, device=self.cuda_device))
    #     # return b
    #     self._gpu_array = b


# GSLine = visuals.create_visual_node(GSLineVisual)
# BoxSystemLine = visuals.create_visual_node(BoxSystemLineVisual)


# noinspection PyAbstractClass
class BoxSystem(RenderedCudaObjectNode):

    def __init__(self, network_config, pos, grid_unit_shape, max_z, color=(0.1, 1., 1., 1.), **kwargs):

        self.network_config = network_config
        self._visual = BoxSystemLineVisual(grid_unit_shape=grid_unit_shape, max_z=max_z, pos=pos, color=color, **kwargs)

        self._output_planes: MeshVisual = self.create_mesh('+z', self.network_config.output_grid_pos)
        self._sensory_input_planes: MeshVisual = self.create_mesh('-y', self.network_config.sensory_grid_pos)

        super().__init__([self.visual, self._sensory_input_planes, self._output_planes])
        # noinspection PyTypeChecker

        # noinspection PyTypeChecker
        self.set_gl_state(polygon_offset_fill=True,
                          polygon_offset=(1, 1), depth_test=False, blend=True)

        self.transform = STTransform(translate=(0, 0, 0), scale=(1, 1, 1))

        self.unfreeze()
        self.states_gpu: Optional[LocationGroupProperties] = None
        self._input_color_array: Optional[RegisteredGPUArray] = None
        self._output_color_array: Optional[RegisteredGPUArray] = None
        self.freeze()

    @property
    def vbo_glir_id(self):
        return self.visual._line_visual._pos_vbo.id

    @property
    def ibo_glir_id(self):
        return self.visual._line_visual._connect_ibo.id

    def create_planes(self, dir_, grid_pos, height=None, width=None, width_segments=None, height_segments=None):

        if 'y' in dir_:
            i = 1
        elif 'z' in dir_:
            i = 2
        else:
            i = 0
        i_planes = np.unique(grid_pos[:, i]) * self.network_config.grid_unit_shape[i]
        # i_planes = np.nunique(grid_pos[:, (i + 1) % 3])

        n_planes = len(i_planes)

        j = (i + 1) % 3
        j_segments = np.unique(grid_pos[:, j])
        n_j_segments = len(j_segments)
        j_pos_shape = self.network_config.N_pos_shape[j]
        k = (i + 2) % 3
        k_segments = np.unique(grid_pos[:, k])
        n_k_segments = len(k_segments)
        k_pos_shape = self.network_config.N_pos_shape[k]

        height = height or [n_j_segments * self.network_config.grid_unit_shape[j]] * n_planes
        width = width or [n_k_segments * self.network_config.grid_unit_shape[k]] * n_planes
        width_segments = width_segments or [n_j_segments] * n_planes
        height_segments = height_segments or [n_k_segments] * n_planes

        planes_m = []

        for idx, y in enumerate(i_planes):
            vertices_p, faces_p, outline_p = create_plane(height[idx], width[idx],
                                                          width_segments[idx], height_segments[idx], dir_)
            vertices_p['position'][:, k] += ((np.min(k_segments) * self.network_config.grid_unit_shape[k] + k_pos_shape)
                                             / 2)
            vertices_p['position'][:, j] += ((np.min(j_segments) * self.network_config.grid_unit_shape[j] + j_pos_shape)
                                             / 2)
            vertices_p['position'][:, i] = y + self.network_config.grid_unit_shape[i] * int('+' in dir_)
            planes_m.append((vertices_p, faces_p, outline_p))

        # noinspection DuplicatedCode
        positions = np.zeros((0, 3), dtype=np.float32)
        texcoords = np.zeros((0, 2), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)

        faces = np.zeros((0, 3), dtype=np.uint32)
        outline = np.zeros((0, 2), dtype=np.uint32)
        offset = 0
        for vertices_p, faces_p, outline_p in planes_m:
            positions = np.vstack((positions, vertices_p['position']))
            texcoords = np.vstack((texcoords, vertices_p['texcoord']))
            normals = np.vstack((normals, vertices_p['normal']))

            faces = np.vstack((faces, faces_p + offset))
            outline = np.vstack((outline, outline_p + offset))
            offset += vertices_p['position'].shape[0]

        vertices = np.zeros(positions.shape[0],
                            [('position', np.float32, 3),
                             ('texcoord', np.float32, 2),
                             ('normal', np.float32, 3),
                             ('color', np.float32, 4)])

        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(np.interp(colors,
                                                 (np.min(colors),
                                                  np.max(colors)),
                                                 (0, 1)),
                                       positions.shape),
                            np.ones((positions.shape[0], 1))))

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    def create_mesh(self, dir_, grid_pos) -> MeshVisual:
        vertices, faces, outline = self.create_planes(dir_, grid_pos)
        face_colors = np.repeat(np.array([[0., 0., 0., 1.]]), len(faces), axis=0)
        return MeshVisual(vertices['position'], faces, None, face_colors, None)

    def init_cuda_arrays(self):
        self._input_color_array = self.face_color_array(self._sensory_input_planes)
        self._input_color_array.tensor[:, 3] = .5
        self._output_color_array = self.face_color_array(self._output_planes, self.output_color_vbo)
        self._output_color_array.tensor[:, 3] = .5

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, property_tensor):
        self.states_gpu: LocationGroupProperties = property_tensor
        super().init_cuda_attributes(device)
        self.states_gpu.input_face_colors = self._input_color_array.tensor
        self.states_gpu.output_face_colors = self._output_color_array.tensor

    @property
    def color_vbo_glir_id(self):
        return self._sensory_input_planes.shared_program.vert['base_color'].id
    
    @property
    def output_color_vbo(self):
        return self.buffer_id(self._output_planes.shared_program.vert['base_color'].id)


# noinspection PyAbstractClass
class IOCells(RenderedCudaObjectNode):

    count: int = 0

    def __init__(self, pos,
                 data: np.array,
                 network,
                 compatible_groups: np.array,
                 data_color_coding=None,
                 face_dir='-y',
                 segmentation=(3, 1, 1),
                 unit_shape=None,
                 color=(1., 1., 1., 1.), name=None, **kwargs):

        if 'x' in face_dir:
            i = 0
        elif 'y' in face_dir:
            i = 1
        else:
            i = 2

        if segmentation[i] != 1:
            raise ValueError
        if len(pos) != 1:
            raise ValueError

        self.network_config = network.network_config
        self.network = network
        pos = np.vstack((pos, np.array([0., 0., self.network_config.max_z + 1])))
        init_pos = pos[0]
        self.pos = pos - init_pos
        self._shape = (segmentation[0] * self.network_config.grid_unit_shape[0],
                       segmentation[1] * self.network_config.grid_unit_shape[1],
                       segmentation[2] * self.network_config.grid_unit_shape[2]) if unit_shape is None else unit_shape
        self._segment_shape = (self._shape[0]/segmentation[0],
                               self._shape[1]/segmentation[1],
                               self._shape[2]/segmentation[2])
        self.compatible_groups = compatible_groups

        self.data_shape = np.zeros(segmentation).squeeze().shape
        if data.shape != self.data_shape:
            data = data.squeeze()
            if data.shape != self.data_shape:
                raise ValueError
        self.data_cpu = data.astype(np.float32)
        if data_color_coding is None:
            data_color_coding = np.array([
                [0., 0., 0., .4],
                [1., 1., 1., .4],
            ])
        if len(np.unique(data[data != -1.])) != len(data_color_coding):
            raise ValueError
        self.data_color_coding_cpu = data_color_coding.astype(np.float32)
        self.segmentation = segmentation
        self.n_cells = len(self.pos[: -1]) * self.segmentation[0] * self.segmentation[1] * self.segmentation[2]
        self.collision_shape = (self.n_cells, len(self.compatible_groups))

        center = np.array([self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2])

        self.pos[:-1] -= center

        self.name = name or f'{self.__class__.__name__}{InputCells.count}'

        filled_indices, vertices = self.create_input_cell_mesh(center, face_dir)

        self._mesh = MeshVisual(vertices['position'], filled_indices, None, self.face_colors_cpu, None)
        # noinspection PyTypeChecker
        self._visual = BoxSystemLineVisual(grid_unit_shape=self._segment_shape,
                                           max_z=self.network_config.max_z, pos=self.pos, color=color, **kwargs)

        super().__init__([self._visual, self._mesh], selectable=True)
        # noinspection PyTypeChecker
        self.set_gl_state(polygon_offset_fill=True, cull_face=False,
                          polygon_offset=(1, 1), depth_test=False, blend=True)

        self.interactive = True
        self.transform = STTransform(translate=center, scale=(1, 1, 1))
        move = np.zeros(3)

        move[i] = .1
        if '-' in face_dir:
            move *= -1
        # self.transform.move(np.array([0., -.1, 0.]))
        self.transform.move(init_pos)
        self.transform.move(move)
        self.unfreeze()
        j = (i + 1) % 3
        j_segments = np.unique(self.network_config.G_grid_pos[self.compatible_groups][:, j])
        self.n_j_segments = len(j_segments)
        j_pos_shape = self.network_config.N_pos_shape[j]
        k = (i + 2) % 3
        k_segments = np.unique(self.network_config.G_grid_pos[self.compatible_groups][:, k])
        self.n_k_segments = len(k_segments)
        k_pos_shape = self.network_config.N_pos_shape[k]

        self.normals = []
        inv = initial_normal_vertices(self.shape)
        for i in range(6):
            arrow = NormalArrow(self, points=inv[i], mod_factor=1 / (self.shape[int(i / 2)]))
            self.normals.append(arrow)

        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / max(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)

        self.states_gpu: Optional[LocationGroupProperties] = None

        self._collision_tensor_gpu: Optional[torch.Tensor] = None
        self._cell_pos_start_xx_gpu: Optional[torch.Tensor] = None
        self._cell_pos_end_xx_gpu: Optional[torch.Tensor] = None
        self._neuron_groups_pos_start_yy_gpu: Optional[torch.Tensor] = None
        self._neuron_groups_pos_end_yy_gpu: Optional[torch.Tensor] = None
        self._segment_shape_gpu: Optional[torch.Tensor] = None
        self._neuron_groups_shape_gpu: Optional[torch.Tensor] = None
        self.io_neuron_group_indices_gpu: Optional[torch.Tensor] = None
        self.io_neuron_group_values_gpu: Optional[torch.Tensor] = None
        self.data_gpu: Optional[torch.Tensor] = None
        self.data_color_coding_gpu: Optional[torch.Tensor] = None
        self.face_color_indices_gpu: Optional[torch.Tensor] = None
        self.scale_gpu: Optional[torch.Tensor] = None

        # self.g_pos = self.network_config.g_pos[:self.network_config.G, :]
        # self.g_pos_end = self.network_config.g_pos_end[:self.network_config.G, :]
        self.freeze()

        for normal in self.normals:
            normal.transform = self.transform

    @property
    def quad_colors_cpu(self):
        quad_colors = np.zeros((self.n_cells, 4))

        for i in range(len(self.data_color_coding_cpu)):
            quad_colors[self.data_cpu == i, :] = self.data_color_coding_cpu[i]

        return quad_colors

    @property
    def face_colors_cpu(self):
        quad_colors = self.quad_colors_cpu
        if self.quad_colors_cpu is None:
            face_colors = np.repeat(np.array([[1., 1., 1., 1.]]), self.n_cells, axis=0)
            face_colors[0] = np.array([1., 0., 0., 1.])
            face_colors[-1] = np.array([0., .75, 0., 1.])
        else:
            if len(quad_colors) == 1 and (self.segmentation != (1, 1, 1)):
                quad_colors = np.repeat(np.array([quad_colors]), self.n_cells, axis=0)
            face_colors = np.repeat(quad_colors, 2, axis=0)
        return face_colors

    @property
    def input_vertices(self):
        return (self.pos[:-1]
                * self.transform.scale[:3]
                + self.transform.translate[:3])

    def create_input_cell_mesh(self, center, dir_):
        init_pos = self.pos.copy()
        for i in range(3):
            if self.segmentation[i] > 1:
                add = np.repeat(np.array([np.linspace(0, self._shape[i], self.segmentation[i], endpoint=False)]),
                                len(self.pos), axis=0).flatten()
                self.pos = np.repeat(self.pos, self.segmentation[i], axis=0)
                self.pos[:, i] += add
                self.pos = self.pos[: - self.segmentation[i] + 1]
        vertices_list = []
        filled_indices_list = []
        for i in range(len(init_pos) - 1):
            vertices_, filled_indices_, _ = create_box(
                self._shape[0], self._shape[2], self._shape[1],
                self.segmentation[0], self.segmentation[2], self.segmentation[1],
                planes=(dir_,))
            vertices_['position'] += center + init_pos[i]
            vertices_list.append(vertices_)
            filled_indices_list.append(filled_indices_)
        vertices = rfn.stack_arrays(vertices_list, usemask=False)
        # filled_indices =
        add = 4 * np.repeat(np.array([np.repeat(np.arange(len(filled_indices_list)),
                                                2 * self.segmentation[0] * self.segmentation[2])]), 3, axis=0).T
        filled_indices = np.vstack(filled_indices_list) + add

        return filled_indices, vertices

    def on_select_callback(self, v: bool):
        # self.swap_select_color(v)
        self.states_gpu.selection_property = None
        for c in self.normals:
            c.visible = v
        self.transform_changed()

    def collision_volume(self):
        start = torch.maximum(self._cell_pos_start_xx_gpu, self._neuron_groups_pos_start_yy_gpu)
        end = torch.minimum(self._cell_pos_end_xx_gpu, self._neuron_groups_pos_end_yy_gpu)
        dist = end - start
        dist = (dist > 0).all(axis=2) * dist.abs_().prod(2)
        return dist.where(dist > 0, torch.tensor([-1], dtype=torch.float32, device=self._cuda_device))

    def get_sensory_input_indices(self):
        self._collision_tensor_gpu[:] = self.collision_volume()
        max_collision = self._collision_tensor_gpu.max(dim=0)

        return max_collision.indices.where(max_collision.values >= 0,
                                           torch.tensor([-1], dtype=torch.int64, device=self._cuda_device))

    def assign_sensory_input(self):
        self.scale_gpu[:] = torch.from_numpy(self.transform.scale).to(self._cuda_device)
        cell_pos = np.repeat(self.input_vertices.reshape((self.collision_shape[0], 1, 3)),
                             self.collision_shape[1], axis=1)
        self._cell_pos_start_xx_gpu[:] = torch.from_numpy(np.round(cell_pos, 6)).to(self._cuda_device)
        self._cell_pos_end_xx_gpu[:] = (self._cell_pos_start_xx_gpu
                                              + self._segment_shape_gpu * self.scale_gpu[:3])
        self.io_neuron_group_indices_gpu[:] = self.get_sensory_input_indices()
        valid_indices_mask = self.io_neuron_group_indices_gpu >= 0
        valid_input_data_indices = self.io_neuron_group_indices_gpu[valid_indices_mask].type(torch.int64)
        self.io_neuron_group_values_gpu[:] = -1
        self.io_neuron_group_values_gpu[valid_indices_mask] = self.data_gpu[valid_input_data_indices]

    def init_cuda_arrays(self):

        self._segment_shape_gpu = torch.from_numpy(np.array(self._segment_shape,
                                                            dtype=np.float32)).to(self._cuda_device)
        self._neuron_groups_shape_gpu = torch.from_numpy(np.array(self.network_config.grid_unit_shape,
                                                                  dtype=np.float32)).to(self._cuda_device)

        self._collision_tensor_gpu = torch.zeros(self.collision_shape, dtype=torch.float32, device=self._cuda_device)
        group_pos = np.repeat(self.network_config.g_pos[self.compatible_groups]
                              .reshape((1, self.collision_shape[1], 3)),
                              self.collision_shape[0], axis=0)

        self._neuron_groups_pos_start_yy_gpu = torch.from_numpy(group_pos).to(self._cuda_device)
        self._neuron_groups_pos_end_yy_gpu = self._neuron_groups_pos_start_yy_gpu + self._neuron_groups_shape_gpu
        shape = (self.collision_shape[0], self.collision_shape[1], 3)
        self._cell_pos_start_xx_gpu = torch.zeros(shape, dtype=torch.float32, device=self._cuda_device)
        self._cell_pos_end_xx_gpu = torch.zeros(shape, dtype=torch.float32, device=self._cuda_device)
        self.io_neuron_group_indices_gpu = torch.zeros(self.collision_shape[1],
                                                       dtype=torch.float32, device=self._cuda_device)
        self.io_neuron_group_values_gpu = torch.zeros(self.collision_shape[1],
                                                      dtype=torch.float32, device=self._cuda_device)
        self.data_gpu = torch.from_numpy(self.data_cpu).to(device=self._cuda_device)
        self.data_color_coding_gpu = torch.from_numpy(self.data_color_coding_cpu).to(self._cuda_device)

        n_sens_gr = len(self.compatible_groups)
        self.face_color_indices_gpu = torch.arange(n_sens_gr * 2 * 3,
                                                   device=self._cuda_device).reshape((self.n_k_segments,
                                                                                      self.n_j_segments,
                                                                                      2, 3))
        self.face_color_indices_gpu = self.face_color_indices_gpu.flip(0).transpose(0, 1).reshape(n_sens_gr, 2, 3)
        self.scale_gpu = torch.from_numpy(np.array(self.transform.scale)).to(self._cuda_device)

        self.transform_changed()

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, property_tensor):
        self.states_gpu: LocationGroupProperties = property_tensor
        super().init_cuda_attributes(device)
        self.transform_connected = True

        for n in self.normals:
            n.visible = False


# noinspection PyAbstractClass
class InputCells(IOCells):

    def __init__(self, pos,
                 data: np.array,
                 network,
                 compatible_groups: np.array,
                 data_color_coding=None,
                 face_dir='-y',
                 segmentation=(3, 1, 1),
                 unit_shape=None,
                 color=(1., 1., 1., 1.), name=None, **kwargs):

        super().__init__(pos=pos, data=data, network=network, compatible_groups=compatible_groups,
                         data_color_coding=data_color_coding, face_dir=face_dir, segmentation=segmentation,
                         unit_shape=unit_shape, color=color, name=name, **kwargs)
        self.unfreeze()
        self._src_weight = None
        self.freeze()

    def transform_changed(self):
        self.assign_sensory_input()
        self.states_gpu.sensory_input_type[:] = -1.
        self.states_gpu.sensory_input_type[self.network_config.sensory_groups] = self.io_neuron_group_values_gpu
        mask = self.states_gpu.sensory_input_type != -1.
        self.states_gpu.selected = mask
        self.states_gpu.b_thalamic_input = 0
        self.states_gpu.b_sensory_input = torch.where(mask, 1., 0.)
        self.states_gpu.input_face_colors[:, 3] = 0

        for i in range(len(self.data_color_coding_cpu)):

            indices = self.face_color_indices_gpu[self.io_neuron_group_values_gpu == i].flatten()

            self.states_gpu.input_face_colors[indices, :] = self.data_color_coding_gpu[i]
            # self.states_gpu.input_face_colors[indices, 3] = .5

        self.network.GPU.actualize_plot_map(self.network_config.sensory_groups[self.io_neuron_group_values_gpu.cpu() != -1.])

    @property
    def src_weight(self):
        return self._src_weight

    @src_weight.setter
    def src_weight(self, value):
        self._src_weight = value
        self.network.GPU.set_src_group_weights(self.network_config.sensory_groups, value)


# noinspection PyAbstractClass
class OutputCells(IOCells):

    def transform_changed(self):
        self.assign_sensory_input()
        self.states_gpu.output_type[:] = -1
        self.states_gpu.output_type[self.network_config.output_groups] = self.io_neuron_group_values_gpu
        mask = self.states_gpu.sensory_input_type != -1.
        self.states_gpu.selected = mask
        self.states_gpu.b_output_group = torch.where(mask, 1, 0)

        self.states_gpu.output_face_colors[:, 3] = 0

        for i in range(len(self.data_color_coding_cpu)):
            indices = self.face_color_indices_gpu[self.io_neuron_group_values_gpu == i].flatten()
            self.states_gpu.output_face_colors[indices, :] = self.data_color_coding_gpu[i]

        self.network.GPU.actualize_plot_map(self.network_config.output_groups[self.io_neuron_group_values_gpu.cpu() != -1.])
