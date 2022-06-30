from dataclasses import asdict
import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from typing import Optional

from vispy.geometry import create_box
from vispy.visuals import MeshVisual, TextVisual
from vispy.visuals.transforms import STTransform

from network.network_config import NetworkConfig
# from network.spiking_neural_network import SpikingNeuronNetwork
from network.network_states import PropertyTensor, LocationGroupProperties, LocationGroupFlags
from rendering import (
    Translate,
    BoxSystemLineVisual,
    InteractiveBoxNormals,
    Scale,
    CudaBox,
    RenderedCudaObjectNode,
    GridArrow
)
from gpu import RegisteredVBO
from .network_grid import NetworkGrid
from .group_mesh_visual import GroupMeshVisual


# noinspection PyAbstractClass
class SelectorBox(RenderedCudaObjectNode):
    count: int = 0

    def __init__(self, network_config: NetworkConfig, grid: NetworkGrid, parent=None, name=None):
        self.name = name or f'{self.__class__.__name__}{SelectorBox.count}'

        self._select_children: list[GridArrow] = []
        # super().__init__(name=name or f'{self.__class__.__name__}{SelectorBox.count}',
        #                  parent=parent, selectable=True)
        self.network_config = network_config
        self.grid = grid
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

        self._visual.normals.transform = self.transform

        SelectorBox.count += 1
        self.interactive = True
        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / min(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)
        self.G_flags: Optional[LocationGroupFlags] = None
        self.G_props: Optional[LocationGroupProperties] = None

        # noinspection PyPep8Naming
        G = self.network_config.G

        self.selected_masks = np.zeros((G, 4), dtype=np.int32, )

        self.group_numbers = np.arange(G)  # .reshape((G, 1))

        self.selection_flag = 'b_thalamic_input'

        self.freeze()
        
    @property
    def parent(self):
        return super().parent
        
    @parent.setter
    def parent(self, value):
        super(RenderedCudaObjectNode, self.__class__).parent.fset(self, value)
        for o in self.visual.normals:
            o.parent = value

    @property
    def g_pos(self):
        return self.grid.pos[:self.network_config.G, :]

    @property
    def g_pos_end(self):
        return self.grid.pos_end[:self.network_config.G, :]

    @property
    def shape(self):
        return self.grid.unit_shape

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
        self.G_flags.selected = torch.from_numpy(self.selected_masks[:, :3].all(axis=1)).to(self._cuda_device)

    def on_select_callback(self, v: bool):
        self.G_flags.selection_flag = self.selection_flag if v is True else None
        self.swap_select_color(v)
        self._visual.normals.visible = v

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, G_flags: LocationGroupFlags, G_props: LocationGroupProperties):
        super().init_cuda_attributes(device)
        self.G_flags = G_flags
        self.G_props = G_props
        self.transform_connected = True
        self._visual.normals.visible = False

        for normal in self._visual.normals:
            self.registered_buffers.append(normal.gpu_array)


# noinspection PyAbstractClass
class GroupBoxes(RenderedCudaObjectNode):

    def __init__(self, network_config: NetworkConfig, grid: NetworkGrid,
                 connect, color=(0.1, 1., 1., 1.), meshes=None):
        if meshes is None:
            meshes = []
        self._visual = BoxSystemLineVisual(grid_unit_shape=grid.unit_shape,
                                           max_z=network_config.max_z,
                                           connect=connect,
                                           pos=grid.pos, color=color)
        self.grid = grid
        super().__init__([self.visual] + meshes)

        # noinspection PyTypeChecker
        self.set_gl_state(polygon_offset_fill=True,
                          polygon_offset=(1, 1), depth_test=False, blend=True)

        self.transform = STTransform(translate=(0, 0, 0), scale=(1, 1, 1))

        self.unfreeze()
        self.G_flags: Optional[LocationGroupProperties] = None
        self.G_props: Optional[LocationGroupProperties] = None
        self.freeze()

    @property
    def vbo_glir_id(self):
        return self.visual._line_visual._pos_vbo.id

    @property
    def ibo_glir_id(self):
        return self.visual._line_visual._connect_ibo.id

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, G_flags: LocationGroupFlags, G_props: LocationGroupProperties):
        self.G_flags: LocationGroupFlags = G_flags
        self.G_props: LocationGroupProperties = G_props
        super().init_cuda_attributes(device)


# noinspection PyAbstractClass
class GroupInfo(GroupBoxes):

    def __init__(self, network_config: NetworkConfig, grid: NetworkGrid,
                 connect, color=(0.1, 1., 1., 1.)):

        self._mesh: GroupMeshVisual = GroupMeshVisual(
            network_config, grid, '+z', grid.grid_coord,
            face_colors=np.array([0., 0., 1., 1.]))

        text_pos = grid.pos_end

        text_pos[:, 0] -= grid.unit_shape[0] * .5
        text_pos[:, 1] -= grid.unit_shape[1] * .5
        text_pos[:, 2] += grid.unit_shape[2] * .35

        self.group_id_texts = {'None': None,
                               'group_ids': [str(i) for i in range(network_config.G)]}

        self.group_id_text_visual = TextVisual(text=self.group_id_texts['group_ids'],
                                               pos=text_pos, color='white', font_size=48)

        self.G_flags_texts = {'None': None}

        for k in asdict(LocationGroupFlags.Rows()).keys():
            self.G_flags_texts[k]: Optional[list[str]] = None

        text_pos[:, 2] -= grid.unit_shape[2] * .12
        # noinspection PyTypeChecker
        self.G_flags_text_visual = TextVisual(text=None,
                                              pos=text_pos, color='white', font_size=48)

        self.G_props_texts = {'None': None}

        for k in asdict(LocationGroupProperties.Rows()).keys():
            self.G_props_texts[k]: Optional[list[str]] = None

        text_pos[:, 2] -= grid.unit_shape[2] * .12
        # noinspection PyTypeChecker
        self.G_props_text_visual = TextVisual(text=None,
                                              pos=text_pos, color='white', font_size=48)

        super().__init__(network_config=network_config, grid=grid,
                         connect=connect, color=color,
                         meshes=[self._mesh,
                                 self.group_id_text_visual,
                                 self.G_flags_text_visual,
                                 self.G_props_text_visual])

        self.unfreeze()
        self.colors_gpu: Optional[RegisteredVBO] = None
        self.vertices_gpu: Optional[RegisteredVBO] = None
        self.face_group_ids: Optional[torch.Tensor] = None
        self.G_flags_cache: Optional[dict] = {}
        self.G_props_cache: Optional[dict] = {}
        self.group_ids: Optional[torch.Tensor] = None
        self.freeze()

    @staticmethod
    def _init_txt_visual(property_tensor: PropertyTensor, cache: dict, text_collection: dict, visual: TextVisual):
        b_visual_set: bool = False
        for k in asdict(property_tensor._rows).keys():
            tensor_row: torch.Tensor = getattr(property_tensor, k)
            cache[k] = tensor_row.clone()
            # noinspection PyTypeChecker
            text_collection[k] = [str(v) for v in list(tensor_row.cpu().numpy())]
            if b_visual_set is False:
                visual.text = text_collection[k]
                b_visual_set = True

    def init_cuda_arrays(self):
        self.colors_gpu: RegisteredVBO = self._mesh.face_color_array(self._cuda_device)
        self.colors_gpu.tensor[:, 3] = .7
        self.vertices_gpu: RegisteredVBO = self._mesh.vbo_array(self._cuda_device)

        self.registered_buffers.append(self.colors_gpu)
        self.registered_buffers.append(self.vertices_gpu)

        face_coords = self.grid.grid_coordinates(self.vertices_gpu.tensor.cpu().numpy()[::6])

        for i in range(3):
            # noinspection PyUnresolvedReferences
            if bool((face_coords[:, i] == self.grid.segmentation[i]).any()) is True:
                face_coords[:, i] -= 1

        self.face_group_ids = (face_coords[:, 0]
                               + self.grid.segmentation[0] * face_coords[:, 1]
                               + self.grid.segmentation[0] * self.grid.segmentation[1] * face_coords[:, 2])

        assert len(np.unique(self.face_group_ids) == len(face_coords))
        assert np.max(self.face_group_ids) == (len(face_coords) - 1)

        self._init_txt_visual(self.G_flags, self.G_flags_cache, self.G_flags_texts, self.G_flags_text_visual)
        self._init_txt_visual(self.G_props, self.G_props_cache, self.G_props_texts, self.G_props_text_visual)
        self.group_ids = torch.arange(self.grid.config.G)
        return

    def set_group_id_text(self, key):
        self.group_id_text_visual.text = self.group_id_texts[key]

    def _set_text(self, key, property_tensor: PropertyTensor, cache: dict, text_collection: dict, visual: TextVisual):
        if key != 'None':
            cached = cache[key]
            # noinspection PyTypeChecker
            current = getattr(property_tensor, key)
            # noinspection PyTypeChecker
            diff: torch.Tensor = cached != current
            if diff.any():
                current = current.cpu().numpy()
                txt: list[str] = text_collection[key]
                for i in self.group_ids[diff.cpu()]:
                    txt[i] = str(current[i])

        visual.text = text_collection[key]

    def set_g_flags_text(self, key):
        self._set_text(key, self.G_flags, self.G_flags_cache, self.G_flags_texts, self.G_flags_text_visual)

    def set_g_props_text(self, key):
        self._set_text(key, self.G_props, self.G_props_cache, self.G_props_texts, self.G_props_text_visual)


# noinspection PyAbstractClass
class SelectedGroups(GroupBoxes):

    def __init__(self, network_config: NetworkConfig, grid: NetworkGrid,
                 connect, color=(0.1, 1., 1., 1.)):

        self._output_planes: GroupMeshVisual = GroupMeshVisual(
            network_config, grid, '+z', grid.output_grid_coord)
        self._sensory_input_planes: GroupMeshVisual = GroupMeshVisual(
            network_config, grid, '-y', grid.sensory_grid_coord)

        super().__init__(network_config=network_config, grid=grid,
                         connect=connect, color=color,
                         meshes=[self._sensory_input_planes, self._output_planes])

        self.unfreeze()
        self._input_color_array: Optional[RegisteredVBO] = None
        self._output_color_array: Optional[RegisteredVBO] = None
        self.freeze()

    def init_cuda_arrays(self):
        self._input_color_array = self._sensory_input_planes.face_color_array(self._cuda_device)
        self.registered_buffers.append(self._input_color_array)
        self._input_color_array.tensor[:, 3] = .5
        self._output_color_array = self._output_planes.face_color_array(self._cuda_device)
        self._output_color_array.tensor[:, 3] = .5
        self.registered_buffers.append(self._output_color_array)

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, G_flags: LocationGroupFlags, G_props: LocationGroupProperties):
        super().init_cuda_attributes(device, G_flags=G_flags, G_props=G_props)
        self.G_props.input_face_colors = self._input_color_array.tensor
        self.G_props.output_face_colors = self._output_color_array.tensor


# noinspection PyAbstractClass
class IOGroups(RenderedCudaObjectNode):

    count: int = 0

    def __init__(self, pos,
                 data: np.array,
                 network,
                 compatible_groups: np.array,
                 state_colors_attr: str,
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

        self.network_config: NetworkConfig = network.network_config
        self.grid: NetworkGrid = network.grid
        self.network = network
        pos = np.vstack((pos, np.array([0., 0., self.network_config.max_z + 1])))
        init_pos = pos[0]
        self.pos = pos - init_pos
        self._shape = (segmentation[0] * self.grid.unit_shape[0],
                       segmentation[1] * self.grid.unit_shape[1],
                       segmentation[2] * self.grid.unit_shape[2]) if unit_shape is None else unit_shape
        self._segment_shape = (self._shape[0]/segmentation[0],
                               self._shape[1]/segmentation[1],
                               self._shape[2]/segmentation[2])
        self.compatible_groups = compatible_groups

        self.data_shape = np.zeros(segmentation).squeeze().shape
        if data.shape != self.data_shape:
            data = data.squeeze()
            if data.shape != self.data_shape:
                raise ValueError
        self.data_cpu = data.astype(np.int32)
        if data_color_coding is None:
            data_color_coding = np.array([
                [0., 0., 0., .6],
                [1., 1., 1., .6],
            ])
        if len(np.unique(data[data != -1.])) != len(data_color_coding):
            raise ValueError
        self.data_color_coding_cpu = data_color_coding.astype(np.float32)
        self.segmentation = segmentation
        self.n_cells = len(self.pos[: -1]) * self.segmentation[0] * self.segmentation[1] * self.segmentation[2]
        self.collision_shape = (self.n_cells, len(self.compatible_groups))

        center = np.array([self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2])

        self.pos[:-1] -= center

        self.name = name or f'{self.__class__.__name__}{InputGroups.count}'

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
        # dirs = ('x', 'y', 'z')
        move[i] = .1
        if '-' in face_dir:
            move *= -1
        self.transform.move(init_pos)
        self.transform.move(move)
        self.unfreeze()
        j = (i + 1) % 3
        j_segments = np.unique(self.grid.grid_coord[self.compatible_groups][:, j])
        self.n_j_segments = len(j_segments)
        k = (i + 2) % 3
        k_segments = np.unique(self.grid.grid_coord[self.compatible_groups][:, k])
        self.n_k_segments = len(k_segments)
        self.normals = InteractiveBoxNormals(self, self.shape)

        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / max(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)

        self.G_flags: Optional[LocationGroupFlags] = None
        self.G_props: Optional[LocationGroupProperties] = None

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
        self._colors_attr: str = state_colors_attr
        self.colors_gpu: Optional[torch.Tensor] = None

        self.freeze()

        self.normals.transform = self.transform

    @property
    def parent(self):
        return super().parent

    @parent.setter
    def parent(self, value):
        super(RenderedCudaObjectNode, self.__class__).parent.fset(self, value)
        for o in self.normals:
            o.parent = value

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
        self.G_flags.selection_flag = None
        self.normals.visible = v
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
        self._neuron_groups_shape_gpu = torch.from_numpy(np.array(self.grid.unit_shape,
                                                                  dtype=np.float32)).to(self._cuda_device)

        self._collision_tensor_gpu = torch.zeros(self.collision_shape, dtype=torch.float32, device=self._cuda_device)
        group_pos = np.repeat(self.grid.pos[self.compatible_groups]
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
                                                      dtype=torch.int32, device=self._cuda_device)
        self.data_gpu = torch.from_numpy(self.data_cpu).to(device=self._cuda_device)
        self.data_color_coding_gpu = torch.from_numpy(self.data_color_coding_cpu).to(self._cuda_device)

        n_sens_gr = len(self.compatible_groups)
        self.face_color_indices_gpu = torch.arange(n_sens_gr * 2 * 3,
                                                   device=self._cuda_device).reshape((self.n_k_segments,
                                                                                      self.n_j_segments,
                                                                                      2, 3))
        self.face_color_indices_gpu = self.face_color_indices_gpu.flip(0).transpose(0, 1).reshape(n_sens_gr, 2, 3)
        self.scale_gpu = torch.from_numpy(np.array(self.transform.scale)).to(self._cuda_device)
        self.colors_gpu = getattr(self.G_props, self._colors_attr)
        self.transform_changed()

    def actualize_colors(self):
        # self.colors_gpu[:, 1] = 1.
        self.colors_gpu[:, 3] = 0.
        for i in range(len(self.data_color_coding_cpu)):
            indices = self.face_color_indices_gpu[self.io_neuron_group_values_gpu == i].flatten()
            self.colors_gpu[indices, :] = self.data_color_coding_gpu[i]
            # print(f"indices ({self.io_neuron_group_indices_gpu}) -> color ({self.data_color_coding_gpu[i]})")

    # noinspection PyMethodOverriding
    def init_cuda_attributes(self, device, G_flags: LocationGroupFlags, G_props: LocationGroupProperties):
        self.G_flags: LocationGroupFlags = G_flags
        self.G_props: LocationGroupProperties = G_props
        super().init_cuda_attributes(device)
        self.transform_connected = True
        self.normals.visible = False

        for normal in self.normals:
            self.registered_buffers.append(normal.gpu_array)


# noinspection PyAbstractClass
class InputGroups(IOGroups):

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
        self.G_flags.sensory_input_type[:] = -1
        self.G_flags.sensory_input_type[self.network_config.sensory_groups] = self.io_neuron_group_values_gpu
        mask = self.G_flags.sensory_input_type != -1
        self.G_flags.selected = mask
        self.G_flags.b_thalamic_input = 0
        self.G_flags.b_sensory_input = torch.where(mask, 1, 0)
        self.actualize_colors()

        self.network.GPU.actualize_plot_map(
            self.network_config.sensory_groups[self.io_neuron_group_values_gpu.cpu() != -1])

        self.network.GPU.plotting_arrays.voltage.map()
        self.network.GPU.plotting_arrays.firings.map()

    @property
    def src_weight(self):
        return self._src_weight

    @src_weight.setter
    def src_weight(self, value):
        self._src_weight = value
        self.network.GPU.set_src_group_weights(self.network_config.sensory_groups, value)


# noinspection PyAbstractClass
class OutputGroups(IOGroups):

    def transform_changed(self):

        prev_act_output_grs = self.network.GPU.active_output_groups

        self.assign_sensory_input()
        self.G_flags.output_type[:] = -1
        self.G_flags.output_type[self.network_config.output_groups] = self.io_neuron_group_values_gpu
        mask = self.G_flags.output_type != -1
        self.G_flags.selected = mask
        self.G_flags.b_output_group = torch.where(mask, 1, 0)

        active_output_groups = self.network.GPU.active_output_groups
        # noinspection PyUnresolvedReferences
        if ((self.network.GPU.active_output_groups is None) or
                (len(active_output_groups) != len(prev_act_output_grs)) or
                (not bool((active_output_groups == prev_act_output_grs).all()))):
            self.network.GPU.set_active_output_groups(active_output_groups)

        self.actualize_colors()

        self.network.GPU.actualize_plot_map(
            self.network_config.output_groups[self.io_neuron_group_values_gpu.cpu() != -1])

        self.network.GPU.plotting_arrays.voltage.map()
        self.network.GPU.plotting_arrays.firings.map()
