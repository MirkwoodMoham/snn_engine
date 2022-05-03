import numpy as np
from numpy.lib import recfunctions as rfn
import torch
from typing import Optional

from vispy.geometry import create_box
from vispy.visuals import MeshVisual
from vispy.visuals.transforms import STTransform

from network.network_config import NetworkConfig
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
        # self._obj.parent = self
        self.unfreeze()
        # self._ghost: Box = Box(name=self.name + '.ghost',
        #                        shape=self.shape,
        #                        translate=(0, 0, 0),
        #                        color=(0, 0.1, 0.2, 0.51),
        #                        edge_color=(0, 0.1, 0.2, 0.95),
        #                        scale=[0.51, 2.1, 1.1],
        #                        depth_test=False,
        #                        border_width=2,
        #                        parent=self,
        #                        use_parent_transform=False)
        # self.some_arrow = ArrowVisual(self, points=self._obj.normals[0].obj._points + np.array([.5, .5, .5]),
        #                               name=name, parent=self,
        #                               color='white')

        self.transform = STTransform()
        self.transform.translate = (self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2)
        self.transform.scale = [1.1, 1.1, 1.1]

        for normal in self._select_children:
            normal.transform = self.transform

        # add_children(self.obj, self.obj.normals)
        # self._ghost._transform = STTransform()
        # self._ghost.transform.scale = [0.51, 2.1, 1.1]

        self.unfreeze()
        SelectorBox.count += 1
        self.interactive = True
        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / min(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)
        # self.translate = Translate(self)

        self.states_gpu: Optional[LocationGroupProperties] = None

        # noinspection PyPep8Naming
        G = self.network_config.G

        self.selected_masks = np.zeros((G, 4), dtype=np.int32, )

        self.group_numbers = np.arange(G)  # .reshape((G, 1))

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
        self.selected_masks[:, 3] = self.network_config.G
        self.selected_masks[:, 3] = np.where(~self.selected_masks[:, :3].all(axis=1),
                                             self.selected_masks[:, 3], self.group_numbers)

        self.states_gpu.selected = torch.from_numpy(self.selected_masks[:, [3]]).to(self._cuda_device)

    def on_select_callback(self, v: bool):
        self.swap_select_color(v)
        for c in self.visual.normals:
            c.visual.visible = v

    def init_cuda_attributes(self, device, property_tensor):
        super().init_cuda_attributes(device)
        self.states_gpu = property_tensor
        self.transform_connected = True

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
class BoxSystem(RenderedObjectNode):

    def __init__(self, pos, grid_unit_shape, max_z, color=(0.1, 1., 1., 1.), **kwargs):

        self._visual = BoxSystemLineVisual(grid_unit_shape=grid_unit_shape, max_z=max_z, pos=pos, color=color, **kwargs)
        self._visual.transform = STTransform(translate=(0, 0, 0), scale=(1, 1, 1))

        super().__init__([self.visual])

    @property
    def vbo_glir_id(self):
        return self.visual._line_visual._pos_vbo.id

    @property
    def ibo_glir_id(self):
        return self.visual._line_visual._connect_ibo.id


# noinspection PyAbstractClass
class InputCells(RenderedCudaObjectNode):

    count: int = 0

    def __init__(self, pos,
                 network_config: NetworkConfig,
                 unit_shape, max_z,
                 quad_colors,
                 segmentation=(3, 1, 1),
                 color=(1., 1., 1., 1.), name=None, **kwargs):

        if segmentation[1] != 1:
            raise ValueError
        if len(pos) != 2:
            raise ValueError
        self.pos = pos
        self.network_config = network_config

        self._shape = unit_shape
        self.n_input_cells = len(self.pos) * segmentation[0] * segmentation[1] * segmentation[2]
        center = np.array([self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2])

        self._initial_selection_vertices = np.zeros((4, 3))
        self._initial_selection_vertices[0] = self.pos[0]
        self._initial_selection_vertices[1] = self.pos[0] + np.array([self.shape[0], 0, 0])
        self._initial_selection_vertices[2] = self.pos[0] + np.array([0, self.shape[1], 0])
        self._initial_selection_vertices[3] = self.pos[0] + np.array([0, 0, self.shape[2]])
        self._initial_selection_vertices -= center
        self.selected_masks = np.zeros((self.network_config.G, 4), dtype=np.int32, )
        self.group_numbers = np.arange(self.network_config.G)

        self.pos[:-1] -= center

        self.name = name or f'{self.__class__.__name__}{InputCells.count}'

        face_colors, filled_indices, vertices = self.create_mesh(center, quad_colors, segmentation)

        self._mesh = MeshVisual(vertices['position'], filled_indices, None, face_colors, None)
        # self._mesh2 = MeshVisual(vertices2['position'], filled_indices2, None, face_colors2, None)
        # noinspection PyTypeChecker
        self._mesh.set_gl_state(polygon_offset_fill=True,
                                polygon_offset=(1, 1), depth_test=True)
        self._visual = BoxSystemLineVisual(grid_unit_shape=(unit_shape[0]/segmentation[0],
                                                            unit_shape[1]/segmentation[1],
                                                            unit_shape[2]/segmentation[2]),
                                           max_z=max_z, pos=self.pos, color=color, **kwargs)
        super().__init__([self._visual,
                          self._mesh,
                          # self._mesh2
                          ], selectable=True)

        self.interactive = True
        self.transform = STTransform(translate=center, scale=(1, 1, 1))
        # self.transform.translate = (self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2)
        self.unfreeze()

        self.normals = []
        inv = initial_normal_vertices(self.shape)
        for i in range(6):
            arrow = NormalArrow(self, points=inv[i], mod_factor=1 / (3 * self.shape[int(i / 2)]))
            # self.add_subvisual(arrow)
            self.normals.append(arrow)

        self.scale = Scale(self, _min_value=0, _max_value=int(3 * 1 / max(self.shape)))
        self.translate = Translate(self, _grid_unit_shape=self.shape, _min_value=-5, _max_value=5)

        self.states_gpu: Optional[LocationGroupProperties] = None

        self._collision_tensor_gpu = None
        self._input_cell_pos_xx_gpu = None
        self._sensory_group_pos_yy_gpu = None

        self.g_pos = self.network_config.g_pos[:self.network_config.G, :]
        self.g_pos_end = self.network_config.g_pos_end[:self.network_config.G, :]
        self.freeze()

        for normal in self.normals:
            normal.transform = self.transform
            # normal.visible = False

    def create_mesh(self, center, quad_colors, segmentation):
        init_pos = self.pos.copy()
        for i in range(3):
            if segmentation[i] > 1:
                add = np.repeat(np.array([np.linspace(0, self._shape[i], segmentation[i], endpoint=False)]),
                                len(self.pos), axis=0).flatten()
                self.pos = np.repeat(self.pos, segmentation[i], axis=0)
                self.pos[:, i] += add
                self.pos = self.pos[: - segmentation[i] + 1]
        vertices_list = []
        filled_indices_list = []
        for i in range(len(init_pos) - 1):
            vertices_, filled_indices_, _ = create_box(
                self._shape[0], self._shape[2], self._shape[1], segmentation[0], segmentation[2], segmentation[1],
                planes=('+y',))
            vertices_['position'] += center + init_pos[i]
            vertices_list.append(vertices_)
            filled_indices_list.append(filled_indices_)
        vertices = rfn.stack_arrays(vertices_list, usemask=False)
        # filled_indices =
        add = 4 * np.repeat(np.array([np.repeat(np.arange(len(filled_indices_list)),
                                                2 * segmentation[0] * segmentation[2])]), 3, axis=0).T
        filled_indices = np.vstack(filled_indices_list) + add
        if quad_colors is None:
            face_colors = np.repeat(np.array([[1., 1., 1., 1.]]), len(filled_indices), axis=0)
            face_colors[0] = np.array([1., 0., 0., 1.])
            face_colors[-1] = np.array([0., .75, 0., 1.])
        else:
            if len(quad_colors) == 1 and (segmentation != (1, 1, 1)):
                quad_colors = np.repeat(np.array([quad_colors]), len(filled_indices), axis=0)
            face_colors = np.repeat(quad_colors, 2, axis=0)
        return face_colors, filled_indices, vertices

    def on_select_callback(self, v: bool):
        # self.swap_select_color(v)
        for c in self.normals:
            c.visible = v

    def init_gpu_tensors(self):
        shape = (self.n_input_cells, len(self.network_config.sensory_groups))
        self._collision_tensor_gpu = torch.zeros(shape, dtype=torch.float32, device=self._cuda_device)
        sensory_group_pos = np.repeat(self.g_pos[self.network_config.sensory_groups].reshape((1, shape[1], 3)),
                                      self.n_input_cells, axis=0)

        self._sensory_group_pos_yy_gpu = torch.from_numpy(sensory_group_pos).to(self._cuda_device)

        # self._input_cell_pos_xx_gpu = torch.from_numpy(cell_pos).to(self._cuda_device)


    @property
    def selection_vertices(self):
        return (self._initial_selection_vertices
                * self.transform.scale[:3]
                + self.transform.translate[:3])

    def transform_changed(self):
        v = self.selection_vertices
        self.selected_masks[:, 0] = (self.g_pos[:, 0] <= v[1, 0]) & (self.g_pos_end[:, 0] >= v[0, 0])
        self.selected_masks[:, 1] = (self.g_pos[:, 1] <= v[2, 1]) & (self.g_pos_end[:, 1] >= v[0, 1])
        self.selected_masks[:, 2] = (self.g_pos[:, 2] <= v[3, 2]) & (self.g_pos_end[:, 2] >= v[0, 2])
        self.selected_masks[:, 3] = self.network_config.G
        self.selected_masks[:, 3] = np.where(~self.selected_masks[:, :3].all(axis=1)
                                             | ~self.network_config.sensory_group_mask.astype(bool),
                                             self.selected_masks[:, 3], self.group_numbers)
        mask = torch.from_numpy(self.selected_masks[:, [3]]).to(self.cuda_device)
        # self.collision_tensor =
        self.states_gpu.selected = mask

    def init_cuda_attributes(self, device, property_tensor):
        super().init_cuda_attributes(device)
        self.states_gpu: LocationGroupProperties = property_tensor
        self.transform_connected = True
        self.init_gpu_tensors()


