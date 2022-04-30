import numpy as np
from typing import Optional, Union

from vispy.scene import visuals, Node
# from vispy.visuals.transforms import STTransform

from gpu import (
    GPUArrayConfig,
    RegisteredGPUArray
)

from .rendered_cuda_object import RenderedCudaObjectNode, CudaObject
from .objects import Box
# from .rendered_object import RenderedObjectNode, RenderedObject


class ArrowVisual(visuals.Tube, CudaObject):

    def __init__(self, points, color=None, name=None, parent: Optional[Node] = None,
                 tube_points=4, radius=np.array([.01, .01, .025, .0])):

        self._points = points
        self._tube_points = tube_points

        visuals.Tube.__init__(self, name=name, points=points, tube_points=tube_points, radius=radius,
                              color=color,
                              parent=parent)
        CudaObject.__init__(self)


# noinspection PyAbstractClass
class NormalArrow(RenderedCudaObjectNode):

    def __init__(self, select_parent, points, color=None, name=None, tube_points=4,
                 radius=np.array([.01, .01, .025, .0]), parent: Optional[Node] = None,
                 selectable=True, draggable=True):
        # super().__init__(parent=parent, selectable=selectable,
        #                  # name=name or parent.name + f'.{self.__class__.__name__}'
        #                  )
        # self.transform: STTransform = STTransform()

        self.last_scale = None

        self.select_parent = select_parent
        # super().__init__(parent=parent, selectable=selectable)

        self.scaled_extraction_dir = 1
        for i, d in enumerate(['x', 'y', 'z']):
            if (points[:, i] != 0).any():
                self.scaled_extraction_dim = d
                if (points[:, i] < 0).any():
                    self.scaled_extraction_dir = -1
        self.modifier_dim = 0
        if self.scaled_extraction_dim == 'z':
            self.modifier_dim = 1
            self.scaled_extraction_dir *= -1

        if name is None:
            name = (f"{self._select_parent.name}.{self.__class__.__name__}:{self.scaled_extraction_dim}"
                    f"{'+' if self.scaled_extraction_dir > 0 else '-'}")

        if color is None:
            if points[:, 0].any():
                color = np.array([1, 0, 0, 0.3], dtype=np.float32)
            elif points[:, 1].any():
                color = np.array([0, 1, 0, 0.3], dtype=np.float32)
            else:
                color = np.array([0, 0, 1, 0.3], dtype=np.float32)

        self._visual = ArrowVisual(points=points,
                                   name=name + '.obj',
                                   parent=None,
                                   tube_points=tube_points, radius=radius, color=color)

        super().__init__([self._visual], parent=parent, selectable=selectable, name=name, draggable=draggable)
        self.interactive = True

    def on_select_callback(self, v):
        print(f'\nselected arrow({v}):', self, '\n')
        # print(self.gpu_array.tensor[:, 3][:6], '...')
        self.gpu_array.tensor[:, 3] = 1. if v is True else .3

        self.last_scale = getattr(self.select_parent.scale, self.scaled_extraction_dim)
        print('last_scale:', self.last_scale)
        # print(self.gpu_array.tensor[:, 3][:6], '...')

    def on_drag_callback(self, v: np.ndarray):
        v = v[self.modifier_dim] * self.scaled_extraction_dir
        print(f'\ndragged arrow({v}):', self, '\n')

        # self.select_parent.scale.x += v * self.scale_extraction_factor
        # current_scale = getattr(self.select_parent.scale, 'x')
        setattr(self.select_parent.scale, self.scaled_extraction_dim, self.last_scale + v)
        self.actualize_ui()

    def actualize_ui(self):
        getattr(self.select_parent.scale.spin_box_sliders, self.scaled_extraction_dim).actualize_values()

    @property
    def pos_vbo_glir_id(self):
        return self._visual._vertices.id

    @property
    def color_vbo_glir_id(self):
        return self._visual.shared_program.vert['base_color'].id

    def init_cuda_arrays(self):
        nbytes = 4
        shape = (self.visual._meshdata.n_faces * 3, 4)
        # print('shape:', shape)
        b = RegisteredGPUArray.from_buffer(
            self.color_vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes, nbytes),
                                                  dtype=np.float32, device=self.cuda_device))
        # return b
        self._gpu_array = b


class CudaBox(Box, CudaObject):

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 select_parent,
                 shape: tuple,
                 segments: tuple = (1, 1, 1),
                 translate=None,
                 scale=None,
                 color: Optional[Union[str, tuple]] = None,
                 edge_color: Union[str, tuple] = 'white',
                 name: str = None,
                 depth_test=True, border_width=1, parent=None,
                 init_normals=True):

        # self._parent = parent
        Box.__init__(self, shape=shape,
                     segments=segments,
                     scale=scale,
                     translate=translate,
                     name=name,
                     color=color,
                     edge_color=edge_color,
                     depth_test=depth_test,
                     border_width=border_width,
                     parent=parent)

        if init_normals:
            assert segments == (1, 1, 1)
            self.normals = []
            inv = self.initial_normal_vertices(shape)
            for i in range(6):
                arrow = NormalArrow(select_parent, points=inv[i])
                # self.add_subvisual(arrow)
                self.normals.append(arrow)

        CudaObject.__init__(self)
        # self.interactive = True

    @staticmethod
    def initial_normal_vertices(shape):
        # isv = self._initial_selection_vertices

        points = np.zeros((6, 4, 3), dtype=np.float32)

        x0 = shape[0]/2
        y0 = shape[1]/2
        z0 = shape[2]/2

        points[0] = np.array([[x0, 0, 0], [x0 + x0/2, 0, 0], [x0 + x0/2, 0, 0], [x0 + 2 * x0/3, 0, 0]])
        points[1] = -1 * points[0]
        points[2] = np.array([[0, y0, 0], [0, y0 + y0/2, 0], [0, y0 + y0/2, 0], [0, y0 + 2 * y0/3, 0]])
        points[3] = -1 * points[2]
        points[4] = np.array([[0, 0, z0], [0, 0, z0 + z0/2], [0, 0, z0 + z0/2], [0, 0, z0 + 2 * z0/3]])
        points[5] = -1 * points[4]

        return points
