import numpy as np
from typing import Optional, Union

from vispy.scene import visuals, Node
from vispy.visuals.transforms import STTransform

from gpu import (
    GPUArrayConfig,
    RegisteredGPUArray
)

from .rendered_cuda_object import RenderedCudaObjectNode, CudaObject
from .objects import Box
from .rendered_object import RenderedObjectNode, RenderedObject


class ArrowVisual(visuals.Tube, CudaObject):

    def __init__(self, points, color=None, name=None, parent: Optional[Node] = None,
                 tube_points=4, radius=np.array([.01, .01, .025, .0])):

        # self.unfreeze()
        self._points = points
        self._tube_points = tube_points
        # self._node_parent = _node_parent
        # self.freeze()

        if color is None:
            if points[:, 0].any():
                color = np.array([1, 0, 0, 0.3], dtype=np.float32)
            elif points[:, 1].any():
                color = np.array([0, 1, 0, 0.3], dtype=np.float32)
            else:
                color = np.array([0, 0, 1, 0.3], dtype=np.float32)
        name = name or parent.name
        if (points[:, 0] > 0).any():
            name += ':x+'
        elif (points[:, 0] < 0).any():
            name += ':x-'
        elif (points[:, 1] > 0).any():
            name += ':y+'
        elif (points[:, 1] < 0).any():
            name += ':y-'
        elif (points[:, 2] > 0).any():
            name += ':z+'
        else:
            name += ':z-'

        # vertex_colors = np.repeat(np.array([color]), self.n_vertices, axis=0)

        visuals.Tube.__init__(self, name=name, points=points, tube_points=tube_points, radius=radius,
                              color=color,
                              # vertex_colors=vertex_colors,
                              parent=parent)
        # self.name = name
        # self.transform: STTransform = parent.transform
        # self.interactive = True

        CudaObject.__init__(self)


# noinspection PyAbstractClass
class NormalArrow(RenderedCudaObjectNode):

    def __init__(self, select_parent, points, color=None, name=None, tube_points=4,
                 radius=np.array([.01, .01, .025, .0]), parent: Optional[Node] = None,
                 selectable=True):
        # super().__init__(parent=parent, selectable=selectable,
        #                  # name=name or parent.name + f'.{self.__class__.__name__}'
        #                  )
        # self.transform: STTransform = STTransform()
        self.select_parent = select_parent
        # super().__init__(parent=parent, selectable=selectable)

        self._obj = ArrowVisual(points=points,
                                name=name or parent.name + f'.{self.__class__.__name__}',
                                parent=None,
                                tube_points=tube_points, radius=radius, color=color)
        super().__init__([self._obj], parent=parent, selectable=selectable)
        # self._obj.parent = self
        self.interactive = True
        # self.unfreeze()

        # self._obj = ArrowVisual(self, points=points, name=name, parent=self,
        #                         tube_points=tube_points, radius=radius, color=color)
        # self._obj.interactive = True
        # self.freeze()


    def on_select_callback(self, v):
        print(f'selected arrow({v}):', self)

        print(self.gpu_array.tensor[:, 3][:6], '...')

        # b=self.gpu_array.tensor
        # print(b[:, 3][:6], '...')

        # self.gpu_array.map()
        self.gpu_array.tensor[:, 3] = 1. if v is True else .3
        # self.gpu_array.unmap()
        print(self.gpu_array.tensor[:, 3][:6], '...')
        # b[:, 3] = 1 if v is True else 0.

        # self.obj.color = 'white'
        # self.obj._update_data()
        # print(b[:, 3][:6], '...')
        # b = self.gpu_array
        # print(b.tensor[:, 3][:6], '...')
        # self.update()
        # self._obj.update()
        # self.gpu_array.unmap()

    @property
    def pos_vbo_glir_id(self):
        return self._obj._vertices.id

    @property
    def color_vbo_glir_id(self):
        return self._obj.shared_program.vert['base_color'].id

    @property
    def gpu_array(self):
        if self._gpu_array is None:
            nbytes = 4
            shape = (self.obj._meshdata.n_faces * 3, 4)
            print('shape:', shape)
            b = RegisteredGPUArray.from_buffer(
                self.color_vbo, config=GPUArrayConfig(shape=shape, strides=(shape[1] * nbytes, nbytes),
                                                      dtype=np.float32, device=self._cuda_device))
            # return b
            self._gpu_array = b

        return self._gpu_array


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
                arrow = NormalArrow(select_parent, points=inv[i], parent=self)
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

