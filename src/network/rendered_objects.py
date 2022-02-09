import numpy as np
from typing import Mapping, Optional, Union
from vispy.scene import visuals
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform

from .network_config import (
    NetworkConfig
)
from .network_structures import (
    NeuronTypeGroup
)


class RenderedObject:

    _grid_unit_shape: Optional[tuple] = (1, 1, 1)

    def __init__(self):

        self._obj: Optional[Union[visuals.visuals.MarkersVisual]] = None

        self._pos_vbo = None
        self._ebo = None
        self._parent = None

        self._shape = None

        self._grid_coordinates = np.zeros(3)

    def __call__(self):
        return self._obj

    @property
    def name(self):
        try:
            return self._obj.name
        except AttributeError:
            return self.__str__()

    @property
    def shape(self):
        return self._shape

    @property
    def pos_vbo_glir_id(self):
        raise NotImplementedError

    # noinspection PyProtectedMember
    @property
    def pos_vbo(self):
        if self._pos_vbo is None:
            self._pos_vbo = get_current_canvas().context.shared.parser._objects[self.pos_vbo_glir_id].handle
        return self._pos_vbo

    def _move(self, i, d=1):
        tr = np.zeros(3)
        tr[i] += d * self._grid_unit_shape[i]

        self._obj.transform.move(tr)
        self._grid_coordinates[i] += 1 * d

        print(f'MOVE {self.name}:',
              # tr,
              '\n', self._grid_coordinates)

    def mv_left(self):
        self._move(0)

    def mv_right(self):
        self._move(0, -1)

    def mv_fw(self):
        self._move(1, -1)

    def mv_bw(self):
        self._move(1)

    def mv_up(self):
        self._move(2)

    def mv_down(self):
        self._move(2, -1)


class NetworkScatterPlot(RenderedObject):

    def __init__(self, config: NetworkConfig):

        super().__init__()

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        self._obj.set_data(config.pos,
                           face_color=(1, 1, 1, .3),
                           edge_color=(0, 0, 0, .5),
                           size=7, edge_width=1)
        self._obj.set_gl_state('translucent', blend=True, depth_test=True)

        self._obj.name = 'Neurons'
        self._shape = config.N_pos_shape
        
    @property
    def pos_vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id


def default_cube_transform(edge_lengths):
    return STTransform(translate=[edge_lengths[0] / 2, edge_lengths[1] / 2, edge_lengths[2] / 2])


def default_box(shape: tuple,
                segments: tuple = (1, 1, 1),
                translate=None,
                scale=None,
                edge_color='white'):
    if translate is None:
        translate = [shape[0] / 2, shape[1] / 2, shape[2] / 2]
    cube = visuals.Box(width=shape[0],
                       height=shape[2],
                       depth=shape[1],
                       color=None,
                       # color=(0.5, 0.5, 1, 0.5),
                       width_segments=segments[0],  # X/RED
                       height_segments=segments[2],  # Y/Blue
                       depth_segments=segments[1],  # Z/Green
                       edge_color=edge_color)

    cube.transform = STTransform(translate=translate, scale=scale)
    return cube


# noinspection PyAbstractClass
class SelectorBox(RenderedObject):

    count: int = 0

    def __init__(self, grid_unit_shape, name=None):
        super().__init__()
        self._obj: visuals.Box = default_box(shape=grid_unit_shape,
                                             edge_color='orange', scale=[1.01, 1.01, 1.01])
        self._obj.name = name or f'SelectorBox{SelectorBox.count}'
        SelectorBox.count += 1
        self._shape = grid_unit_shape
        

class VoltagePlot(RenderedObject):
    
    def __init__(self):
        
        super().__init__()
        N = 10
        pos = np.empty((N, 2), np.float32)
        pos[:, 0] = np.linspace(.05, .95, N)

        color = np.ones((N, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, N)
        color[:, 1] = color[::-1, 0]

        lines = []

        # print('Generating points...')
        pos[:, 1] = np.random.normal(scale=.025, loc=.3, size=N)

        print('voltage pos:\n', pos, '\n')

        line = visuals.Line(pos=pos, color=color)
        lines.append(line)
        line.transform = STTransform()
        
        self._obj: visuals.Line = line

    @property
    def pos_vbo_glir_id(self):
        return self._obj._line_visual._pos_vbo.id
    