from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from vispy.scene import visuals
from vispy.gloo.context import get_current_canvas
from vispy.visuals.transforms import STTransform

from .network_config import (
    NetworkConfig
)
# from .network_structures import (
#     NeuronTypeGroup
# )

from rendered_object import RenderedObject, Scale, Position


class NetworkScatterPlot(RenderedObject):

    def __init__(self, config: NetworkConfig):

        super().__init__()

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        self._obj.set_data(config.pos,
                           face_color=(1, 1, 1, .3),
                           edge_color=(0, 0, 0, .5),
                           size=7, edge_width=1)
        # noinspection PyTypeChecker
        self._obj.set_gl_state('translucent', blend=True, depth_test=True)

        self._obj.name = 'Neurons'
        self._shape = config.N_pos_shape

        self.scale = Scale(self._obj.transform)
        
    @property
    def pos_vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id


def default_cube_transform(edge_lengths):
    return STTransform(translate=[edge_lengths[0] / 2, edge_lengths[1] / 2, edge_lengths[2] / 2])


class DefaultBox(visuals.Box):

    def __init__(self, shape: tuple,
                 segments: tuple = (1, 1, 1),
                 translate=None,
                 scale=None,
                 edge_color='white'):

        if translate is None:
            translate = (shape[0] / 2, shape[1] / 2, shape[2] / 2)
        super().__init__(width=shape[0],
                         height=shape[2],
                         depth=shape[1],
                         color=None,
                         # color=(0.5, 0.5, 1, 0.5),
                         width_segments=segments[0],  # X/RED
                         height_segments=segments[2],  # Y/Blue
                         depth_segments=segments[1],  # Z/Green
                         edge_color=edge_color)
        self.transform = STTransform(translate=translate, scale=scale)


# noinspection PyAbstractClass
class SelectorBox(RenderedObject):

    count: int = 0

    def __init__(self, grid_unit_shape, name=None):
        super().__init__()
        self._obj: visuals.Box = DefaultBox(shape=grid_unit_shape,
                                            edge_color='orange', scale=[1.01, 1.01, 1.01])
        self._obj.name = name or f'{self.__class__.__name__}{SelectorBox.count}'
        SelectorBox.count += 1
        self._shape = grid_unit_shape

        self.scale = Scale(self._obj.transform, obj=self._obj)
        self.pos = Position(self._obj.transform)


def plot_pos(n_plots, plot_length):

    pos = np.empty((n_plots * plot_length, 2), np.float32)

    x = np.linspace(0, plot_length - 1, plot_length)
    y = np.linspace(0.5, n_plots - 0.5, n_plots)
    # noinspection PyUnresolvedReferences
    pos[:, 0] = np.meshgrid(x, y)[0].flatten()
    # print('Generating points...')
    # pos[:, 1] = np.random.normal(scale=.025, loc=.3, size=N)
    pos[:, 1] = y.repeat(plot_length)
    return pos


def pos_color(size):
    color = np.ones((size, 4), dtype=np.float32)
    color[:, 0] = np.linspace(0, 1, size)
    color[:, 1] = color[::-1, 0]
    return color


class VoltagePlot(RenderedObject):
    
    def __init__(self, n_plots, plot_length):
        
        super().__init__()

        connect = np.ones(plot_length).astype(bool)
        connect[-1] = False
        connect = connect.reshape(1, plot_length).repeat(n_plots, axis=0).flatten()

        self._obj: visuals.Line = visuals.Line(pos=plot_pos(n_plots, plot_length),
                                               color=pos_color(n_plots * plot_length),
                                               connect=connect,
                                               antialias=False, width=1)
        # line = visuals.Line(pos=pos, color=color, connect='strip', antialias=True, method='agg')
        self._obj.transform = STTransform()

    @property
    def pos_vbo_glir_id(self):
        return self._obj._line_visual._pos_vbo.id
        # return self._obj._line_visual._vbo.id


class FiringScatterPlot(RenderedObject):

    def __init__(self, n_plots, plot_length):
        super().__init__()

        pos = plot_pos(n_plots, plot_length)
        color = pos_color(n_plots * plot_length)
        color[:, 3] = 0

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers()
        # noinspection PyTypeChecker
        self._obj.set_data(pos,
                           face_color=color,
                           edge_color=(1, 1, 1, 1),
                           size=3, edge_width=0)
        # noinspection PyTypeChecker
        self._obj.set_gl_state('translucent', blend=True, depth_test=True)

    @property
    def pos_vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id
