import numpy as np
from vispy.scene import visuals
# from vispy.visuals.transforms import STTransform

from rendering import RenderedObject


def plot_pos(n_plots, plot_length):
    pos = np.empty((n_plots * plot_length, 2), np.float32)

    x = np.linspace(0, plot_length - 1, plot_length)
    y = np.linspace(0.5, n_plots - 0.5, n_plots)
    # noinspection PyUnresolvedReferences
    pos[:, 0] = np.meshgrid(x, y)[0].flatten()
    pos[:, 1] = y.repeat(plot_length)
    return pos


def pos_color(size):
    color = np.ones((size, 4), dtype=np.float32)
    color[:, 0] = np.linspace(0, 1, size)
    color[:, 1] = color[::-1, 0]
    return color


# noinspection PyAbstractClass
class VoltagePlot(RenderedObject):

    def __init__(self, n_plots, plot_length):
        super().__init__()

        connect = np.ones(plot_length).astype(bool)
        connect[-1] = False
        connect = connect.reshape(1, plot_length).repeat(n_plots, axis=0).flatten()

        self._obj: visuals.Line = visuals.Line(pos=plot_pos(n_plots, plot_length),
                                               color=pos_color(n_plots * plot_length),
                                               connect=connect,
                                               antialias=False, width=1, parent=self)
        # line = visuals.Line(pos=pos, color=color, connect='strip', antialias=True, method='agg')
        # self._obj.transform = STTransform()

    @property
    def vbo_glir_id(self):
        return self._obj._line_visual._pos_vbo.id


# noinspection PyAbstractClass
class FiringScatterPlot(RenderedObject):

    def __init__(self, n_plots, plot_length):
        super().__init__()

        pos = plot_pos(n_plots, plot_length)
        color = pos_color(n_plots * plot_length)
        color[:, 3] = 0

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers(parent=self)
        # noinspection PyTypeChecker
        self._obj.set_data(pos,
                           face_color=color,
                           edge_color=(1, 1, 1, 1),
                           size=3, edge_width=0)
        # noinspection PyTypeChecker
        self._obj.set_gl_state('translucent', blend=True, depth_test=True)

    @property
    def vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id
