import numpy as np
from vispy.scene import visuals
# from vispy.visuals.transforms import STTransform

from rendering import RenderedObjectNode
from vispy.gloo.context import get_current_canvas


class PlotData:

    def __init__(self, n_plots, plot_length, n_groups=9):

        self._n_plots = n_plots
        self._plot_length = plot_length

        mesh_ = np.meshgrid(np.linspace(0, self._plot_length - 1, self._plot_length),
                            np.linspace(0.5, self._n_plots - 0.5, self._n_plots))
        # noinspection PyUnresolvedReferences
        self.pos = np.vstack([mesh_[0].ravel(), mesh_[1].ravel()]).T

        size = self._plot_length * self._n_plots
        self.color = np.ones((size, 4), dtype=np.float32)
        self.color[:, 0] = np.linspace(0, 1, size)
        self.color[:, 1] = self.color[::-1, 0]

        self.group_separators_pos = np.zeros((n_groups * 2 * 2, 2))
        self.group_separators_color = np.ones((n_groups * 2 * 2, 4), dtype=np.float32)
        self.group_separators_color[:, 3] = 0
        self.group_separators_pos[:, 0] = (np.expand_dims(np.array([-2, plot_length]), 0)
                                           .repeat(n_groups * 2, 0)).flatten()
        self.group_separators_pos[:, 1] = np.linspace(0, self._n_plots, n_groups * 2).repeat(2)


# noinspection PyAbstractClass
class BasePlot:

    def __init__(self, n_plots, plot_length, n_groups, group_separator_width):

        self.plot_data = PlotData(n_plots, plot_length, n_groups=n_groups)

        self.group_separator_lines = visuals.Line(pos=self.plot_data.group_separators_pos,
                                                  color=self.plot_data.group_separators_color,
                                                  connect='segments', width=group_separator_width)

    @property
    def group_lines_pos_vbo_glir_id(self):
        return self.group_separator_lines._line_visual._pos_vbo.id

    @property
    def group_lines_color_vbo_glir_id(self):
        return self.group_separator_lines._line_visual._color_vbo.id

    @property
    def group_lines_pos_vbo(self):
        return RenderedObjectNode.buffer_id(self.group_lines_pos_vbo_glir_id)

    @property
    def group_lines_color_vbo(self):
        return RenderedObjectNode.buffer_id(self.group_lines_color_vbo_glir_id)


# noinspection PyAbstractClass
class VoltagePlot(RenderedObjectNode, BasePlot):

    def __init__(self, n_plots, plot_length, n_groups):

        BasePlot.__init__(self, n_plots, plot_length, n_groups, 1)

        connect = np.ones(plot_length).astype(bool)
        connect[-1] = False
        connect = connect.reshape(1, plot_length).repeat(n_plots, axis=0).flatten()

        self._obj: visuals.Line = visuals.Line(pos=self.plot_data.pos,
                                               color=self.plot_data.color,
                                               connect=connect,
                                               antialias=False, width=1, parent=None)

        RenderedObjectNode.__init__(self, [self._obj, self.group_separator_lines])

    @property
    def vbo_glir_id(self):
        return self._obj._line_visual._pos_vbo.id


# noinspection PyAbstractClass
class FiringScatterPlot(RenderedObjectNode, BasePlot):

    def __init__(self, n_plots, plot_length, n_groups):

        BasePlot.__init__(self, n_plots, plot_length, n_groups, 1)

        self.plot_data.color[:, 3] = 0

        self._obj: visuals.visuals.MarkersVisual = visuals.Markers(parent=None)
        self._obj.set_data(self.plot_data.pos,
                           face_color=self.plot_data.color,
                           edge_color=(1, 1, 1, 1),
                           size=3, edge_width=0)
        # noinspection PyTypeChecker
        self._obj.set_gl_state('translucent', blend=True, depth_test=True)

        RenderedObjectNode.__init__(self, [self.group_separator_lines, self._obj])

    @property
    def vbo_glir_id(self):
        # noinspection PyProtectedMember
        return self._obj._vbo.id

