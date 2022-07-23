import numpy as np
from typing import Optional
from vispy.scene import visuals
# from vispy.visuals.transforms import STTransform

from rendering import RenderedObjectNode, RenderedCudaObjectNode, CudaLine
from gpu import RegisteredVBO


class PlotData:

    def __init__(self, n_plots, plot_length, n_group_seperator_lines=0, group_separator_line_offset_left=0):

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

        if n_group_seperator_lines > 0:

            self.group_separators_pos = np.zeros((n_group_seperator_lines * 2, 2))
            self.group_separators_color = np.ones((n_group_seperator_lines * 2, 4), dtype=np.float32)
            self.group_separators_color[:, 3] = 0
            self.group_separators_pos[:, 0] = (
                np.expand_dims(np.array([-group_separator_line_offset_left, plot_length]), 0)
                .repeat(n_group_seperator_lines, 0)).flatten()
            self.group_separators_pos[:, 1] = np.linspace(
                0, self._n_plots, n_group_seperator_lines).repeat(2)


# noinspection PyAbstractClass
class BaseMultiPlot:

    def __init__(self, n_plots, plot_length, n_group_separator_lines, group_line_offset_left,
                 group_separator_line_width=1):

        self.plot_data = PlotData(n_plots, plot_length, n_group_separator_lines, group_line_offset_left)

        self.group_separator_lines = visuals.Line(pos=self.plot_data.group_separators_pos,
                                                  color=self.plot_data.group_separators_color,
                                                  connect='segments', width=group_separator_line_width)

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
class VoltagePlot(RenderedObjectNode, BaseMultiPlot):

    def __init__(self, n_plots, plot_length, n_group_separator_lines):

        BaseMultiPlot.__init__(self, n_plots, plot_length, n_group_separator_lines, 2)

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
class FiringScatterPlot(RenderedObjectNode, BaseMultiPlot):

    def __init__(self, n_plots, plot_length, n_group_separator_lines):

        BaseMultiPlot.__init__(self, n_plots, plot_length, n_group_separator_lines, 20)

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


# noinspection PyAbstractClass
class GroupFiringCountsPlot(RenderedObjectNode, BaseMultiPlot):

    def __init__(self, n_plots, plot_length, n_groups, color=None):

        BaseMultiPlot.__init__(self, n_plots, plot_length, n_groups + 1, 2)

        if color is not None:
            if isinstance(color, list):
                for i, c in enumerate(color):
                    color_arr = np.repeat(np.array(c).reshape(1, 4), plot_length, 0)
                    self.plot_data.color[i * plot_length: (i+1) * plot_length, :] = color_arr
            else:
                raise NotImplementedError

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
class SingleNeuronPlot(RenderedCudaObjectNode):

    def __init__(self, plot_length):

        plot_data = PlotData(2, plot_length)

        colors = [[1., 1., 0., 1.], [0., 1., 1., 1.]]

        for i, c in enumerate(colors):
            color_arr = np.repeat(np.array(c).reshape(1, 4), plot_length, 0)
            plot_data.color[i * plot_length: (i + 1) * plot_length, :] = color_arr

        connect = np.ones(plot_length).astype(bool)
        connect[-1] = False
        connect = connect.reshape(1, plot_length).repeat(2, axis=0).flatten()

        self.line: CudaLine = CudaLine(pos=plot_data.pos,
                                       color=plot_data.color,
                                       connect=connect,
                                       antialias=False, width=1, parent=None)

        RenderedCudaObjectNode.__init__(self, [self.line])

    def init_cuda_attributes(self, device):
        super().init_cuda_attributes(device)
        self.registered_buffers += self.line.registered_buffers


# noinspection PyAbstractClass
class PhasePortrait(RenderedCudaObjectNode):

    def __init__(self, plot_length):

        self.plot_data = PlotData(1, plot_length)

        self.line: CudaLine = CudaLine(pos=self.plot_data.pos,
                                       color=self.plot_data.color,
                                       antialias=False, width=1, parent=None)
        self._v_null_cline_data = PlotData(1, plot_length)
        self._v_null_cline: CudaLine = CudaLine(pos=self._v_null_cline_data.pos,
                                                color=self._v_null_cline_data.color,
                                                connect='segments',
                                                antialias=False, width=1, parent=None)

        self._u_null_cline_data = PlotData(1, plot_length)
        self._u_null_cline: CudaLine = CudaLine(pos=self._u_null_cline_data.pos,
                                                color=self._u_null_cline_data.color,
                                                connect='segments',
                                                antialias=False, width=1, parent=None)

        RenderedCudaObjectNode.__init__(self, [self.line, self._v_null_cline, self._u_null_cline])

