import numpy as np
from vispy import scene
from vispy.scene import ColorBarWidget, PanZoomCamera, Widget

from network import PlottingConfig


class PlotWidget(Widget):

    # noinspection PyTypeChecker
    def __init__(self,
                 title_str, n_plots: int, plot_length: int, cam_yscale: int = 1,
                 width_min=100, width_max=None,
                 height_min=100, height_max=None):

        view_row_span, view_col_span = 1, 1

        super().__init__()

        self.unfreeze()

        row, col = 0, 0
        self.grid = self.add_grid()

        self.title = scene.Label(title_str, color='white')
        self.title.height_min = 30
        self.title.height_max = 30

        yoffset = 0.05 * n_plots
        self.y_axis = scene.AxisWidget(orientation='left')
        self.y_axis.stretch = (0.12, 1)
        self.y_axis.width_min = 50
        self.y_axis.width_max = self.y_axis.width_min

        xoffset = 0.05 * plot_length
        self.x_axis = scene.AxisWidget(orientation='bottom')
        self.x_axis.stretch = (1, 0.15)
        self.x_axis.height_min = 30
        self.x_axis.height_max = 30

        self.grid.add_widget(self.title, row=row, col=col+1, col_span=view_col_span)
        self.grid.add_widget(self.y_axis, row=row + 1, col=col, row_span=view_row_span)
        self.grid.add_widget(self.x_axis, row=row + view_row_span + 1, col=col + 1, row_span=1, col_span=view_col_span)

        self.view = self.grid.add_view(row=row + 1, col=col + 1, border_color='w', row_span=view_row_span,
                                       col_span=view_col_span)
        if width_min is not None:
            self.width_min = width_min
        if width_max is not None:
            self.width_max = width_max
        if height_min is not None:
            self.height_min = height_min
        if height_max is not None:
            self.height_max = height_max

        scene.visuals.GridLines(parent=self.view.scene)

        self.view.camera = PanZoomCamera((-xoffset,
                                          -yoffset * cam_yscale,
                                          plot_length + xoffset + xoffset,
                                          (n_plots + yoffset + yoffset) * cam_yscale))
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.freeze()

    @property
    def visible(self):
        return self.view.visible

    @visible.setter
    def visible(self, value):
        self.view.visible = value
        self.x_axis.visible = value
        self.y_axis.visible = value
        self.title.visible = value

    def add(self, widget):
        self.view.add(widget)


class VoltagePlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None,
                 height_min=100, height_max=None):

        super().__init__(title_str="Voltage Plot",
                         n_plots=plotting_confing.n_voltage_plots,
                         plot_length=plotting_confing.voltage_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class ScatterPlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None, height_min=100, height_max=None):

        super().__init__(title_str="Scatter Plot",
                         n_plots=plotting_confing.n_scatter_plots,
                         plot_length=plotting_confing.scatter_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class GroupFiringsPlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None, height_min=100, height_max=None):

        super().__init__(title_str="Group Firings",
                         n_plots=plotting_confing.G,
                         plot_length=plotting_confing.scatter_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class GroupInfoColorBar(ColorBarWidget):

    def __init__(self):
        super().__init__(label="ColorBarWidget", clim=(0, 99), border_color='white',
                         cmap="cool", orientation="right", border_width=1, label_color='white')

    @property
    def cmap(self):
        return self._colorbar._colorbar.cmap

    @cmap.setter
    def cmap(self, v):
        self._colorbar._colorbar.cmap = v
        print(v.map(np.array([0.5])))
        self._colorbar._colorbar._update()
