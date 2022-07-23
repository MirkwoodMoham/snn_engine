from dataclasses import asdict
import numpy as np
from vispy.scene import ColorBarWidget

from app.content.widgets.plot_widget import AxisVisualConfig, CustomLinkedAxisWidget, PlotWidget
from network import PlottingConfig


class VoltagePlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None,
                 height_min=100, height_max=None):

        super().__init__(title="Voltage [V]",
                         plot_height=plotting_confing.n_voltage_plots,
                         plot_length=plotting_confing.voltage_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class ScatterPlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None, height_min=100, height_max=None):

        super().__init__(title="Firings",
                         plot_height=plotting_confing.n_scatter_plots,
                         plot_length=plotting_confing.scatter_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class GroupFiringsPlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=200, width_max=None, height_min=100, height_max=None):

        super().__init__(title="Group Firings",
                         plot_height=plotting_confing.G,
                         plot_length=plotting_confing.scatter_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max)


class SingleNeuronPlotWidget(PlotWidget):

    def __init__(self, plotting_confing: PlottingConfig,
                 width_min=300, width_max=300, height_min=190, height_max=190):

        x_axis_config = AxisVisualConfig(tick_label_margin=12)
        y_axis_config = AxisVisualConfig(scale=20, tick_label_margin=6)
        y_axis_right_config = AxisVisualConfig(scale=200, tick_label_margin=6)

        y_axis_width = 10

        super().__init__(title=None,
                         plot_height=1,
                         plot_length=plotting_confing.voltage_plot_length,
                         width_min=width_min, width_max=width_max,
                         height_min=height_min, height_max=height_max,
                         y_axis_width_min=y_axis_width + 25,
                         y_axis_width_max=y_axis_width + 25,
                         x_axis_height_min=15,
                         x_axis_height_max=15,
                         x_axis_config=x_axis_config,
                         y_axis_config=y_axis_config,
                         )
        self.unfreeze()
        self.y_axis_right = CustomLinkedAxisWidget(orientation='right', **asdict(y_axis_right_config))
        self.y_axis_right.stretch = (0.12, 1)
        self.y_axis_right.width_min = y_axis_width
        self.y_axis_right.width_max = y_axis_width
        self.grid.add_widget(self.y_axis_right, row=0, col=2, row_span=1)
        self.y_axis_right.link_view(self.view)
        # noinspection PyUnresolvedReferences
        self.view.camera.rect = (
            self.view.camera.rect.pos[0],
            -.5,
            self.view.camera.rect.size[0],
            self.view.camera.rect.size[1] + .5
        )
        self.view.camera.set_default_state()
        self.cam_reset()
        self.freeze()

    def cam_reset(self):
        super().cam_reset()
        self.y_axis_right._view_changed()


class GroupInfoColorBar(ColorBarWidget):

    def __init__(self):
        super().__init__(clim=(0, 99), border_color='white',
                         cmap="cool", orientation="right", border_width=1, label_color='white')

    @property
    def cmap(self):
        return self._colorbar._colorbar.cmap

    @cmap.setter
    def cmap(self, v):
        self._colorbar._colorbar.cmap = v
        print(v.map(np.array([0.5])))
        self._colorbar._colorbar._update()
