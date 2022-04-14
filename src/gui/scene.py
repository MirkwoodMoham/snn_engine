from dataclasses import asdict, dataclass
from typing import Union, Optional

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.scene.cameras import PanZoomCamera

from network import SpikingNeuronNetwork


@dataclass
class CanvasConfig:
    title: str = 'VisPy canvas'
    size: tuple = (1600, 1200)
    position: Optional[tuple] = None
    show: bool = False
    autoswap: bool = True

    create_native: bool = True
    vsync: bool = False
    resizable: bool = True
    decorate: bool = True
    fullscreen: bool = False
    config: Optional[dict] = None
    shared = Optional[Union[Canvas, GLContext]]
    keys: Optional[Union[str, dict]] = 'interactive'
    parent: Optional = None
    dpi: Optional[float] = None
    always_on_top: bool = False
    px_scale: int = 1
    bgcolor: Union[str, Color] = 'black'


class EngineSceneCanvas(scene.SceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 network: SpikingNeuronNetwork):

        conf = conf or CanvasConfig()
        super().__init__(**asdict(conf), app=app)

        self.unfreeze()
        self.n_voltage_plots = network.plotting_config.n_voltage_plots
        self.voltage_plot_length = network.plotting_config.voltage_plot_length
        self.n_scatter_plots = network.plotting_config.n_scatter_plots
        self.scatter_plot_length = network.plotting_config.scatter_plot_length

        # self._central_view = None
        grid: scene.widgets.Grid = self.central_widget.add_grid()
        # row_span = 6
        self.network_view = grid.add_view(row=0, col=0)

        self.grid: scene.widgets.Grid = self.network_view.add_grid()

        # self.network_view = vispy.scene.ViewBox()

        # time_text.pos = 100, 100

        self.network_view.camera = 'turntable'  # or try 'arcball'
        # add a colored 3D axis for orientation
        axis = scene.visuals.XYZAxis(parent=self.network_view.scene)
        axis.transform = STTransform()
        axis.transform.move((-0.1, -0.1, -0.1))

        plot_col = 1
        text_grid: scene.widgets.Grid = self.grid.add_grid(row=0, col=5, row_span=2, col_span=1,
                                                           border_color='w')

        time_txt = scene.Label('t', color='white')
        time_txt.border_color = 'w'

        self.time_txt2 = scene.Label('0', color='white')
        self.time_txt2.border_color = 'w'
        time_txt.height_min = 100
        time_txt.height_max = 100
        # text_grid.height_min = 30
        # text_grid.height_max = 30
        text_grid.width_min = 150
        text_grid.width_max = 150

        text_grid.add_widget(time_txt, row=0, col=0, row_span=1)
        text_grid.add_widget(self.time_txt2, row=0, col=1, row_span=1)

        self.voltage_plot_view = self._voltage_plot_view(row=0, col=plot_col)
        self.voltage_plot_view.width_min = 450
        self.voltage_plot_view.width_max = 600
        self.scatter_plot_view = self._scatter_plot_view(row=3, col=plot_col)

        self.freeze()

        if network is not None:
            self.add_to_network_view(network.neurons.obj)
            self.add_to_network_view(network.outer_grid)
            self.add_to_network_view(network.selector_box.obj)
            self.add_voltage_plot(network)
            self.add_firing_scatter_plot(network)
            self.add_to_network_view(network.selected_group_boxes.obj)

    @property
    def _window_id(self):
        # noinspection PyProtectedMember
        return self._backend._id

    def add_to_network_view(self, obj):
        obj.parent = self.network_view.scene
        self.network_view.add(obj)

    def add_voltage_plot(self, network: SpikingNeuronNetwork):
        line: scene.visuals.Box = network.voltage_plot.obj
        line.parent = self.voltage_plot_view.scene
        self.voltage_plot_view.add(line)

    def add_firing_scatter_plot(self, network: SpikingNeuronNetwork):
        line: scene.visuals.Box = network.firing_scatter_plot.obj
        line.parent = self.scatter_plot_view.scene
        self.scatter_plot_view.add(line)

    # noinspection PyTypeChecker
    def _plot_view(self,
                   row, col, title_str, n_plots, plot_length, cam_yscale=1,
                   height_min=None, height_max=None):

        title = scene.Label(title_str, color='white')
        title.height_min = 30
        title.height_max = 30

        yoffset = 0.05 * n_plots
        y_axis = scene.AxisWidget(orientation='left',
                                  # domain=(-yoffset, n_plots + yoffset)
                                  )
        y_axis.stretch = (0.12, 1)
        y_axis.width_min = 50
        y_axis.width_max = y_axis.width_min

        xoffset = 0.05 * plot_length
        x_axis = scene.AxisWidget(orientation='bottom',
                                  # domain=(-xoffset, plot_length + xoffset)
                                  )
        x_axis.stretch = (1, 0.15)
        x_axis.height_min = 20
        x_axis.height_max = 30

        self.grid.add_widget(title, row=row, col=col, col_span=1)
        self.grid.add_widget(y_axis, row=row + 1, col=col - 1, row_span=1)
        self.grid.add_widget(x_axis, row=row + 1 + 1, col=col, row_span=1)

        # v.camera = PanZoomCamera((x_axis.axis.domain[0],
        #                           y_axis.axis.domain[0] * cam_yscale,
        #                           x_axis.axis.domain[1] + xoffset,
        #                           (y_axis.axis.domain[1] + yoffset) * cam_yscale))
        v = self.grid.add_view(row=row + 1, col=col, border_color='w', row_span=1)
        if height_min is not None:
            v.height_min = height_min
        if height_max is not None:
            v.height_max = height_max

        scene.visuals.GridLines(parent=v.scene)
        v.camera = PanZoomCamera((-xoffset,
                                  -yoffset * cam_yscale,
                                  plot_length + xoffset + xoffset,
                                  (n_plots + yoffset + yoffset) * cam_yscale))
        x_axis.link_view(v)
        y_axis.link_view(v)
        # v.camera.set_range()
        return v

    # noinspection PyTypeChecker
    def _voltage_plot_view(self, row, col):
        return self._plot_view(row=row, col=col, title_str="Voltage Plot",
                               n_plots=self.n_voltage_plots, plot_length=self.voltage_plot_length,
                               cam_yscale=1, height_min=350)

    # noinspection PyTypeChecker
    def _scatter_plot_view(self, row, col):
        return self._plot_view(row=row, col=col, title_str="Scatter Plot",
                               n_plots=self.n_scatter_plots, plot_length=self.scatter_plot_length, cam_yscale=1,
                               height_min=200, height_max=500)

    def set_keys(self, keys):
        self.unfreeze()
        # noinspection PyProtectedMember
        self._set_keys(keys)
        self.freeze()
