from copy import copy
from dataclasses import asdict, dataclass
from typing import Union, Optional
import numpy as np

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.util import keys
from vispy.scene.widgets import Widget

from .plot_widgets import (
    GroupInfoColorBar,
    GroupFiringsPlotWidget,
    PlotWidget,
    VoltagePlotWidget,
    ScatterPlotWidget)

from network import SpikingNeuronNetwork
from rendering import RenderedObject
from network import PlottingConfig


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


class TextTableWidget(Widget):

    def __init__(self, labels: list[str], heights_min=None, heights_max=None,
                 height_min_global=None, height_max_global=None):

        super().__init__()

        self.unfreeze()
        self.item_count = 0
        self.grid = self.add_grid()
        width = 130
        self.width_min = width
        self.width_max = width

        if height_min_global is None:
            generate_height_min_global = True
            height_min_global = 0
            if height_max_global is None:
                height_min_default = 25
            else:
                height_min_default = int(height_max_global / len(labels))
        else:
            generate_height_min_global = False
            height_min_default = int(height_min_global / len(labels))

        if height_max_global is None:
            generate_height_max_global = True
            height_max_global = 0
            height_max_default = 25
        else:
            generate_height_max_global = False
            height_max_default = int(height_max_global / len(labels)) + 1

        for i, label in enumerate(labels):
            height_min = heights_min[i] if heights_min is not None else height_min_default
            f = label.count('_')
            height_min += f * height_min_default
            height_max = heights_max[i] if heights_max is not None else height_max_default
            height_max += f * height_max_default
            if generate_height_min_global is True:
                height_min_global += height_min
            if generate_height_max_global is True:
                height_max_global += height_max
            self.add_label(label, height_min=height_min, height_max=height_max)

        self.grid.height_min = (min(height_min_global, height_max_global)
                                if generate_height_min_global is True else height_min_global)
        self.grid.height_max = height_max_global

        # self.height_max = height_max_global
        # if width_min_global is not None:
        #     self.grid.width_min = width_min_global

        self.freeze()

    # noinspection PyTypeChecker
    def add_label(self, label_name, initial_value='0', height_min=28, height_max=28):
        font_size = 9
        label = scene.Label(label_name.replace('_', '\n'), color='white', font_size=font_size)
        label.border_color = 'w'
        label_value = scene.Label(initial_value, color='white', font_size=font_size)
        label_value.border_color = 'w'
        label.height_min = height_min
        label.height_max = height_max
        label_value.height_max = height_max
        self.grid.add_widget(label, row=self.item_count, col=0)
        self.grid.add_widget(label_value, row=self.item_count, col=1)
        self.item_count += 1
        setattr(self, label_name.replace('\n', '_'), label_value)


class BaseEngineSceneCanvas(scene.SceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application]):

        conf = conf or CanvasConfig()
        super().__init__(**asdict(conf), app=app)


class MainSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app,
                 plotting_config: PlottingConfig):

        super().__init__(conf, app)

        self.unfreeze()

        self.network: SpikingNeuronNetwork = app.network

        self.n_voltage_plots = plotting_config.n_voltage_plots
        self.voltage_plot_length = plotting_config.voltage_plot_length
        self.n_scatter_plots = plotting_config.n_scatter_plots
        self.scatter_plot_length = plotting_config.scatter_plot_length

        main_grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.network_view = main_grid.add_view(row=0, col=0)

        self.central_widget.margin = 10

        self.grid: scene.widgets.Grid = self.network_view.add_grid()

        self.network_view.camera = 'turntable'
        axis = scene.visuals.XYZAxis(parent=self.network_view.scene)
        axis.transform = STTransform()
        axis.transform.move((-0.1, -0.1, -0.1))

        row_span_0 = 2
        col_span0 = 2
        plot_col0 = 0
        plot_col1 = 4
        plot_col2 = plot_col1 + col_span0
        plot_row1 = row_span_0
        row_span10 = 2
        row_span_11 = 1
        height_min0 = 350
        height_max1 = 500

        self.table = TextTableWidget(labels=['t', 'update_duration'])
        # self.info_grid_right(row=0, col=plot_col2 + 1, row_span=row_span_0, height_min=height_min0)
        self.grid.add_widget(self.table, 0, plot_col2+1)
        if plotting_config.windowed_neuron_plots is False:
            self.voltage_plot = VoltagePlotWidget(plotting_confing=plotting_config,
                                                  width_max=600, height_min=height_min0)

            self.scatter_plot = ScatterPlotWidget(plotting_confing=plotting_config,
                                                  width_max=600, height_max=height_max1)

            self.grid.add_widget(self.voltage_plot, 0, plot_col0, row_span=row_span_0, col_span=col_span0)
            self.grid.add_widget(self.scatter_plot, plot_row1, plot_col0, row_span=row_span10, col_span=col_span0)
        else:
            self.voltage_plot = None
            self.scatter_plot = None

        if plotting_config.group_info_view_mode.scene is True:
            self.group_firings_plot = GroupFiringsPlotWidget(plotting_confing=plotting_config)
            self.grid.add_widget(self.group_firings_plot, plot_row1, plot_col1, col_span=col_span0, row_span=row_span10)
        else:
            self.group_firings_plot = None

        self.group_firings_plot_single0 = PlotWidget(
            title_str="Group Firings: XXX",
            n_plots=1, plot_length=self.scatter_plot_length,
            cam_yscale=1)

        self.group_firings_plot_single1 = PlotWidget(
            title_str="Group Firings: YYY",
            n_plots=1, plot_length=self.scatter_plot_length)

        self.grid.add_widget(self.group_firings_plot_single0, plot_row1, plot_col2,
                             col_span=col_span0, row_span=row_span_11)
        self.grid.add_widget(self.group_firings_plot_single1, plot_row1 + row_span_11, plot_col2,
                             col_span=2, row_span=row_span_11)

        # self.group_firings_plot = None
        # self.group_firings_plot_single0 = None
        # self.group_firings_plot_single1 = None

        self._clicked_obj = None
        self._selected_objects = []
        self._last_selected_obj = None

        self._click_pos = np.zeros(2)
        self._last_mouse_pos = np.zeros(2)

        self.mouse_pressed = True

        self.grid_transform = self.scene.node_transform(self.grid)

        self.freeze()

    @property
    def _window_id(self):
        # noinspection PyProtectedMember
        return self._backend._id

    def set_keys(self, keys_):
        self.unfreeze()
        # noinspection PyProtectedMember
        self._set_keys(keys_)
        self.freeze()

    def mouse_pos(self, event):
        return self.grid_transform.map(event.pos)[:2]

    def _select_clicked_obj(self):

        if self._clicked_obj is not self._last_selected_obj:

            self.network.GPU._N_pos_edge_color[:, 3] = 0.05
            self.network.GPU._N_pos_face_color[:, 3] = 0.05
            self.network._neurons.set_gl_state(depth_test=False)

            for o in copy(self._selected_objects):
                if ((not (o is self._clicked_obj))
                        and ((self._clicked_obj is None) or (not o.is_select_child(self._clicked_obj)))):
                    self._select(o, False)

            if isinstance(self._clicked_obj, RenderedObject):
                if self._clicked_obj.selected:
                    # self.clicked_obj.on_select_callback(self.clicked_obj.selected)
                    self._clicked_obj.update()
                    self._last_selected_obj = self._clicked_obj
                else:
                    # print('\nSELECTED:', self._clicked_obj)
                    self._select(self._clicked_obj, True)

    def on_mouse_press(self, event):

        if event.button == 1:
            self.network_view.camera.interactive = False
            self.network_view.interactive = False
            self._clicked_obj = self.visual_at(event.pos)
            # print('\nCLICKED:', self._clicked_obj)
            self.network_view.interactive = True
            self._click_pos[:2] = self.mouse_pos(event)

            if isinstance(self._clicked_obj, RenderedObject) and self._clicked_obj.draggable:
                self._select_clicked_obj()

    def _mouse_moved(self, event):
        self._last_mouse_pos[:2] = self.mouse_pos(event)
        return (self._last_mouse_pos[:2] - self._click_pos[:2]).any()

    def _select(self, obj: RenderedObject, v: bool):
        obj.select(v)
        if v is True:
            self._selected_objects.append(obj)
            self._last_selected_obj = obj
        else:
            self._selected_objects.remove(obj)
            if obj is self._last_selected_obj:
                self._last_selected_obj = None
        return obj

    def on_mouse_release(self, event):
        self.network_view.camera.interactive = True
        if event.button == 1:
            if (not self._mouse_moved(event)) or (self._clicked_obj is self.visual_at(event.pos)):
                self._select_clicked_obj()
            if isinstance(self._last_selected_obj, RenderedObject) and self._last_selected_obj.draggable:
                self._select(self._last_selected_obj, False).update()
                self._last_selected_obj = self._selected_objects[-1]
            # self._last_selected_obj = None

            print(f'currently selected ({len(self._selected_objects)}):', self._selected_objects)
            if len(self._selected_objects) == 0:
                self.network.GPU._N_pos_face_color[:, 3] = 0.3
                self.network.GPU._N_pos_edge_color[:, 3] = 0.5
                self.network._neurons.set_gl_state(depth_test=True)

    def on_mouse_move(self, event):
        self.network_view.camera.interactive = True
        if event.button == 1:
            if isinstance(self._clicked_obj, RenderedObject) and self._clicked_obj.draggable:
                # print(keys.SHIFT in event.modifiers)
                self.network_view.camera.interactive = False
                self._last_mouse_pos[:2] = self.mouse_pos(event)
                # dist = np.linalg.norm(self._last_mouse_pos - self._click_pos)
                diff = self._last_mouse_pos - self._click_pos
                # print('diff:', diff)
                if keys.SHIFT in event.modifiers:
                    mode = 0
                elif keys.CONTROL in event.modifiers:
                    mode = 1
                else:
                    mode = 2
                self._clicked_obj.on_drag_callback(diff/100, mode=mode)
                # self._clicked_obj.


class VoltagePlotSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 plotting_config: PlottingConfig):

        super().__init__(conf, app)

        self.unfreeze()
        self.n_voltage_plots = plotting_config.n_voltage_plots
        self.voltage_plot_length = plotting_config.voltage_plot_length

        main_grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.central_widget.margin = 10

        self.plot = VoltagePlotWidget(plotting_confing=plotting_config, height_min=200)
        main_grid.add_widget(self.plot, row=0, row_span=9, col_span=4)
        self.table = TextTableWidget(labels=['t'], height_max_global=25)
        self.table.height_max = 25
        main_grid.add_widget(self.table, 0, 3)


class ScatterPlotSceneCanvas(BaseEngineSceneCanvas):

    # noinspection PyTypeChecker
    def __init__(self,
                 conf: CanvasConfig,
                 app: Optional[Application],
                 plotting_config: PlottingConfig):

        super().__init__(conf, app)

        self.unfreeze()
        self.n_scatter_plots = plotting_config.n_scatter_plots
        self.scatter_plot_length = plotting_config.scatter_plot_length

        grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.central_widget.margin = 10

        self.plot = ScatterPlotWidget(plotting_confing=plotting_config, height_min=200)

        grid.add_widget(self.plot, row=0, row_span=9, col_span=4)

        self.table = TextTableWidget(labels=['t'], height_max_global=25)
        self.table.height_max = 25
        grid.add_widget(self.table, 0, 3)


class LocationGroupInfoCanvas(BaseEngineSceneCanvas):

    def __init__(self, conf: CanvasConfig, app: Optional[Application], plotting_config: PlottingConfig):

        super().__init__(conf, app)
        self.unfreeze()
        main_grid: scene.widgets.Grid = self.central_widget.add_grid()
        self.view = main_grid.add_view(row=0, col=0)
        self.view.camera = 'turntable'
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis.transform = STTransform()
        axis.transform.move((-0.1, -0.1, -0.1))

        self.grid = self.view.add_grid()

        self.table = TextTableWidget(labels=['t'], height_max_global=25)
        self.table.height_max = 25

        self.grid.add_widget(self.table, 0, 6)

        self.group_firings_plot = GroupFiringsPlotWidget(plotting_confing=plotting_config, width_max=600)

        self.grid.add_widget(self.group_firings_plot, 5, 5, col_span=2, row_span=6)

        self.color_bar = GroupInfoColorBar()

        self.grid.add_widget(self.color_bar, 5, 0, 6, 1)

        self.freeze()
