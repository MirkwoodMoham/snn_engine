from copy import copy
from dataclasses import asdict, dataclass
from typing import Union, Optional
import numpy as np

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.scene.cameras import PanZoomCamera
from vispy.util import keys

from network import SpikingNeuronNetwork
from rendering import RenderedObject


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
        self.network = network
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

        self.time_txt2 = None
        self.update_duration_value_txt = None

        plot_col = 1
        self.info_grid_right()

        self.voltage_plot_view = self._voltage_plot_view(row=0, col=plot_col)
        self.voltage_plot_view.width_min = 450
        self.voltage_plot_view.width_max = 600
        self.scatter_plot_view = self._scatter_plot_view(row=3, col=plot_col)

        self._clicked_obj = None
        self._selected_objects = []
        self._last_selected_obj = None

        self._click_pos = np.zeros(2)
        self._last_mouse_pos = np.zeros(2)

        self.mouse_pressed = True

        self.grid_transform = self.scene.node_transform(self.grid)

        self.freeze()

        if network is not None:
            network.add_rendered_objects(self.network_view, self.voltage_plot_view, self.scatter_plot_view)

            # self._select(network.selector_box, True)
            # self._selected_objects.append(network.selector_box)

    def info_grid_right(self):
        # noinspection PyTypeChecker
        text_grid: scene.widgets.Grid = self.grid.add_grid(row=0, col=5, row_span=2, col_span=1,
                                                           border_color='w')
        # noinspection PyTypeChecker
        time_txt = scene.Label('t', color='white')
        time_txt.border_color = 'w'
        # noinspection PyTypeChecker
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

        update_duration_text = scene.Label('update\nduration', color='white')
        update_duration_text.height_max = 100
        update_duration_text.border_color = 'w'
        self.update_duration_value_txt = scene.Label('0', color='white')
        self.update_duration_value_txt.border_color = 'w'

        text_grid.add_widget(update_duration_text, row=1, col=0, row_span=1)
        text_grid.add_widget(self.update_duration_value_txt, row=1, col=1, row_span=1)


    @property
    def _window_id(self):
        # noinspection PyProtectedMember
        return self._backend._id

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


