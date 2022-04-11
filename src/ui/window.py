from dataclasses import asdict, dataclass
from typing import Optional, Union

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QAction

from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget
)

from vispy.app import Application, Canvas
from vispy.color import Color
from vispy.gloo.context import GLContext
from vispy import scene
from vispy.scene.cameras import PanZoomCamera

from network import SpikingNeuronNetwork
from .ui_element import ButtonMenuAction, SpinBoxSlider


@dataclass
class ButtonMenuActions:

    window: Optional[QtWidgets.QMainWindow] = None

    START_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Start Simulation',
                                                          menu_short_cut='F9',
                                                          status_tip='Start Simulation',
                                                          icon_name='control.png')

    PAUSE_SIMULATION: ButtonMenuAction = ButtonMenuAction(menu_name='&Pause Simulation',
                                                          menu_short_cut='F10',
                                                          status_tip='Pause Simulation',
                                                          icon_name='control-pause.png',
                                                          disabled=True)

    EXIT_APP: ButtonMenuAction = ButtonMenuAction(menu_name='&Exit',
                                                  name='Exit',
                                                  status_tip='Close Application',
                                                  menu_short_cut='Ctrl+Q')

    TOGGLE_OUTERGRID: ButtonMenuAction = ButtonMenuAction(menu_name='&Outergrid',
                                                          name='Show OuterGrid',
                                                          status_tip='Show/Hide OuterGrid',
                                                          menu_short_cut='Ctrl+G', checkable=True)

    def __post_init__(self):
        window_ = self.window
        self.window = None
        dct = asdict(self)
        self.window = window_
        print()
        for k in dct:
            v = getattr(self, k)
            if isinstance(v, ButtonMenuAction):
                if (v.menu_short_cut is not None) and (v.status_tip is not None):
                    v.status_tip = v.status_tip + f" ({v.menu_short_cut})"
                    print(v.status_tip)
                if v.window is None:
                    v.window = self.window


class UI(object):

    class MenuBar:
        class MenuActions:
            def __init__(self):
                self.start: QAction = ButtonMenuActions.START_SIMULATION.action()
                self.pause: QAction = ButtonMenuActions.PAUSE_SIMULATION.action()
                self.toggle_outergrid: QAction = ButtonMenuActions.TOGGLE_OUTERGRID.action()
                self.exit: QAction = ButtonMenuActions.EXIT_APP.action()
                self.exit.triggered.connect(QApplication.instance().quit)

        def __init__(self, window):
            self.actions = self.MenuActions()

            self.bar = QtWidgets.QMenuBar(window)

            self.bar.setGeometry(QtCore.QRect(0, 0, 440, 130))
            self.bar.setObjectName("menubar")

            self.file_menu = self.bar.addMenu('&File')
            self.file_menu.addAction(self.actions.start)
            self.file_menu.addAction(self.actions.pause)
            self.file_menu.addAction(self.actions.exit)

            self.view_menu = self.bar.addMenu('&View')
            self.view_menu.addAction(self.actions.toggle_outergrid)

    class UiLeft:

        class Buttons:
            def __init__(self):
                max_width = 140
                self.start: QPushButton = ButtonMenuActions.START_SIMULATION.button()
                self.pause: QPushButton = ButtonMenuActions.PAUSE_SIMULATION.button()
                self.exit: QPushButton = ButtonMenuActions.EXIT_APP.button()
                self.toggle_outergrid: QPushButton = ButtonMenuActions.TOGGLE_OUTERGRID.button()

                self.toggle_outergrid.setMinimumWidth(max_width)
                self.toggle_outergrid.setMaximumWidth(max_width)
                self.start.setMaximumWidth(max_width)
                self.exit.setMaximumWidth(max_width)

                self.exit.clicked.connect(QApplication.instance().quit)

        class Sliders:
            def __init__(self, window):

                self.thalamic_inh_input_current = SpinBoxSlider(name='Th. Inh. Input [I]',
                                                                window=window,
                                                                status_tip='Thalamic Inhibitory Input Current [I]',
                                                                prop_id='thalamic_inh_input_current',
                                                                min_value=0, max_value=10000)
                self.thalamic_exc_input_current = SpinBoxSlider(name='Th. Exc. Current [I]',
                                                                window=window,
                                                                status_tip='Thalamic Excitatory Input Current [I]',
                                                                prop_id='thalamic_exc_input_current',
                                                                min_value=0, max_value=10000)

        def __init__(self, window, central_widget):
            self.frame = QFrame(central_widget)
            self.buttons = self.Buttons()
            self.sliders = self.Sliders(window)
            self.layout = QVBoxLayout(self.frame)

            # self.vbox.addStretch(1)
            self.layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            # splitter.setSizes([125, 150])
            # hbox.addStretch(1)
            play_pause_widget = QWidget(central_widget)
            play_pause_hbox = QHBoxLayout(play_pause_widget)
            play_pause_hbox.addWidget(self.buttons.start)
            play_pause_hbox.addWidget(self.buttons.pause)
            self.layout.addWidget(play_pause_widget)
            self.layout.addWidget(self.buttons.toggle_outergrid)
            self.layout.addWidget(self.sliders.thalamic_inh_input_current.widget)
            self.layout.addWidget(self.sliders.thalamic_exc_input_current.widget)
            self.layout.addWidget(self.buttons.exit)
            # self.layout.width_max = 100

    def __init__(self, window):

        self.ui_elements = ButtonMenuActions(window)

        self.menubar = self.MenuBar(window)
        window.setMenuBar(self.menubar.bar)

        self.central_widget = QWidget(window)

        self.frame_3d = QFrame(self.central_widget)
        self.frame_3d.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3d.setFrameShadow(QFrame.Shadow.Raised)
        self.layout_3d = QVBoxLayout(self.frame_3d)
        # self.grid_layout.addWidget(self.frame_3d, 0, 1, 1, 30)

        self.ui_left = self.UiLeft(window, self.central_widget)
        
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.ui_left.frame)
        splitter.addWidget(self.frame_3d)
        splitter.setStretchFactor(1, 30)

        hbox = QHBoxLayout(self.central_widget)
        hbox.addWidget(splitter)

        window.setCentralWidget(self.central_widget)

        window.setStatusBar(QStatusBar(window))

        self.retranslate_ui(window)

    @staticmethod
    def retranslate_ui(window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("SNN Engine", "SNN Engine"))


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


class EngineWindow(QtWidgets.QMainWindow):

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
            scene.visuals.XYZAxis(parent=self.network_view.scene)

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
                self.add_scatter_plot(network)
                self.add_outer_box(network)
                self.add_selector_box(network)
                self.add_voltage_plot(network)
                self.add_firing_scatter_plot(network)

        @property
        def _window_id(self):
            # noinspection PyProtectedMember
            return self._backend._id

        def add_scatter_plot(self, network: SpikingNeuronNetwork):
            plot = network.scatter_plot()
            plot.parent = self.network_view.scene
            self.network_view.add(plot)

        def add_outer_box(self, network: SpikingNeuronNetwork):
            box: scene.visuals.Box = network.outer_grid
            box.parent = self.network_view.scene
            self.network_view.add(box)

        def add_selector_box(self, network: SpikingNeuronNetwork):
            box: scene.visuals.Box = network.selector_box()
            box.parent = self.network_view.scene
            self.network_view.add(box)

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

    def __init__(self,
                 name: str,
                 app: Optional[Application],
                 network: SpikingNeuronNetwork,
                 show=True,
                 keys=None
                 ):
        super().__init__()

        for attr in ['ui', 'scene', 'main_view']:
            if hasattr(self, attr):
                raise AttributeError(f'\'{attr}\' ')

        self.setObjectName(name)
        self.resize(800, 600)

        # self.central_scene = self.set_central_scene(keys=keys, app=app)
        self.scene_3d = self.EngineSceneCanvas(conf=CanvasConfig(keys=keys), app=app, network=network)

        self.ui = UI(self)
        self.ui.layout_3d.addWidget(self.scene_3d.native)

        if show is True:
            self.show()

    def set_keys(self, keys):
        self.scene_3d.set_keys(keys)
